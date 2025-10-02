import torch
import torch.nn as nn
import copy
from thop import profile
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementing an early stopping class based on val loss
class EarlyStopping:
    """
    Custom early stopping class based on validation score difference per epoch.
    Saves the best model weights based on validation loss.

    args:
        patience (int): How long to hold out till breaking loop ends.
        delta (float): Tolerable difference btw val loss to increment patience by 1.
        path (string): Path to save best model weights:
    """

    def __init__(self, patience=5, delta=0.0, path="checkpoint.pt"):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss, model):
        score = -val_loss  # lower loss = better

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            print("💾 Saved best model!")

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class MetaOptimizer:
    """
    Layer wise learning rate annealing function.
    """

    def __init__(
        self, model, base_lr=1e-3, decay_factor=0.9, optimizer_class=torch.optim.Adam
    ):
        self.model = model
        self.base_lr = base_lr
        self.decay_factor = decay_factor
        self.optimizer_class = optimizer_class
        self.current_optimizer = None
        self.effective_base_lr = base_lr  # This gets reduced by annealing

    def collect_blocks(self, model):
        """Defines model blocks, creates a param group for the optimizer based on some criteria"""
        groups = []
        layer_id = 0
        total_layer_num = self.count_layers()

        for name, child in model.named_children():
            if isinstance(child, nn.Sequential):
                for _, subchild in child.named_children():
                    if self.has_parameters(subchild):
                        dictionary = {"params": subchild.parameters(), "lr": 0.0}
                        layer_id += 1

                        for m in subchild.modules():
                            if any(p.requires_grad for p in m.parameters()):
                                dictionary["lr"] = round(
                                    self.effective_base_lr
                                    * (
                                        self.decay_factor
                                        ** (total_layer_num - layer_id)
                                    ),
                                    7,
                                )

                        groups.append(dictionary)

            else:
                if self.has_parameters(child):
                    dictionary = {"params": child.parameters(), "lr": 0.0}
                    layer_id += 1
                    for n in child.modules():
                        if any(p.requires_grad for p in n.parameters()):
                            dictionary["lr"] = round(
                                self.effective_base_lr
                                * (self.decay_factor ** (total_layer_num - layer_id)),
                                7,
                            )

                    groups.append(dictionary)

        return groups

    def has_parameters(self, module):
        return len(list(module.parameters())) > 0

    def update_unfrozen_lrs(self):
        """Updates optimizer learning rates |IN PLACE| based on the effective base lr"""
        total_layers = self.count_layers()

        for layer_id, param_group in enumerate(self.current_optimizer.param_groups):
            # Check if this group has any unfrozen params
            if any(p.requires_grad for p in param_group["params"]):
                param_group["lr"] = self.effective_base_lr * (
                    self.decay_factor ** (total_layers - (layer_id + 1))
                )
                layer_id += 1
            else:
                param_group["lr"] = 0.0  # Keep frozen

    def count_layers(self):
        # count parameterized blocks
        total_layer_num = 0
        for name, child in self.model.named_children():
            if isinstance(child, nn.Sequential):
                for _, subchild in child.named_children():
                    if self.has_parameters(subchild):
                        total_layer_num += 1
            else:
                if self.has_parameters(child):
                    total_layer_num += 1

        return total_layer_num

    def anneal_base_lr(self, factor=0.7):
        self.effective_base_lr = (
            self.effective_base_lr * factor
        )  # Reduce the effective base lr
        self.update_unfrozen_lrs()  # Recalculate all LRs with new base

    def initialize_optimizer(self):
        groups = self.collect_blocks(self.model)
        self.current_optimizer = self.optimizer_class(groups)
        return self.current_optimizer


def unfreezeLayers(model, n_blocks):
    """Unfreeze last `n_blocks` layers"""
    layers = (
        [model.features[i] for i in range(len(model.features))]
        + [model.classifier[0]]
        + [model.classifier[3]]
    )  # model architecture dependent

    print(f"DEBUG: unfreezeLayers called with n_blocks={n_blocks}")

    for param in model.parameters():
        param.requires_grad = False

    # Debug the unfreezing loop
    blocks_to_unfreeze = layers[-n_blocks:]

    for block in layers[-n_blocks:]:
        for param in block.parameters():
            param.requires_grad = True


def checkDataLoading(train_loader, val_loader):
    """Test data loading to make sure it works"""
    print("🔍 Testing data loading...")

    # Test train loader
    try:
        batch = next(iter(train_loader))
        inputs, labels = batch
        print(f"✅ Train batch shape: {inputs.shape}, Labels: {labels.shape}")
        print(f"   Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
        print(f"   Label range: [{labels.min()}, {labels.max()}]")
    except Exception as e:
        print(f"❌ Train loader error: {e}")
        return False

    # Test val loader
    try:
        batch = next(iter(val_loader))
        inputs, labels = batch
        print(f"✅ Val batch shape: {inputs.shape}, Labels: {labels.shape}")
    except Exception as e:
        print(f"❌ Val loader error: {e}")
        return False

    return True


def evalModel(model, test_loader, effective_base):
    """
    Evaluate models using the test loader, returns accuracy, classification report, FLOPs, trainable params, confusion matrix in fine print.

    args:
        effective_base (str): Directory to save the classification report png to.
    """
    y_true = []  # ground truth labels
    y_pred = []  # predicted labels

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)

            all_labels.append(labels)
            all_preds.append(preds)

    # concat all batches, we vectorize
    y_true = torch.cat(all_labels).cpu().numpy()
    y_pred = torch.cat(all_preds).cpu().numpy()

    # Accuracy
    print("Accuracy:", accuracy_score(y_true, y_pred))

    # macro averages included in the report
    print(classification_report(y_true, y_pred, digits=4))

    # FLOPs and Param count
    model_copy = copy.deepcopy(model)
    dummy_input = torch.randn(1, 1, 64, 64).to(device)
    flops, params = profile(model_copy, inputs=(dummy_input,))
    print(f"FLOPs: {flops:,}, Number of Params: {params:,}\n")

    # Confustion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(f"{effective_base}/confusion_matrix_mobilenet.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
