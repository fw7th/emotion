from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class PathConfig:
    repo_name: str
    data_dir: str

    def __post_init__(self):
        self.is_colab = "COLAB_GPU" in os.environ

        if self.is_colab:
            self.root = Path("/content")
            self.project = Path(f"/content/drive/MyDrive/{self.repo_name}")
            self.data = Path(f"/content/drive/MyDrive/{self.data_dir}")
            self.cache = Path("/content/cache")
        else:
            self.root = Path.cwd()
            self.project = Path.cwd()
            self.data = Path("./data")
            self.cache = Path("./cache")

    def ensure_dirs(self):
        self.data.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)


# Usage
"""
paths = PathConfig('my-ml-project', 'datasets')
paths.ensure_dirs()
"""
