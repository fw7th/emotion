# Real-Time Multi-Face Emotion Detection with ncnn

CPU-only, real-time emotion recognition built on ncnn, designed for low-power edge and retail analytics.

This project demonstrates how to structure a multi-threaded vision pipeline around ncnn that sustains high frame rates under tight CPU and memory constraints.

<p align="center">
  <img src="assets/github.gif" alt="Demo" width="480">
</p>

---

## Key Properties

- CPU-only inference using ncnn
- Real-time, multi-face processing
- ~90 FPS end-to-end on constrained hardware
- Low memory footprint
- Suitable for edge and IoT deployments

---

## Why ncnn

This project uses ncnn for both face detection and emotion classification.

ncnn was chosen because it provides:
- Efficient CPU execution with low runtime overhead
- Small binary and memory footprint
- Strong performance on edge and embedded systems
- Simple deployment without GPU dependencies

The pipeline is designed to stress real-time CPU inference and thread scheduling using ncnn.

---

## Use Cases

- Retail customer engagement analysis
- Classroom attention monitoring
- Mood tracking in clinical settings
- Audience reaction analysis

---

## Architecture Overview
```mermaid
graph LR
    A[Camera] --> B[Reader Thread]
    B --> C["Face Detection<br/>(UltraFace ncnn)"]
    C --> D["Emotion Classification<br/>(MobileNetV2 ncnn)"]
    D --> E[Display]

    B -.-> Q1[Queue]
    C -.-> Q2[Queue]
    D -.-> Q3[Queue]

    Q1 --> C
    Q2 --> D
    Q3 --> E
```

The pipeline uses lock-free queues and avoids unnecessary copies to maintain low latency.

---

## Performance Summary

**Test device**: Intel i5-3320M, 8 GB RAM

| Metric | Value |
|--------|-------|
| End-to-end throughput | ~90 FPS |
| End-to-end latency | ~15 ms |
| Memory usage | ~50 MB |
| CPU utilization | <10% |

### Edge and IoT Evaluation

The pipeline was evaluated under a Docker-simulated IoT environment:

- 1 vCPU
- 512 MB RAM
- Throttled I/O

The system sustains ~90 FPS end-to-end with under 10 percent CPU utilization.

Benchmark scripts and Docker configuration are included for reproducibility.

---

## Models

**Face detection**: Ultra-Light-Fast Face Detector (ncnn)

**Emotion classification**: MobileNetV2 (ncnn)
- Grayscale input
- 7 emotion classes: angry, disgust, fear, happy, neutral, sad, surprise

| Model | Accuracy | Inference Time |
|-------|----------|----------------|
| MobileNetV2 | 79.3% | 9.1 ms |

Training details, datasets, and ablations are documented in `src/python/README.md`.

---

## Image Embeddings

The emotion model exposes intermediate image embeddings that can be reused for:

- Temporal smoothing
- Identity-aware emotion tracking
- Downstream analytics

This allows extensions beyond per-frame classification.

---

## Quick Start

### Dependencies
```bash
sudo apt install opencv-dev cmake
```

### Build
```bash
git clone https://github.com/fw7th/emotion.git
cd src
mkdir build && cd build
cmake .. && make -j$(nproc)
```

### Run
```bash
./emotion 0
```

---

## Benchmarks

<details>
<summary><strong>Detailed Benchmarks</strong></summary>

### Face Detection
- ~150 FPS
- ~6.1 ms per frame

### Emotion Classification
- ~90 FPS
- ~9.1 ms per frame

### End-to-End Pipeline
- ~15 ms latency
- ~90 FPS sustained

### IoT Node Breakdown

| Stage | Latency (ms) |
|-------|--------------|
| Face Detector | ~6.1 |
| Emotion Detector | ~9.1 |
| Display | ~0.03 |
| End-to-End | ~15 |

Benchmarks averaged over 5 runs under real-time scheduling.
Docker constraints applied using `--cpus=1 --memory=512m`.

</details>

---

## Limitations

- Limited multi-face temporal tracking
- SORT-based tracking planned

---

## Relation to ncnn

This project is a downstream application of ncnn and serves as a reference for:

- Real-time multi-threaded pipelines
- CPU-only inference at high frame rates
- Edge and low-power deployment scenarios

---

## Acknowledgements

- [ncnn](https://github.com/Tencent/ncnn), BSD 3-Clause License
- [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) by Linzaer, MIT License
