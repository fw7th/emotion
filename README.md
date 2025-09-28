# Real-Time Multi-Face Emotion Detection
*High-performance emotion recognition pipeline achieving 65+ FPS on CPU*

[Demo GIF here]

## Performance
- Face Detection: 150 FPS
- Emotion Classification: 65 FPS  
- End-to-end Latency: <14ms

## Quick Start
git clone https://github.com/fw7th/emotion
cd src
mkdir build && cd build
cmake ..
make -j4
./emotion 0

## Architecture
[System diagram]

## Technical Details
[The meaty stuff]

## Benchmarks
[Detailed performance analysis]

## Limitations
N-face emotion smoothing and tracking with SORT (planned).

## Acknowledgements
- This project uses [ncnn](https://github.com/Tencent/ncnn), a high-performance neural network inference framework (BSD 3-Clause License).
- Face detection based on [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) by Linzaer (MIT License).

### Third-Party Licenses
- [ncnn](https://github.com/Tencent/ncnn) — BSD 3-Clause License.
- [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) by Linzaer — MIT License.
