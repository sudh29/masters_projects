# Master's Engineering Projects Portfolio

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![C/C++](https://img.shields.io/badge/C%2FC%2B%2B-POSIX%20Sockets-orange.svg)](https://en.wikipedia.org/wiki/C_(programming_language))
[![Embedded](https://img.shields.io/badge/Hardware-Arduino%20%7C%20ESP32%20%7C%20RPi-green.svg)](https://www.arduino.cc/)
[![uv](https://img.shields.io/badge/Environment-uv%20Package%20Manager-blueviolet.svg)](https://github.com/astral-sh/uv)
[![Testing](https://img.shields.io/badge/Tests-45%20Passed-brightgreen.svg)](https://docs.pytest.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](./LICENSE)

A curated, production-grade portfolio of Master of Technology (M.Tech) engineering projects and research implementations completed at the **Bharti School of Telecommunication Technology and Management, Indian Institute of Technology (IIT) Delhi**.

---

## 1. Domain Architecture & Curriculum Matrix

```
                                  +---------------------------------------+
                                  |   IIT Delhi M.Tech Projects Portfolio |
                                  +---------------------------------------+
                                                      |
         +--------------------+-----------------------+----------------------+--------------------+
         |                    |                       |                      |                    |
         v                    v                       v                      v                    v
+------------------+ +------------------+   +-------------------+  +------------------+ +------------------+
|  Computer Comms  | | Embedded Systems |   | Image Compression |  | Audio Compression| | Machine Learning |
|    (ELL-785)     | |    (ELP-720)     |   |    (ELL-786)      |  |    (ELL-786)     | | (Computer Vision)|
+------------------+ +------------------+   +-------------------+  +------------------+ +------------------+
| • BSD TCP Sockets| | • Smart Keypad   |   | • 2D-DCT & JPEG   |  | • Optimal LPC    | | • Adaptive GMM   |
| • Client/Server  | |   Calculator     |   | • Exact Arithmetic|  |   Yule-Walker    | | • 3D Tensor MoG  |
| • Student/Teacher| | • 4:2 Priority   |   | • LZ77 / LZ78/ LZW|  | • Uniform Quant  | | • 112+ FPS CPU   |
|   RBAC Portal    | |   Encoder & SD   |   | • Repetition Error|  | • Signed Inter-  |   Acceleration   |
| • Wireshark RTT  | | • ESP32 Telegram |   |   Correction (3,1)|    leaving Map      | • Moving Object    |
|   Hop Analysis   | | • Blynk IoT Cloud|   | • PSNR / MSE Bench|  | • Golomb-Rice    |   Segmentation   |
|                  | | • RPi Serial Comm|   |                   |  | • SPER / SNR Eval|                    |
+------------------+ +------------------+   +-------------------+  +------------------+ +------------------+
```

---

## 2. Project Catalog & Directory Guide

| Project Domain | Academic Course | Core Technologies | Primary Artifacts | Technical Documentation |
|:---|:---|:---|:---|:---:|
| **[Socket Programming](./socket_programming/)** | ELL-785 Computer Communication Networks | C, POSIX Sockets, TCP/IP, Wireshark, Make | `server.c`, `client.c`, `Makefile`, `student_marks.txt`, `user_pass.txt` | [README.md](./socket_programming/README.md) |
| **[Embedded Systems](./embedded/)** | ELP-720 Telecommunication Networks Lab | Arduino Uno, ESP32, Raspberry Pi, C++, Python | `calculator.ino`, `priority_encoder.ino`, `weather_station.ino`, `blynk_touch.ino`, `rpi_sender.py` | [README.md](./embedded/README.md) |
| **[Image Compression](./image_compression/)** | ELL-786 Multimedia Systems | Python, NumPy, 2D-DCT, JPEG Quantization, LZ Dictionary | `compress_image.py`, `dct_codec.py`, `arithmetic_codec.py`, `dictionary_codecs.py`, `repetition_codec.py` | [README.md](./image_compression/README.md) |
| **[Audio Compression](./audio_compression/)** | ELL-786 Multimedia Systems | Python, Linear Prediction (LPC), DPCM, Golomb-Rice | `compress_audio.py`, `linear_predictor.py`, `quantizer.py`, `golomb_rice.py`, `audio_pipeline.py` | [README.md](./audio_compression/README.md) |
| **[Machine Learning](./machine_learning/)** | Computer Vision / Machine Learning | Python, OpenCV, Stauffer-Grimson GMM, NumPy | `tracker_cli.py`, `gmm_model.py`, `config.py`, `synthetic_feed.py` | [README.md](./machine_learning/README.md) |

---

## 3. Quickstart & Environment Setup

This repository uses [`uv`](https://github.com/astral-sh/uv) for blazingly fast, deterministic Python environment and dependency management.

### 1. Clone & Synchronize Environment
```bash
git clone <repo-url> masters_projects
cd masters_projects

# Sync virtual environment and install dependencies
uv sync --extra dev
```

### 2. Run the Full Automated Test Suite
The repository includes 45 unit and integration tests across all 5 engineering domains:
```bash
uv run pytest -v
```

---

## 4. Running Individual Project Suites

### 1. Socket Programming (C/C++ Network Server & Client)
```bash
# Compile server and client binaries
make -C socket_programming

# Run server on port 4080
./socket_programming/server 4080 &

# Query student records (Student View)
./socket_programming/client 127.0.0.1 4080 sudhanshu s123

# Query gradebook and class average (Instructor View)
./socket_programming/client 127.0.0.1 4080 instructor i123
```

### 2. Embedded Systems & IoT
```bash
# Run 4-to-2 Priority Encoder truth table verification
uv run python embedded/arduino/priority_encoder/verify_priority_encoder.py

# Run Raspberry Pi string transmitter simulation
uv run python embedded/raspberry_pi/rpi_sender.py "IIT Delhi ELP720"
```

### 3. Image Compression & Source Coding
```bash
# 2D-DCT compression on cat.jpg with quality factor 75
uv run python image_compression/compress_image.py --algo dct --quality 75

# Exact interval arithmetic coding
uv run python image_compression/compress_image.py --algo arithmetic --text "MULTIMEDIA SYSTEMS IIT DELHI"

# LZW lossless compression
uv run python image_compression/compress_image.py --algo lzw --text "a.bar.array.by.barrayar.bay."
```

### 4. Audio Compression (DPCM & Golomb-Rice)
```bash
# Multi-order prediction comparison (N=1, 2, 4) on speech audio
uv run python audio_compression/compress_audio.py --input audio_compression/1Dialogue.wav --compare

# Compress dialogue audio with 2nd-order predictor and 8-bit quantization
uv run python audio_compression/compress_audio.py --input audio_compression/1Dialogue.wav --order 2 --bits 8 --output audio_compression/rec_dialogue.wav
```

### 5. Machine Learning Video Activity Tracker
```bash
# Run headless synthetic benchmark (30 frames at > 110 FPS)
uv run python machine_learning/tracker_cli.py --input synthetic --frames 30

# Process custom video file with GMM configuration
uv run python machine_learning/tracker_cli.py --input path/to/video.mp4 --params machine_learning/Params.txt
```

---

## 5. Academic Integrity & References

All original laboratory reports, IEEE research papers, and circuit diagrams are preserved in their respective project directories:
- **Networks:** [`socket_programming/Socket Programming.pdf`](./socket_programming/Socket%20Programming.pdf)
- **Embedded:** [`embedded/Arduino.pdf`](./embedded/Arduino.pdf), [`ESP32.pdf`](./embedded/ESP32.pdf), [`ESP32_1.pdf`](./embedded/ESP32_1.pdf), [`Raspberry_Pi.pdf`](./embedded/Raspberry_Pi.pdf)
- **Image Coding:** [`image_compression/MultiA1.pdf`](./image_compression/MultiA1.pdf), [`2019JTM2207_2019JTM2088.pdf`](./image_compression/2019JTM2207_2019JTM2088.pdf)
- **Audio Coding:** [`audio_compression/As3_2019JTM2207.pdf`](./audio_compression/As3_2019JTM2207.pdf), [`Paper.pdf`](./audio_compression/Paper.pdf), [`Paper1.pdf`](./audio_compression/Paper1.pdf)
- **Machine Learning:** [`machine_learning/1.pdf`](./machine_learning/1.pdf) (PAMI 2000), [`2.pdf`](./machine_learning/2.pdf) (CVPR 1999)

---

## 6. License

This repository is licensed under the [GNU General Public License v3.0](./LICENSE).
