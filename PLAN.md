# Master's Projects: Repository Enhancement & Modernization Plan

> **Academic Program:** M.Tech in Telecommunication Technology & Management, IIT Delhi  
> **Repository:** `/home/liber_primus/code/masters_projects`  
> **Last Updated:** 2026-09-30 20:50 IST  
> **Current Overall Repo Score:** **10.0 / 10.0** (All Phases Completed)

---

## 1. Executive Summary & Goals

This repository contains academic projects and lab assignments completed during Master of Technology studies at IIT Delhi, spanning **Multimedia Systems (ELL-786)**, **Computer Communication Networks (ELL-785)**, **Telecommunication Networks Laboratory (ELP-720)**, and **Machine Learning Computer Vision**.

While the theoretical reports (PDFs) and research papers in each directory are detailed and comprehensive, the repository currently suffers from:
1. **Missing or Inaccessible Code:** Core implementations for `socket_programming` and `embedded` reside only inside report appendices rather than versioned source files.
2. **Monolithic & Fragile Code:** Existing Python scripts have hardcoded absolute paths (e.g., `/home/sudhanshu/...`), procedural global state, and lack modular architecture or CLI interfaces.
3. **Zero Automated Testing:** No automated unit or integration tests exist across the entire repository.
4. **Outdated / Incomplete Documentation:** Root `README.md` is empty (1 byte), `embedded/README.md` has unmerged git conflict markers, and other directory READMEs only contain `# masters_projects`.
5. **No Python Environment Management:** Dependencies are unpinned and unmanaged; no `uv` project or `pyproject.toml` exists.

### Transformation Objectives
- **Phase-by-Phase Execution:** Upgrade one project at a time in strict sequence.
- **Code Completeness:** Extract, reconstruct, and modularize all code files from PDF reports and legacy scripts.
- **Automated Test Coverage:** Add comprehensive `pytest` and test harness scripts for every project.
- **Modern Python Tooling with `uv`:** Configure root `pyproject.toml` managed via `uv`, with pinned dependencies, virtual environment setup, and linting/formatting tools.
- **Rich Documentation:** Create production-grade READMEs for every project with system diagrams, mathematical formulations, hardware wiring, usage commands, and PDF cross-references, along with an outstanding portfolio root `README.md`.
- **Git Hygiene:** Clean `.gitignore` and resolve git merge conflict markers.
- **Quality Scoring:** Assess and record a rigorous repository score out of 10.0 with a timestamp after every phase.

---

## 2. Repository Quality Scoring Rubric (10-Point Scale)

To track progress objectively, the repository is evaluated across 5 core dimensions totaling 10.0 points:

| Dimension | Weight | Description | Baseline | Phase 0 | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 | Final (Phase 6) |
|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Repository Structure & Workspace Hygiene** | **2.0 pts** | Clean directory layout, resolved git conflicts, complete `.gitignore` (ignoring `.venv`, caches, build binaries), standard file naming. | 0.4 / 2.0 | 1.2 / 2.0 | 1.5 / 2.0 | 1.7 / 2.0 | 1.8 / 2.0 | 1.9 / 2.0 | 2.0 / 2.0 | **2.0 / 2.0** |
| **2. Environment & Dependency Management (`uv`)** | **1.5 pts** | Root `pyproject.toml`, `uv` lockfile/environment, pinned dependencies, standardized CLI/execution scripts. | 0.2 / 1.5 | 1.2 / 1.5 | 1.2 / 1.5 | 1.2 / 1.5 | 1.3 / 1.5 | 1.4 / 1.5 | 1.5 / 1.5 | **1.5 / 1.5** |
| **3. Code Completeness, Modularity & Quality** | **3.0 pts** | All projects contain runnable code files (extracted from reports), clean functions/classes, eliminated hardcoded paths, type annotations, error handling. | 0.8 / 3.0 | 0.8 / 3.0 | 1.3 / 3.0 | 1.8 / 3.0 | 2.3 / 3.0 | 2.6 / 3.0 | 2.9 / 3.0 | **3.0 / 3.0** |
| **4. Testing, Simulation & Verification** | **2.0 pts** | Automated test suites (`pytest`, integration tests, C mock clients, synthetic test generation) passing across all projects. | 0.0 / 2.0 | 0.4 / 2.0 | 0.8 / 2.0 | 1.2 / 2.0 | 1.6 / 2.0 | 1.8 / 2.0 | 1.9 / 2.0 | **2.0 / 2.0** |
| **5. Technical Documentation & Academic Presentation** | **1.5 pts** | Comprehensive root `README.md`, individual project READMEs with architecture diagrams, mathematical equations, hardware pinouts, and report links. | 0.4 / 1.5 | 0.7 / 1.5 | 0.9 / 1.5 | 1.1 / 1.5 | 1.3 / 1.5 | 1.4 / 1.5 | 1.4 / 1.5 | **1.5 / 1.5** |
| **Total Repository Score** | **10.0 pts** | **Weighted Composite Score** | **1.8 / 10.0** | **4.3 / 10.0** | **5.7 / 10.0** | **7.0 / 10.0** | **8.3 / 10.0** | **9.1 / 10.0** | **9.7 / 10.0** | **10.0 / 10.0** |

---

## 3. Score Progression & Audit Log

| Phase | Description | Completion Timestamp | Repo Score (out of 10) | Status |
|:---:|:---|:---:|:---:|:---:|
| **—** | **Initial Baseline Audit** | **2026-09-30 18:05 IST** | **1.8 / 10.0** | Completed |
| **Phase 0** | **Foundation: Git Hygiene, `.gitignore` & `uv` Tooling Setup** | **2026-09-30 18:07 IST** | **4.3 / 10.0** | **Completed** |
| **Phase 1** | **Project 1: Socket Programming (C/C++ Network Programming)** | **2026-09-30 18:19 IST** | **5.7 / 10.0** | **Completed** |
| **Phase 2** | **Project 2: Embedded Systems (Arduino, ESP32, Raspberry Pi)** | **2026-09-30 18:23 IST** | **7.0 / 10.0** | **Completed** |
| **Phase 3** | **Project 3: Image Compression (DCT, Arithmetic & LZ Codecs)** | **2026-09-30 18:27 IST** | **8.3 / 10.0** | **Completed** |
| **Phase 4** | **Project 4: Audio Compression (DPCM & Golomb Coding)** | **2026-09-30 20:04 IST** | **9.1 / 10.0** | **Completed** |
| **Phase 5** | **Project 5: Machine Learning (Adaptive GMM Background Subtraction)** | **2026-09-30 20:42 IST** | **9.7 / 10.0** | **Completed** |
| **Phase 6** | **Portfolio Synthesis: Root README, Unified Test Harness & Polish** | **2026-09-30 20:50 IST** | **10.0 / 10.0** | **Completed** |

---

## 4. Phased Implementation Roadmap

---

### Phase 0: Workspace Foundation & Modern Tooling Setup (`uv` & Git Hygiene)
- **Status:** **Completed** (2026-09-30 18:07 IST)
- **Objective:** Establish the development environment, fix broken git states, update `.gitignore`, and configure `uv` for reproducible execution.
- **Key Tasks:**
  - [x] Resolve merge conflict markers in `embedded/README.md`.
  - [x] Modernize `.gitignore` to ignore `.venv/`, `__pycache__/`, `.pytest_cache/`, `*.egg-info`, compiled C binaries (`*.o`, `server`, `client`), and temporary compressed output artifacts while preserving sample inputs (`.wav`, `.bmp`, `.jpg`).
  - [x] Initialize Python packaging with `uv` (`pyproject.toml`):
    - Specify Python `>=3.11, <3.13` (using Python 3.12).
    - Runtime dependencies: `numpy`, `scipy`, `opencv-python`, `pillow`, `matplotlib`
    - Development dependencies: `pytest`, `pytest-cov`, `ruff`
  - [x] Generate `uv.lock` and verify virtual environment creation (`uv sync --extra dev`).
  - [x] Create basic `tests/` directory structure with a smoke test (`tests/test_smoke.py`) to verify `uv run pytest`.
- **Verification:**
  - Ran `git status` (no unresolved conflicts).
  - Ran `uv run pytest` (2 passed in 1.03s).
- **Deliverables:** Clean `.gitignore`, root `pyproject.toml`, `uv.lock`, initial `tests/test_smoke.py`.
- **Post-Phase Score:** **4.3 / 10.0** (recorded 2026-09-30 18:07 IST).

---

### Phase 1: Project 1 — Socket Programming (C/C++ Network Programming)
- **Status:** **Completed** (2026-09-30 18:19 IST)
- **Academic Context:** Course **ELL-785 (Computer Communication Networks)**, Assignment 1.
- **Report Reference:** `socket_programming/Socket Programming.pdf`.
- **Description:** Client-server TCP internet socket application with authentication and role-based access for student marks queries (5 subjects out of 100), RTT measurement, and Wireshark protocol analysis.
- **Key Tasks:**
  - [x] Extract and format server code from PDF Appendix A.1 into `socket_programming/src/server.c` with signal handling, dynamic path discovery, and robust token framing.
  - [x] Extract and format client code from PDF Appendix A.2 into `socket_programming/src/client.c` supporting interactive mode and scripted CLI flags (`./client <host> <port> <user> <pass>`).
  - [x] Create `socket_programming/Makefile` with targets (`all`, `server`, `client`, `clean`).
  - [x] Create mock student database `socket_programming/data/student_marks.txt` (20 students) and credentials file `socket_programming/data/user_pass.txt` matching report specifications.
  - [x] Build an automated test suite (`tests/test_socket_programming.py`) using Python `subprocess` and socket mocks to verify:
    - Server startup and TCP port binding.
    - Instructor login and retrieval of full student records.
    - Student login and retrieval of individual marks.
    - Rejection of invalid credentials.
    - Data integrity across databases.
  - [x] Write comprehensive `socket_programming/README.md`:
    - Overview, architecture, and TCP state machine.
    - Protocol exchange specification & message format.
    - Step-by-step compilation and execution guide.
    - Wireshark frame & packet analysis summary (referencing report findings).
- **Verification:**
  - Ran `make -C socket_programming clean all` (clean build, 0 warnings).
  - Ran `uv run pytest tests/test_socket_programming.py` (4 passed in 1.16s).
- **Deliverables:** `server.c`, `client.c`, `Makefile`, sample data, automated test suite, `socket_programming/README.md`.
- **Post-Phase Score:** **5.7 / 10.0** (recorded 2026-09-30 18:19 IST).

---

### Phase 2: Project 2 — Embedded Systems (Arduino, ESP32, Raspberry Pi IoT)
- **Status:** **Completed** (2026-09-30 18:23 IST)
- **Academic Context:** Course **ELP-720 (Telecommunication Networks Laboratory)**, Assignments 2, 4, 5, 8.
- **Report References:** `embedded/Arduino.pdf`, `ESP32.pdf`, `ESP32_1.pdf`, `Raspberry_Pi.pdf`, and schematics `Arduino_Dia.png`, `ESP32_Dia.png`, `Raspberry_Pi_Dia.png`.
- **Sub-Projects & Tasks:**
  - **Arduino Lab (Assignment 2):**
    - [x] Extract PS1 code into `embedded/arduino/calculator/calculator.ino` (authenticated 16x2 LCD keypad calculator with LED binary output).
    - [x] Extract PS2 code into `embedded/arduino/priority_encoder/priority_encoder.ino` (4-to-2 hardware priority encoder with SD card truth table logger).
    - [x] Provide Python truth table verification script (`embedded/arduino/priority_encoder/verify_priority_encoder.py`).
  - **ESP32 IoT Labs (Assignments 4 & 5):**
    - [x] Extract Assignment 4 code into `embedded/esp32/weather_station/weather_station.ino` (IITD WiFi, Telegram Bot API, I2C LCD, capacitive touch alert).
    - [x] Extract Assignment 5 code into `embedded/esp32/blynk_touch/blynk_touch.ino` (ESP32 touch sensor and Blynk IoT cloud integration).
    - [x] Provide simulated mock handlers and verification logic.
  - **Raspberry Pi Inter-board Serial Communication (Assignment 8):**
    - [x] Extract Raspberry Pi Python transmitter into `embedded/raspberry_pi/rpi_sender.py` (string encoding, parity/pattern matching, serial transmission).
    - [x] Extract Arduino serial receiver into `embedded/raspberry_pi/arduino_receiver.ino`.
    - [x] Extract pattern matching module into `embedded/raspberry_pi/pattern_matcher.py`.
  - **Testing & Verification:**
    - [x] Add unit tests (`tests/test_embedded.py`) for Raspberry Pi bit encoding/pattern matching algorithms and priority encoder logic truth tables (6 tests).
  - **Documentation:**
    - [x] Write detailed `embedded/README.md`:
      - Hardware requirements and pinout mapping for all 4 lab exercises.
      - Embedded diagrams and schematics referencing the PNG files.
      - Instructions for compilation and flashing via Arduino CLI / IDE.
- **Verification:**
  - Ran `uv run pytest tests/test_embedded.py` (6 passed in 0.02s).
  - Validated syntax of all `.ino`, `.c`, and `.py` embedded files.
- **Deliverables:** Modular embedded directories, firmware sketches, Python communication scripts, test suite, `embedded/README.md`.
- **Post-Phase Score:** **7.0 / 10.0** (recorded 2026-09-30 18:23 IST).

---

### Phase 3: Project 3 — Image Compression (Multimedia Source Coding: DCT & Dictionaries)
- **Status:** **Completed** (2026-09-30 18:27 IST)
- **Academic Context:** Course **ELL-786 (Multimedia Systems)**, Assignments 1 & 2.
- **Report References:** `image_compression/MultiA1.pdf`, `2019JTM2207_2019JTM2088.pdf`, `PROGRAMMING ASSIGNMENT 1.pdf`, `A2.pdf`.
- **Key Tasks:**
  - [x] Clean up redundant/legacy files (`Part1old.py`, redundant `dct_inversedct.py`).
  - [x] Fix hardcoded paths in `A2.py` and `Part2.py` (`/home/sudhanshu/...` -> dynamic relative paths).
  - [x] Refactor and modularize Assignment 1:
    - `image_compression/src/repetition_codec.py`: Repetition coding with Hamming error injection, BER computation, and majority logic decoding.
    - `image_compression/src/arithmetic_codec.py`: Exact interval arithmetic encoder and decoder with arbitrary-length precision using `fractions.Fraction`.
    - `image_compression/src/dct_codec.py`: 8x8 2D-DCT transform matrix computation, JPEG quantization matrix scaling, zigzag scan, inverse transform, MSE and PSNR calculation.
  - [x] Refactor and modularize Assignment 2:
    - `image_compression/src/dictionary_codecs.py`: Clean implementations of LZ77 (sliding window), LZ78, and LZW compression/decompression algorithms.
  - [x] Create a unified CLI tool: `image_compression/compress_image.py` supporting algorithms `--algo [dct|arithmetic|lz77|lz78|lzw|repetition]`.
  - [x] Add comprehensive test suite (`tests/test_image_compression.py`):
    - Repetition code error-correction capability test.
    - Arithmetic coder lossless roundtrip tests.
    - DCT transform invertibility ($T \cdot T^T = I$) and image reconstruction PSNR threshold test.
    - LZ77, LZ78, LZW lossless compression roundtrips (14 tests total).
  - [x] Write comprehensive `image_compression/README.md`:
    - Mathematical formulation of 2D-DCT, Quantization, Arithmetic interval subdivision, and LZ dictionary structures.
    - Benchmark results table: Compression ratio, PSNR, MSE on `1.bmp` and `cat.jpg`.
    - Usage instructions with example CLI commands.
- **Verification:**
  - Ran `uv run pytest tests/test_image_compression.py` (14 passed in 0.39s).
  - Executed CLI compression on sample images without errors.
- **Deliverables:** Clean modular source files, unified CLI, comprehensive test suite, `image_compression/README.md`.
- **Post-Phase Score:** **8.3 / 10.0** (recorded 2026-09-30 18:27 IST).

---

### Phase 4: Project 4 — Audio Compression (DPCM & Golomb-Rice Coding)
- **Status:** **Completed** (2026-09-30 20:04 IST)
- **Academic Context:** Course **ELL-786 (Multimedia Systems)**, Assignment 3.
- **Report Reference:** `audio_compression/As3_2019JTM2207.pdf` and reference papers (`Paper.pdf`, `Paper1.pdf`).
- **Media Files:** `0Music.wav`, `1Dialogue.wav`, `2Speech.wav`, `3Music.wav`.
- **Key Tasks:**
  - [x] Refactor monolithic `A3.py` (421 lines) into a clean, modular Python package `audio_compression/src/`:
    - `quantizer.py`: Uniform quantizer with parameterizable bit depth $N$ and dynamic step size $\Delta$.
    - `linear_predictor.py`: Optimal predictor coefficient computation using autocorrelation and Yule-Walker matrix solving for prediction orders $p \in \{1, 2, 3, 4\}$.
    - `golomb_rice.py`: Non-linear residual mapping, Golomb quotient unary coding, and remainder binary coding with tunable parameter $m$.
    - `audio_pipeline.py`: End-to-end DPCM encoder, decoder, residual signal analysis, and SNR/SPER computation.
  - [x] Create CLI entrypoint `audio_compression/compress_audio.py` for processing audio files with configurable predictor order, quantizer bits, and Golomb parameter.
  - [x] Add comprehensive test suite (`tests/test_audio_compression.py`):
    - Quantizer bounds and step reconstruction test.
    - Yule-Walker predictor stability and coefficient validation.
    - Golomb-Rice lossless encode/decode roundtrip test on synthetic residual arrays.
    - End-to-end WAV compression and decompression fidelity test across the audio files (14 tests).
  - [x] Write rich `audio_compression/README.md`:
    - Theoretical background: DPCM block diagram, Linear Predictive Coding, Golomb-Rice coding.
    - Comparative experimental results across the four audio files (`0Music.wav`, `1Dialogue.wav`, `2Speech.wav`, `3Music.wav`).
    - Comparison tables: SNR vs. Predictor Order (1, 2, 4) and Bit Rates (4, 8, 12 bits).
    - Reproduction guide with CLI examples.
- **Verification:**
  - Ran `uv run pytest tests/test_audio_compression.py` (14 passed in 0.14s).
  - Ran CLI compressor on `1Dialogue.wav` across multi-order comparison table.
- **Deliverables:** Modular DPCM/Golomb package, CLI runner, test suite, `audio_compression/README.md`.
- **Post-Phase Score:** **9.1 / 10.0** (recorded 2026-09-30 20:04 IST).

---

### Phase 5: Project 5 — Machine Learning (Adaptive GMM Background Subtraction)
- **Status:** **Completed** (2026-09-30 20:42 IST)
- **Academic Context:** Computer Vision / Machine Learning Course.
- **Paper References:** Stauffer & Grimson (CVPR 1999, PAMI 2000), `machine_learning/1.pdf`, `2.pdf`, `3.pdf`, `4.pdf`.
- **Key Tasks:**
  - [x] Refactor procedural `ML1.py` into high-performance vectorized package `machine_learning/src/`:
    - `gmm_model.py`: `GaussianMixtureBackground` class implementing adaptive Gaussian mixture modeling per pixel:
      - Multi-modal Gaussian maintenance ($K=3$ to $5$ distributions).
      - Parameter updates: online mean $\mu$, variance $\sigma^2$, and mixture weight $\omega$ adjusted by learning rate $\alpha$.
      - Fitness ordering by $\omega / \sigma$.
      - Background thresholding $T$ to segment foreground vs. background.
      - Vectorized 3D NumPy state tensors $(H, W, K)$ achieving $> 110$ FPS.
    - `config.py`: Parameter management supporting `Params.txt` as well as CLI overrides.
    - `synthetic_feed.py`: Synthetic video generator (moving geometry against static/noisy backgrounds) enabling automated headless testing without external webcam or video assets.
    - `tracker_cli.py`: Interactive CLI supporting synthetic stream, video file input, or live camera feed.
  - [x] Add comprehensive test suite (`tests/test_machine_learning.py`):
    - Initialization and weight sum normalization ($\sum \omega_i = 1$).
    - Parameter update equations correctness under matching vs. non-matching observations.
    - Background adaptation on static scenes and foreground mask generation on moving objects (5 tests).
  - [x] Write comprehensive `machine_learning/README.md`:
    - Detailed mathematical formulation of adaptive mixture models (Stauffer-Grimson algorithm).
    - Step-by-step update equations for $\omega_k, \mu_k, \sigma_k$.
    - Parameter tuning guide ($\alpha$, $T$, $K$, initial variance).
    - Performance benchmarks ($112.5$ FPS) and CLI usage examples.
- **Verification:**
  - Ran `uv run pytest tests/test_machine_learning.py` (5 passed in 0.19s).
  - Ran `tracker_cli.py --input synthetic --frames 30` (processed 30 frames in 0.27s).
- **Deliverables:** Object-oriented GMM background subtractor, synthetic generator, CLI runner, test suite, `machine_learning/README.md`.
- **Post-Phase Score:** **9.7 / 10.0** (recorded 2026-09-30 20:42 IST).

---

### Phase 6: Portfolio Synthesis, Root README & Final Polish
- **Status:** **Completed** (2026-09-30 20:50 IST)
- **Objective:** Present the entire repository as an exemplary Master's engineering portfolio, tie all project suites together, run repository-wide verification, and assign final evaluation score.
- **Key Tasks:**
  - [x] Create a world-class root `README.md`:
    - Project hero banner and IIT Delhi academic credentials.
    - Architectural overview and domain matrix (Networks, Embedded Systems, Image Compression, Audio Compression, Machine Learning).
    - Quickstart guide using `uv` (one-command environment setup, running tests across all projects).
    - Unified project index with direct links to project directories, reports, and code modules.
    - Tech stack badges (Python, C/C++, Arduino, OpenCV, uv, pytest).
  - [x] Code formatting and linting pass with `ruff format` and `ruff check`.
  - [x] Run full test suite covering all projects: `uv run pytest`.
  - [x] Final repository quality score computation and audit sign-off.
- **Verification:**
  - `uv run ruff check .` passed with 0 errors across all 30 source files.
  - `uv run ruff format --check .` confirmed 100% compliant formatting.
  - `uv run pytest -v` executed with **45 passed in 2.06s** (87% overall statement coverage).
- **Deliverables:** Master root `README.md`, verified repository configuration, full passing test suite, final score audit.
- **Post-Phase Score:** **10.0 / 10.0** (recorded 2026-09-30 20:50 IST).

---

## 5. Execution Protocol & Verification Standard

1. **One Project at a Time:** Execution must proceed sequentially through Phase 0 to Phase 6 without mixing project domains.
2. **Phase Completion Gate:** Before advancing from Phase $N$ to Phase $N+1$:
   - All tasks in Phase $N$ must be checked `[x]`.
   - All associated unit/integration tests must pass cleanly.
   - The project's documentation (`README.md`) must be complete.
   - The **Score Progression & Audit Log** in this document must be updated with the new score and exact timestamp.
3. **Preservation of Academic Evidence:** All original PDF reports, research papers, diagrams, and reference media must be preserved intact and directly linked from the new documentation.
