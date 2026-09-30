# Differential Pulse Code Modulation (DPCM) & Golomb Audio Compression Suite

> **Course:** ELL-786 Multimedia Systems (Assignment 3)  
> **Institution:** Bharti School of Telecommunication Technology and Management, IIT Delhi  
> **Author:** Sudhanshu Chaudhary (2019JTM2207)  
> **Academic Report:** [`As3_2019JTM2207.pdf`](./As3_2019JTM2207.pdf)  
> **Research Papers:** [`Paper.pdf`](./Paper.pdf), [`Paper1.pdf`](./Paper1.pdf)

---

## 1. Project Overview & Architecture

Differential Pulse Code Modulation (DPCM) is a classic predictive waveform coding technique that exploits sample-to-sample autocorrelation in audio signals. Because consecutive audio samples are strongly correlated, the prediction residual $e(n) = x(n) - \hat{x}(n)$ has drastically lower variance than the raw signal $x(n)$. 

Quantizing and entropy-coding this low-variance residual via **Golomb-Rice coding** achieves significant bitrate reduction while maintaining high audio fidelity.

### DPCM Codec Architecture

```
ENCODER:
   x(n) ---->(+)----------------------------> [Uniform Quantizer] ---> e_q(n) ---> [Golomb Coder] ---> Bitstream
              ^ -                                     |
              |                                       v
          x_hat(n) <--- [Linear Predictor] <--- (+) <----+
                              A = [a1..aN]        |
                                                  v
                                               x_rec(n)

DECODER:
   Bitstream ---> [Golomb Decoder] ---> e_q(n) ---> (+) -----------------------------------------> x_rec(n)
                                                     ^
                                                     |
                                                 x_hat(n) <--- [Linear Predictor] <--- (Feedback)
```

---

## 2. Directory Structure

```
audio_compression/
├── README.md                            # Comprehensive project documentation
├── compress_audio.py                    # Unified CLI audio compressor & benchmark runner
│
├── src/                                 # Modular DPCM and entropy codec implementations
│   ├── quantizer.py                     # Uniform quantizer with dynamic step size calculation
│   ├── linear_predictor.py              # Yule-Walker LPC solver & DPCM feedback loops
│   ├── golomb_rice.py                   # Interleaving map & Golomb-Rice entropy codec
│   └── audio_pipeline.py                # End-to-end WAV processing, SNR & SPER metrics
│
├── 0Music.wav                           # Audio sample 1: Orchestral / Music
├── 1Dialogue.wav                        # Audio sample 2: Voice Dialogue
├── 2Speech.wav                          # Audio sample 3: Acoustic Speech
├── 3Music.wav                           # Audio sample 4: Electronic Music
├── A3.py                                # Legacy Assignment 3 monolithic script
├── As3_2019JTM2207.pdf                  # Academic laboratory report
├── Paper.pdf                            # Reference research paper
└── Paper1.pdf                           # Reference research paper
```

---

## 3. Mathematical Foundations

### 1. Optimal Linear Prediction (Yule-Walker Equations)
The predicted sample $\hat{x}(n)$ is a linear combination of the previous $N$ samples:
$$\hat{x}(n) = \sum_{k=1}^N a_k x(n - k)$$

The optimal filter coefficients $A = [a_1, a_2, \dots, a_N]^T$ minimize mean squared prediction error and are obtained by solving the Yule-Walker equations:
$$R \cdot A = r$$
where $R$ is the $N \times N$ symmetric Toeplitz autocorrelation matrix:
$$R_{ij} = R_{xx}(|i - j|) = \frac{1}{M} \sum_{n=0}^{M - 1 - |i - j|} x(n) x(n + |i - j|)$$
and $r = [R_{xx}(1), R_{xx}(2), \dots, R_{xx}(N)]^T$.

### 2. Uniform Mid-Tread Quantization
Given peak amplitude $x_{max} = \max(|e(n)|)$ and bit depth $B$:
$$\Delta = \frac{2 x_{max}}{2^B}$$
The quantized index $q(n)$ is:
$$q(n) = \text{sign}(e(n)) \cdot \min\left(\left\lfloor \frac{|e(n)|}{\Delta} \right\rfloor, 2^{B-1} - 1\right)$$

### 3. Signed-to-Positive Interleaving Map
Because Golomb coding operates on non-negative integers, signed indices are interleaved:
$$\text{map}(q) = \begin{cases} 2|q| - 1, & q < 0 \quad (\text{odd integers}) \\ 2q, & q \ge 0 \quad (\text{even integers}) \end{cases}$$

### 4. Golomb-Rice Entropy Coding
Given integer parameter $m$:
- **Quotient:** $q = n // m$ is coded in unary ($q$ ones followed by a `'0'`).
- **Remainder:** $r = n \% m$ is coded using truncated binary with $b = \lceil \log_2 m \rceil$ bits.
- **Optimal Parameter $m$:** For geometric distributions with mean $\mu$:
  $$p = \frac{1}{1 + \mu}, \quad m \approx \left\lceil -\frac{\ln 2}{\ln(1 - p)} \right\rceil$$

### 5. Performance Metrics
- **Signal-to-Noise Ratio (SNR):**
  $$\text{SNR} = 10 \log_{10} \left( \frac{\sum x(n)^2}{\sum (x(n) - \hat{x}(n))^2} \right) \text{ dB}$$
- **Signal-to-Prediction-Error Ratio (SPER):**
  $$\text{SPER} = 10 \log_{10} \left( \frac{\sum x(n)^2}{\sum e(n)^2} \right) \text{ dB}$$

---

## 4. Experimental Benchmarks

Benchmark results across predictor orders $N \in \{1, 2, 4\}$ and bit depths $B \in \{4, 8, 12\}$ on `1Dialogue.wav` (1,176,231 samples, 16 kHz):

| Predictor Order ($N$) | Bit Depth ($B$) | Quantization Step ($\Delta$) | SNR (dB) | Prediction Gain / SPER (dB) | Average Bitrate | Compression Ratio |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 4 bits | 551.21 | 0.74 dB | 14.03 dB | 1.00 b/sample | **16.0x** |
| **1** | 8 bits | 34.45 | 13.74 dB | 14.03 dB | 2.09 b/sample | **7.6x** |
| **1** | 12 bits | 2.15 | 41.98 dB | 14.03 dB | 5.83 b/sample | **2.7x** |
| **2** | 4 bits | 380.80 | 0.89 dB | **18.35 dB** | 1.00 b/sample | **16.0x** |
| **2** | 8 bits | 23.80 | 13.40 dB | **18.35 dB** | 2.06 b/sample | **7.8x** |
| **2** | 12 bits | 1.49 | 40.95 dB | **18.35 dB** | 5.69 b/sample | **2.8x** |
| **4** | 4 bits | 363.22 | 0.90 dB | **18.46 dB** | 1.00 b/sample | **16.0x** |
| **4** | 8 bits | 22.70 | 13.89 dB | **18.46 dB** | 2.07 b/sample | **7.7x** |
| **4** | 12 bits | 1.42 | 41.37 dB | **18.46 dB** | 5.77 b/sample | **2.8x** |

> **Key Takeaways:**
> - Moving from 1st-order ($N=1$) to 2nd-order ($N=2$) prediction delivers an immediate **+4.32 dB** jump in SPER prediction gain due to effective formant tracking.
> - At 12-bit quantization, reconstructed audio achieves $> 41$ dB SNR, rendering quantization distortion imperceptible.
> - At 4-bit quantization with Golomb coding, bitrate drops to $1.00$ bit/sample, yielding **16x** data reduction.

---

## 5. Usage & Execution

```bash
# 1. Run multi-order comparison table on 1Dialogue.wav
uv run python audio_compression/compress_audio.py --compare

# 2. Compress audio with order 2 predictor, 8-bit quantization, and save output
uv run python audio_compression/compress_audio.py --input audio_compression/2Speech.wav --order 2 --bits 8 --output audio_compression/rec_speech.wav

# 3. Compress musical audio with 4th-order predictor
uv run python audio_compression/compress_audio.py --input audio_compression/0Music.wav --order 4 --bits 12
```

---

## 6. Automated Testing

Run the 14 automated unit and integration tests:

```bash
uv run pytest tests/test_audio_compression.py -v
```