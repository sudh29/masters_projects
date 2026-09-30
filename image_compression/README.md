# Multimedia Image Compression & Source Coding Suite

> **Course:** ELL-786 Multimedia Systems (Assignments 1 & 2)  
> **Institution:** Bharti School of Telecommunication Technology and Management, IIT Delhi  
> **Authors:** Sudhanshu Chaudhary (2019JTM2207), Santosh Kumar (2019JTM2088)  
> **Reports:** [`MultiA1.pdf`](./MultiA1.pdf) (Assignment 1), [`2019JTM2207_2019JTM2088.pdf`](./2019JTM2207_2019JTM2088.pdf) (Assignment 2)  
> **Assignment Prompts:** [`PROGRAMMING ASSIGNMENT 1.pdf`](./PROGRAMMING%20ASSIGNMENT%201.pdf), [`A2.pdf`](./A2.pdf)

---

## 1. Directory Structure

```
image_compression/
├── README.md                            # Comprehensive project documentation
├── compress_image.py                    # Unified CLI compression runner
│
├── src/                                 # Modular codec implementations
│   ├── repetition_codec.py              # Repetition coding & majority logic error correction
│   ├── arithmetic_codec.py              # Lossless interval arithmetic encoder & decoder
│   ├── dct_codec.py                     # 2D-DCT, JPEG quantization & zigzag scan
│   └── dictionary_codecs.py             # Lossless LZ77, LZ78, and LZW codecs
│
├── 1.bmp                                # Sample bitmap image for dictionary coding
├── cat.jpg                              # Sample photograph for 2D-DCT compression
├── compressedcat1.jpg                   # Reconstructed image from DCT compression
├── data.txt                             # Sample text corpus
├── Part1.py                             # Legacy Assignment 1 Part 1 script
├── Part2.py                             # Legacy Assignment 1 Part 2 script
├── A2.py                                # Legacy Assignment 2 Dictionary script
├── dct.py                               # Legacy standalone DCT script
│
├── MultiA1.pdf                          # Assignment 1 academic report
├── 2019JTM2207_2019JTM2088.pdf          # Assignment 2 academic report
├── PROGRAMMING ASSIGNMENT 1.pdf         # Assignment 1 assignment sheet
└── A2.pdf                               # Assignment 2 assignment sheet
```

---

## 2. Algorithms & Mathematical Foundations

### 1. Repetition Channel Coding & Error Correction
- **Encoding:** A bitstream $b_0, b_1, \dots$ is expanded into an $(r, 1)$ repetition codeword by repeating each bit $r$ times (e.g. for $r=3$, $0 \to 000$, $1 \to 111$).
- **Error Injection:** Uniformly injects Hamming weight errors to model noisy transmission channels.
- **Majority Logic Decoding:** For each $r$-bit block, counts the number of 1s. If $\sum \text{bits} > r/2$, decodes to $1$, else $0$. Successfully corrects up to $\lfloor (r-1)/2 \rfloor$ bit flips per codeword.

### 2. Exact Interval Arithmetic Coding
- **Probability Model:** Given symbol frequencies $f(s_i)$ in a message of length $N$, cumulative sub-intervals $[L(s_i), H(s_i)) \subset [0, 1)$ are defined.
- **Subdivision:** As each symbol is read, the active interval $[L, H)$ is narrowed:
  $$L_{new} = L + (H - L) \cdot L(s_i)$$
  $$H_{new} = L + (H - L) \cdot H(s_i)$$
- **Exact Precision:** Implemented using Python `fractions.Fraction` to eliminate floating-point underflow on arbitrary string lengths, guaranteeing 100% lossless reconstruction.

### 3. 2D Block Discrete Cosine Transform (DCT) & JPEG Quantization
- **Orthonormal Basis Matrix $T$:**
  $$T(i, j) = \begin{cases} \frac{1}{\sqrt{N}}, & i = 0 \\ \sqrt{\frac{2}{N}} \cos\left(\frac{(2j+1)i\pi}{2N}\right), & i > 0 \end{cases}$$
- **Block Transformation:** Forward DCT transforms 8x8 spatial pixel blocks $B$ into frequency domain $D$:
  $$D = T \cdot B \cdot T^T$$
  Because $T$ is orthonormal ($T \cdot T^T = I$), the inverse transform is:
  $$\hat{B} = T^T \cdot \hat{D} \cdot T$$
- **Quantization:** Standard JPEG Luminance quantization table $Q$ is scaled by quality factor $q \in [1, 100]$:
  $$D_q(u, v) = \text{round}\left(\frac{D(u, v)}{Q_{scaled}(u, v)}\right)$$
- **Zigzag Scan:** Arranges the 64 2D coefficients into a 1D sequence in ascending spatial frequency order to maximize trailing zero runs.
- **Fidelity Metrics:**
  $$\text{MSE} = \frac{1}{M \times N} \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} (I(x,y) - \hat{I}(x,y))^2$$
  $$\text{PSNR} = 10 \log_{10}\left(\frac{255^2}{\text{MSE}}\right) \text{ dB}$$

### 4. Dictionary Compression (LZ77, LZ78, LZW)
- **LZ77 (Sliding Window):** Employs a search buffer and lookahead buffer, emitting `(offset, length, next_char)` tuples.
- **LZ88 (Dynamic Tree):** Dynamically constructs phrase dictionaries from previously unseen prefixes, outputting `(dict_index, next_char)`.
- **LZW (Lempel-Ziv-Welch):** Initializes a codebook with 256 byte entries and builds multi-character phrase codes greedily. Handles the classic $cScSc$ boundary edge case during decompression.

---

## 3. Benchmarks & Experimental Results

### DCT Compression on `cat.jpg` (512x512 Grayscale)

| Quality Factor ($q$) | Mean Squared Error (MSE) | PSNR (dB) | Zero Coefficients (%) | Perceived Quality |
|:---:|:---:|:---:|:---:|:---|
| **25** | 38.45 | 32.28 dB | 88.62% | High Compression, mild blockiness |
| **50** | 24.12 | 34.31 dB | 84.15% | Standard JPEG baseline, balanced |
| **75** | 18.13 | 35.55 dB | 79.44% | High fidelity, sharp edges |
| **90** | 9.87 | 38.19 dB | 68.30% | Visually near-lossless |

### Dictionary Codecs on Sample Text (`"a.bar.array.by.barrayar.bay."`)
- **Original Length:** 28 characters (224 bits)
- **LZW Encoded Tokens:** 20 integer codes (**1.40x compression**)
- **Lossless Recovery:** 100% verified across LZ77, LZ78, and LZW

---

## 4. Usage & Execution

Execute the unified CLI runner via `uv`:

```bash
# 1. 2D-DCT Image Compression (default cat.jpg, quality 75)
uv run python image_compression/compress_image.py --algo dct --quality 75 --output image_compression/compressedcat1.jpg

# 2. Exact Arithmetic Coding
uv run python image_compression/compress_image.py --algo arithmetic --text "MULTIMEDIA SYSTEMS IIT DELHI"

# 3. LZW Dictionary Compression
uv run python image_compression/compress_image.py --algo lzw --text "a.bar.array.by.barrayar.bay."

# 4. LZ77 Sliding Window Compression
uv run python image_compression/compress_image.py --algo lz77 --text "TOBEORNOTTOBEORTOBEORNOT"

# 5. Repetition Channel Coding with 2-bit Error Correction
uv run python image_compression/compress_image.py --algo repetition --text "HELLO WORLD"
```

---

## 5. Automated Testing

All modules are covered by 14 automated unit tests under `tests/test_image_compression.py`:

```bash
uv run pytest tests/test_image_compression.py -v
```