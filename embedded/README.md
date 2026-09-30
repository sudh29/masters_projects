# Embedded Systems & IoT Laboratory Projects

> **Course:** ELP-720 Telecommunication Networks Laboratory  
> **Institution:** Bharti School of Telecommunication Technology and Management, IIT Delhi  
> **Author:** Sudhanshu Chaudhary (2019JTM2207)  
> **Target Hardware:** Arduino Uno / Nano, ESP32 Dev Module, Raspberry Pi 3/4  
> **Reports & Schematics:** Listed below per project

---

## 1. Directory Structure

```
embedded/
├── README.md                            # Comprehensive laboratory documentation
│
├── arduino/                             # Assignment 2: Microcontroller Fundamentals
│   ├── calculator/
│   │   └── calculator.ino               # Authenticated LCD Calculator with Keypad
│   └── priority_encoder/
│       ├── priority_encoder.ino         # 4-to-2 Hardware Priority Encoder
│       └── verify_priority_encoder.py   # Python truth table verification & simulator
│
├── esp32/                               # Assignments 4 & 5: IoT & Cloud Interfacing
│   ├── weather_station/
│   │   └── weather_station.ino          # ESP32 WiFi Weather Station & Telegram Bot
│   └── blynk_touch/
│       └── blynk_touch.ino              # Smart Appliance Monitor & Blynk IoT Cloud
│
├── raspberry_pi/                        # Assignment 8: Single-Board Computer Communication
│   ├── arduino_receiver.ino             # Arduino Serial Receiver Firmware
│   ├── pattern_matcher.py               # Bitstream pattern matching & encoding
│   └── rpi_sender.py                    # Serial & GPIO bit-banging transmitter
│
├── Arduino.pdf                          # Lab 2 Report: Arduino Calculator & Priority Encoder
├── Arduino_Dia.png                      # Lab 2 Circuit Schematic
├── ESP32.pdf                            # Lab 4 Report: ESP32 Weather Station & Telegram
├── ESP32_Dia.png                        # Lab 4 Circuit Schematic
├── ESP32_1.pdf                          # Lab 5 Report: ESP32 Blynk IoT & Touch Alert
├── Raspberry_Pi.pdf                     # Lab 8 Report: Raspberry Pi to Arduino Comm
└── Raspberry_Pi_Dia.png                 # Lab 8 Inter-board Communication Schematic
```

---

## 2. Laboratory Experiments & Implementations

### Experiment 1: Authenticated LCD Smart Calculator (Arduino Lab 2, PS1)
- **Report Reference:** [`Arduino.pdf`](./Arduino.pdf) (Section 1)
- **Source Code:** [`arduino/calculator/calculator.ino`](./arduino/calculator/calculator.ino)
- **Description:**
  - Implements PIN-protected access requiring a 4-digit code (`1234`) entered via a 4x4 matrix keypad. Passcode input is masked (`*`) on a 16x2 LCD display.
  - Upon authentication, users enter multi-digit operands terminated by the delimiter key `'c'`.
  - Supports basic arithmetic operations (`+`, `-`, `*`, `/`).
  - Results are displayed both in decimal on the LCD and in 4-bit binary format on 4 dedicated LEDs (Pins 10, 11, 12, 13).

### Experiment 2: 4-to-2 Hardware Priority Encoder (Arduino Lab 2, PS2)
- **Report Reference:** [`Arduino.pdf`](./Arduino.pdf) (Section 2)
- **Schematic:** [`Arduino_Dia.png`](./Arduino_Dia.png)
- **Source Code:** [`arduino/priority_encoder/priority_encoder.ino`](./arduino/priority_encoder/priority_encoder.ino)
- **Verification Script:** [`arduino/priority_encoder/verify_priority_encoder.py`](./arduino/priority_encoder/verify_priority_encoder.py)
- **Truth Table & Priority Hierarchy:**
  - Pin 3 acts as the master Enable switch ($E$). When $E = 0$, all outputs are $0$.
  - When $E = 1$, highest priority input overrides all lower-level inputs:

| Enable ($E$) | $D_3$ (Highest) | $D_2$ | $D_1$ | $D_0$ (Lowest) | $Y_1$ (MSB) | $Y_0$ (LSB) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | X | X | X | X | **0** | **0** |
| 1 | **1** | X | X | X | **1** | **1** |
| 1 | 0 | **1** | X | X | **1** | **0** |
| 1 | 0 | 0 | **1** | X | **0** | **1** |
| 1 | 0 | 0 | 0 | **1** | **0** | **0** |

---

### Experiment 3: ESP32 WiFi Weather Station & Telegram Bot (ESP32 Lab 4)
- **Report Reference:** [`ESP32.pdf`](./ESP32.pdf)
- **Schematic:** [`ESP32_Dia.png`](./ESP32_Dia.png)
- **Source Code:** [`esp32/weather_station/weather_station.ino`](./esp32/weather_station/weather_station.ino)
- **Key Features:**
  - Connects to institute WiFi (IITD WPA2-Enterprise) using 802.1x EAP credentials.
  - Polling Telegram Bot API via `UniversalTelegramBot` library. When a user sends a city name (e.g. `Delhi`, `London`), ESP32 queries the OpenWeatherMap REST API.
  - Displays real-time temperature, humidity, and atmospheric pressure on a 16x2 LCD display and replies to the Telegram chat.
  - Configures capacitive touch sensor interrupt on pin `T3` (GPIO 15) to detect physical enclosure tampering, triggering high-priority LED strobe alarms.

---

### Experiment 4: Smart Appliance Energy Monitor with Blynk Cloud (ESP32 Lab 5)
- **Report Reference:** [`ESP32_1.pdf`](./ESP32_1.pdf)
- **Source Code:** [`esp32/blynk_touch/blynk_touch.ino`](./esp32/blynk_touch/blynk_touch.ino)
- **Key Features:**
  - Two-way remote control of simulated household appliances (Lighting on Virtual Pin `V0`, Fan on Virtual Pin `V1`).
  - Real-time power calculation based on operational duration and wattage estimation ($P = W \times \Delta t$), logged to Blynk Virtual Pins `V8` and `V9`.
  - Capacitive touch security thresholding with automatic email notification generation upon intrusion.

---

### Experiment 5: Raspberry Pi & Arduino Inter-Board Serial Communication (Lab 8)
- **Report Reference:** [`Raspberry_Pi.pdf`](./Raspberry_Pi.pdf)
- **Schematic:** [`Raspberry_Pi_Dia.png`](./Raspberry_Pi_Dia.png)
- **Source Code:**
  - Raspberry Pi: [`raspberry_pi/rpi_sender.py`](./raspberry_pi/rpi_sender.py), [`raspberry_pi/pattern_matcher.py`](./raspberry_pi/pattern_matcher.py)
  - Arduino: [`raspberry_pi/arduino_receiver.ino`](./raspberry_pi/arduino_receiver.ino)
- **Key Features:**
  - Python string encoding into ASCII bitstreams.
  - Transmission over hardware UART / USB serial (`/dev/ttyACM0`) at 9600 baud.
  - Optional GPIO bit-banging protocol using clock and data lines.
  - Arduino interrupt-driven byte buffering and LCD string reconstruction.
  - Sliding-window pattern matching algorithm for signal decoding.

---

## 3. Flashing & Running Instructions

### Flashing Firmware (Arduino IDE / CLI)
1. Open any `.ino` sketch in Arduino IDE or compile via `arduino-cli`:
   ```bash
   # Example: compile priority encoder
   arduino-cli compile --fqbn arduino:avr:uno embedded/arduino/priority_encoder
   arduino-cli upload -p /dev/ttyACM0 --fqbn arduino:avr:uno embedded/arduino/priority_encoder
   ```
2. For ESP32 sketches, ensure the ESP32 board package is installed in Arduino IDE (`https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`).

### Running Raspberry Pi Scripts
```bash
# Run transmitter with sample message
python3 embedded/raspberry_pi/rpi_sender.py "Hello Arduino"
```

---

## 4. Automated Testing

The embedded logic, truth tables, and pattern matching algorithms are covered by unit tests in `tests/test_embedded.py`:

```bash
uv run pytest tests/test_embedded.py -v
```
