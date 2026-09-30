#!/usr/bin/env python3
"""ELP-720 Telecommunication Networks Laboratory - Assignment 8.

Raspberry Pi Serial & GPIO String Transmitter to Arduino.

Authors: Sudhanshu Chaudhary (2019JTM2207)
Institution: IIT Delhi
Platform: Raspberry Pi (Raspbian OS)
"""

import sys
import time

from pattern_matcher import string_to_bit_list


def send_serial(text: str, port: str = "/dev/ttyACM0", baud: int = 9600):
    """Sends string to Arduino via USB CDC / serial port."""
    try:
        import serial

        ser = serial.Serial(port, baud, timeout=2)
        time.sleep(2)  # Wait for Arduino reset upon connection
        encoded = text.encode("utf-8")
        ser.write(encoded)
        print(f"[RPi] Sent '{text}' ({len(encoded)} bytes) to Arduino on {port}")
        ser.close()
    except ImportError:
        print("[RPi] PySerial not installed; simulating serial transmission:")
        print(f"[RPi Mock] Wrote: {text}")
    except (OSError, RuntimeError) as e:
        print(f"[RPi Error] Serial transmission failed: {e}")


def send_gpio_bits(text: str, clock_pin: int = 10, data_pin: int = 11):
    """Bit-bangs encoded string bits to Arduino over GPIO pins."""
    bit_list = string_to_bit_list(text)
    print(f"[RPi] Bit representation of '{text}': {bit_list}")
    try:
        from RPi import GPIO

        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(clock_pin, GPIO.OUT)
        GPIO.setup(data_pin, GPIO.OUT)

        GPIO.output(clock_pin, GPIO.HIGH)
        for bit in bit_list:
            GPIO.output(data_pin, GPIO.HIGH if bit == 1 else GPIO.LOW)
            time.sleep(0.01)
        GPIO.output(clock_pin, GPIO.LOW)
        GPIO.cleanup()
        print(f"[RPi] Bit-banged {len(bit_list)} bits via GPIO.")
    except ImportError:
        print("[RPi] RPi.GPIO not available in host environment (expected on Raspberry Pi).")


if __name__ == "__main__":
    msg = sys.argv[1] if len(sys.argv) > 1 else "IIT Delhi - ELP720"
    print(f"Transmitting message: '{msg}'")
    send_serial(msg)
    send_gpio_bits(msg)
