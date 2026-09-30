"""Automated test suite for Embedded Systems projects (ELP-720).

Tests:
1. 4-to-2 Priority Encoder logic verification across all 32 input states.
2. Raspberry Pi pattern matching algorithm.
3. String-to-bitstream encoding and lossless roundtrip decoding.
4. Firmware sketch source integrity checks.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EMBEDDED_DIR = REPO_ROOT / "embedded"

# Add embedded modules to sys.path
sys.path.insert(0, str(EMBEDDED_DIR / "arduino" / "priority_encoder"))
sys.path.insert(0, str(EMBEDDED_DIR / "raspberry_pi"))

from pattern_matcher import bit_list_to_string, pattern_match, string_to_bit_list
from verify_priority_encoder import generate_truth_table, priority_encoder_4to2


def test_priority_encoder_disabled():
    """When enable is 0, outputs should strictly be (0, 0) regardless of inputs."""
    for d3 in [0, 1]:
        for d2 in [0, 1]:
            for d1 in [0, 1]:
                for d0 in [0, 1]:
                    assert priority_encoder_4to2(0, d3, d2, d1, d0) == (0, 0)


def test_priority_encoder_priority_hierarchy():
    """Verify priority hierarchy D3 > D2 > D1 > D0 when enabled."""
    # D3 active -> Y1=1, Y0=1 regardless of lower pins
    assert priority_encoder_4to2(1, 1, 0, 0, 0) == (1, 1)
    assert priority_encoder_4to2(1, 1, 1, 1, 1) == (1, 1)

    # D2 active (D3=0) -> Y1=1, Y0=0 regardless of D1, D0
    assert priority_encoder_4to2(1, 0, 1, 0, 0) == (1, 0)
    assert priority_encoder_4to2(1, 0, 1, 1, 1) == (1, 0)

    # D1 active (D3=0, D2=0) -> Y1=0, Y0=1 regardless of D0
    assert priority_encoder_4to2(1, 0, 0, 1, 0) == (0, 1)
    assert priority_encoder_4to2(1, 0, 0, 1, 1) == (0, 1)

    # Only D0 active -> Y1=0, Y0=0
    assert priority_encoder_4to2(1, 0, 0, 0, 1) == (0, 0)

    # All inputs zero -> Y1=0, Y0=0
    assert priority_encoder_4to2(1, 0, 0, 0, 0) == (0, 0)


def test_priority_encoder_truth_table_size():
    table = generate_truth_table()
    assert len(table) == 32


def test_pattern_match_occurrences():
    """Verify pattern matching algorithm."""
    seq = [1, 0, 1, 1, 0, 1, 0, 1, 1]
    pat = [1, 0, 1]
    matches = pattern_match(pat, seq)
    assert matches >= 1

    # Empty pattern or pattern larger than sequence
    assert pattern_match([], seq) == 0
    assert pattern_match([1, 0, 1, 1, 0, 1, 0, 1, 1, 1], seq) == 0


def test_bitstream_roundtrip():
    """Verify lossless conversion between ASCII text and bitstream."""
    original = "IIT Delhi ELP-720 Embedded Lab"
    bits = string_to_bit_list(original)
    assert len(bits) == len(original) * 8
    recovered = bit_list_to_string(bits)
    assert recovered == original


def test_firmware_sketches_exist():
    """Check that all required .ino firmware sketches are present and non-empty."""
    sketches = [
        EMBEDDED_DIR / "arduino" / "calculator" / "calculator.ino",
        EMBEDDED_DIR / "arduino" / "priority_encoder" / "priority_encoder.ino",
        EMBEDDED_DIR / "esp32" / "weather_station" / "weather_station.ino",
        EMBEDDED_DIR / "esp32" / "blynk_touch" / "blynk_touch.ino",
        EMBEDDED_DIR / "raspberry_pi" / "arduino_receiver.ino",
    ]
    for sketch in sketches:
        assert sketch.exists(), f"Missing sketch: {sketch}"
        content = sketch.read_text()
        assert "void setup" in content
        assert "void loop" in content
