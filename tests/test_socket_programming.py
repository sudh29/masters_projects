"""Automated integration and unit test suite for Socket Programming (ELL-785 Assignment 1).

Tests:
1. Compilation of C server and client binaries via Makefile.
2. Server startup and port binding.
3. Student authentication and individual marks query (English, Math, Physics, Chem, Bio, percentage, max, min).
4. Instructor authentication and full student gradebook processing with class average.
5. Invalid credential rejection and security handling.
"""

import socket
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SOCKET_DIR = REPO_ROOT / "socket_programming"
SERVER_BIN = SOCKET_DIR / "server"
CLIENT_BIN = SOCKET_DIR / "client"


@pytest.fixture(scope="session", autouse=True)
def build_binaries():
    """Ensure server and client are compiled before running tests."""
    res = subprocess.run(
        ["make", "-C", str(SOCKET_DIR), "clean", "all"], capture_output=True, text=True, check=False
    )
    assert res.returncode == 0, f"Compilation failed:\n{res.stderr}"
    assert SERVER_BIN.exists()
    assert CLIENT_BIN.exists()


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture
def running_server():
    """Fixture that spins up the socket server on a dynamic port and tears it down."""
    port = str(get_free_port())
    proc = subprocess.Popen(
        [str(SERVER_BIN), port],
        cwd=str(SOCKET_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.3)  # Allow socket to bind and listen
    assert proc.poll() is None, "Server failed to start!"
    yield port
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_student_valid_login(running_server):
    port = running_server
    res = subprocess.run(
        [str(CLIENT_BIN), "127.0.0.1", port, "sudhanshu", "s123"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert res.returncode == 0
    out = res.stdout
    assert "Student logged in : Marks of the student only" in out
    assert "Student Name         : sudhanshu" in out
    assert "English      : 85 / 100" in out
    assert "Mathematics  : 90 / 100" in out
    assert "Physics      : 78 / 100" in out
    assert "Chemistry    : 88 / 100" in out
    assert "Biology      : 92 / 100" in out
    assert "Aggregate Percentage : 86%" in out
    assert "Highest Subject      : Biology (92 / 100)" in out
    assert "Lowest Subject       : Physics (78 / 100)" in out


def test_instructor_valid_login(running_server):
    port = running_server
    res = subprocess.run(
        [str(CLIENT_BIN), "127.0.0.1", port, "instructor", "i123"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert res.returncode == 0
    out = res.stdout
    assert "Instructor logged in : Marks of all students" in out
    assert "INSTRUCTOR GRADEBOOK REPORT" in out
    assert "sudhanshu" in out
    assert "santosh" in out
    assert "akanksha" in out
    assert "Total Students Processed : 20" in out
    assert "Class Average Percentage" in out


def test_invalid_credentials(running_server):
    port = running_server
    res = subprocess.run(
        [str(CLIENT_BIN), "127.0.0.1", port, "invalid_user", "wrong_password"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert res.returncode == 1
    assert "Invalid user credentials" in res.stdout
    assert "Access Denied" in res.stdout


def test_data_integrity():
    """Verify user_pass.txt and student_marks.txt consistency."""
    user_pass = SOCKET_DIR / "data" / "user_pass.txt"
    student_marks = SOCKET_DIR / "data" / "student_marks.txt"
    assert user_pass.exists()
    assert student_marks.exists()

    with open(user_pass) as f:
        users = [line.split()[0] for line in f if line.strip()]

    with open(student_marks) as f:
        students = [line.split()[0] for line in f if line.strip()]

    assert "instructor" in users
    assert len(students) >= 20
    # Every student in marks must have credentials
    for st in students:
        assert st in users, f"Student {st} missing credentials in user_pass.txt"
