# Network Programming Using Internet Sockets

> **Course:** ELL-785 Computer Communication Networks  
> **Institution:** Bharti School of Telecommunication Technology and Management, IIT Delhi  
> **Authors:** Sudhanshu Chaudhary (2019JTM2207), Santosh Kumar (2019JTM2088)  
> **Report Document:** [`Socket Programming.pdf`](./Socket%20Programming.pdf)  
> **Implementation Language:** C (POSIX BSD Sockets, TCP/IP)

---

## 1. Project Overview

This project implements a secure, role-based Client-Server academic examination portal using raw Internet Sockets (TCP/IP) in C. The application manages examination marks for a class across 5 subjects (English, Mathematics, Physics, Chemistry, Biology) and provides tailored access permissions depending on client authentication.

### Core Objectives
1. **Authentication:** Pre-shared credentials validated on the server side against an access database (`data/user_pass.txt`).
2. **Role-Based Access Control (RBAC):**
   - **Student Access:** Upon logging in with student credentials, access is strictly limited to that student's records. Displays subject marks, aggregate percentage, and identifies highest and lowest scoring subjects.
   - **Instructor Access:** Upon logging in with instructor credentials (`instructor`), access is granted to all student records in the class. Displays the complete class gradebook and calculates overall class average.
3. **Network Performance & Wireshark Analysis:** Quantitative packet-level analysis measuring Round-Trip Time (RTT), TCP 3-way handshake overhead, frame sizes, and hop counts.

---

## 2. System Architecture & Protocol Design

### Client-Server Architecture

```
       +-----------------------+              +------------------------+
       |     Client (CLI)      |              |   Server (Daemon)      |
       +-----------------------+              +------------------------+
                   |                                       |
                   | ----- 1. TCP 3-Way Handshake -------> | [Listens on Port 4080]
                   |                                       |
                   | ----- 2. Send Username & Pass ------> | Authenticate against
                   |                                       | data/user_pass.txt
                   | <---- 3. Auth Status Header --------- |
                   |                                       | Query data/student_marks.txt
                   | <---- 4. Streamed Grade Data -------- |
                   |                                       |
                   | ----- 5. Close Connection ----------> | [Closes Socket]
                   v                                       v
```

### Protocol Exchange Specification

| Step | Sender | Message Format | Description |
|:---:|:---:|:---|:---|
| 1 | Client | `"<username>\n"` | Client transmits username token |
| 2 | Client | `"<password>\n"` | Client transmits password token |
| 3 | Server | `"<Auth Message>\n"` | `"Student logged in..."`, `"Instructor logged in..."`, or `"Invalid user credentials\n"` |
| 4a | Server (Student) | `"<name>\n<m1>\n...<m5>\n"` | 6 line-delimited tokens representing student identity and 5 subject marks |
| 4b | Server (Instructor) | `[<name>\n<m1>\n...<m5>\n] * 20` | Full stream of 20 students with marks |
| 5 | Both | `FIN / ACK` | Graceful TCP socket teardown |

---

## 3. Directory Layout

```
socket_programming/
├── Makefile                     # Compilation harness (gcc flags -Wall -Wextra -O2)
├── README.md                    # Project documentation
├── Socket Programming.pdf       # Original IEEE/IITD academic report
├── data/
│   ├── student_marks.txt        # Database of 20 students with 5 subject marks
│   └── user_pass.txt            # Credential database (usernames and passwords)
└── src/
    ├── client.c                 # C client program (interactive & CLI modes)
    └── server.c                 # Robust C server with file discovery & signal handling
```

---

## 4. Compilation & Build

The project includes a Makefile for standard Linux compilation with GCC:

```bash
# Build both server and client binaries
make -C socket_programming

# Remove compiled binaries
make -C socket_programming clean
```

---

## 5. Usage & Execution

### 1. Launching the Server
By default, the server binds to port `4080` on all available network interfaces (`0.0.0.0`):

```bash
# Run server with default port (4080)
./socket_programming/server

# Or specify a custom port
./socket_programming/server 5000
```

### 2. Client Queries

#### Interactive Mode:
```bash
./socket_programming/client
# Prompts interactively for Username and Password
```

#### CLI / Scripted Mode:
```bash
# Usage: ./client <host> <port> <username> <password>

# Student Login (Individual Marks Report):
./socket_programming/client 127.0.0.1 4080 sudhanshu s123

# Instructor Login (Class Gradebook & Average):
./socket_programming/client 127.0.0.1 4080 instructor i123

# Invalid Login (Access Rejection):
./socket_programming/client 127.0.0.1 4080 hacker wrongpass
```

### Sample Output: Student Query
```
Connected to server at 127.0.0.1:4080

Student logged in : Marks of the student only

====================================================
             STUDENT ACADEMIC REPORT                
====================================================
Student Name         : sudhanshu
Marks in Each Subject:
  - English      : 85 / 100
  - Mathematics  : 90 / 100
  - Physics      : 78 / 100
  - Chemistry    : 88 / 100
  - Biology      : 92 / 100
Aggregate Percentage : 86%
Highest Subject      : Biology (92 / 100)
Lowest Subject       : Physics (78 / 100)
====================================================
```

### Sample Output: Instructor Query
```
Connected to server at 127.0.0.1:4080

Instructor logged in : Marks of all students

-----------------------------------------------------------------------
                       INSTRUCTOR GRADEBOOK REPORT                     
-----------------------------------------------------------------------
Student #01: sudhanshu    | Eng: 85 | Math: 90 | Phy: 78 | Chem: 88 | Bio: 92 | Agg: 86%
Student #02: santosh      | Eng: 80 | Math: 85 | Phy: 75 | Chem: 82 | Bio: 88 | Agg: 82%
Student #03: akanksha     | Eng: 92 | Math: 88 | Phy: 95 | Chem: 90 | Bio: 94 | Agg: 91%
...
Student #20: aditya       | Eng: 65 | Math: 68 | Phy: 64 | Chem: 70 | Bio: 66 | Agg: 66%
-----------------------------------------------------------------------
Total Students Processed : 20
Class Average Percentage  : 79%
-----------------------------------------------------------------------
```

---

## 6. Automated Testing

Automated tests are implemented using Python and `pytest` under `tests/test_socket_programming.py`. Tests spin up dynamic server instances, assert protocol responses, verify mathematical averages, and test security validation.

Run the tests via `uv`:
```bash
uv run pytest tests/test_socket_programming.py -v
```

---

## 7. Network Performance & Wireshark Analysis

As documented in [`Socket Programming.pdf`](./Socket%20Programming.pdf) (Section 1.5, Figures 9–12):

1. **Protocol Stack Encapsulation:**
   - **Data Link Layer:** Ethernet II Frame (header: 14 bytes).
   - **Network Layer:** IPv4 Packet (header: 20 bytes).
   - **Transport Layer:** TCP Segment (header: 20–32 bytes with options).
   - **Application Layer:** Payload containing user credentials and examination records.
2. **Round Trip Time (RTT):**
   - Measured round-trip latency on loopback interface: **< 0.05 ms**.
   - Network path hops: **1 hop** (local host loopback `127.0.0.1`).
3. **TCP Connection Teardown:**
   - Clean 4-way FIN/ACK handshakes confirmed via Wireshark traces, avoiding lingering TCP TIME_WAIT socket exhaustion.
