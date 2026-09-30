/*
 * Client side C program for ELL785 Computer Communication Networks Assignment 1
 * Network Programming Using Internet Sockets: Student Marks Examination System
 *
 * Authors: Sudhanshu Chaudhary (2019JTM2207), Santosh Kumar (2019JTM2088)
 * Course: ELL-785, Bharti School of Telecom, IIT Delhi
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>

#define DEFAULT_PORT 4080
#define BUFFER_SIZE 512

static const char *SUBJECT_NAMES[5] = {
    "English", "Mathematics", "Physics", "Chemistry", "Biology"
};

static ssize_t read_line(int fd, char *buf, size_t maxlen) {
    size_t count = 0;
    while (count < maxlen - 1) {
        char c;
        ssize_t n = read(fd, &c, 1);
        if (n <= 0) break;
        if (c == '\n') break;
        if (c != '\r') buf[count++] = c;
    }
    buf[count] = '\0';
    return count;
}

static ssize_t read_token(int fd, char *buf, size_t maxlen) {
    size_t count = 0;
    while (count < maxlen - 1) {
        char c;
        ssize_t n = read(fd, &c, 1);
        if (n <= 0) break;
        if (c == '\n' || c == '\r' || c == ' ' || c == '\0') {
            if (count == 0) continue;
            break;
        }
        buf[count++] = c;
    }
    buf[count] = '\0';
    return count;
}

int main(int argc, char *argv[]) {
    const char *host = "127.0.0.1";
    int port = DEFAULT_PORT;
    char user[64] = {0};
    char pwd[64] = {0};

    if (argc >= 3) {
        host = argv[1];
        port = atoi(argv[2]);
    }
    if (argc >= 5) {
        snprintf(user, sizeof(user), "%s", argv[3]);
        snprintf(pwd, sizeof(pwd), "%s", argv[4]);
    }

    int sock = 0;
    struct sockaddr_in serv_addr;
    char buffer[BUFFER_SIZE] = {0};

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("\n[Client] Socket creation error");
        return -1;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);

    if (inet_pton(AF_INET, host, &serv_addr.sin_addr) <= 0) {
        fprintf(stderr, "\n[Client] Invalid address / Address not supported: %s\n", host);
        close(sock);
        return -1;
    }

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("\n[Client] Connection Failed");
        close(sock);
        return -1;
    }

    printf("Connected to server at %s:%d\n", host, port);

    // Prompt for credentials if not supplied via command line
    if (strlen(user) == 0) {
        printf("Username : ");
        if (scanf("%63s", user) != 1) {
            close(sock);
            return -1;
        }
    }
    dprintf(sock, "%s\n", user);

    if (strlen(pwd) == 0) {
        printf("Password : ");
        if (scanf("%63s", pwd) != 1) {
            close(sock);
            return -1;
        }
    }
    dprintf(sock, "%s\n", pwd);

    // Read initial response header from server
    memset(buffer, 0, sizeof(buffer));
    ssize_t valread = read_line(sock, buffer, sizeof(buffer));
    if (valread <= 0) {
        printf("\n[Client] Server closed connection without response.\n");
        close(sock);
        return 0;
    }

    printf("\n%s\n", buffer);

    if (strstr(buffer, "Invalid user") != NULL) {
        printf("[Client] Access Denied: Incorrect username or password.\n");
        close(sock);
        return 1;
    }

    if (strcmp(user, "instructor") == 0) {
        // Instructor view
        int count = 0;
        int total_percentage_sum = 0;
        char student[64];
        char marks[5][16];

        printf("-----------------------------------------------------------------------\n");
        printf("                       INSTRUCTOR GRADEBOOK REPORT                     \n");
        printf("-----------------------------------------------------------------------\n");

        while (1) {
            if (read_token(sock, student, sizeof(student)) <= 0) break;

            int read_ok = 1;
            int sum = 0;
            for (int s = 0; s < 5; s++) {
                if (read_token(sock, marks[s], sizeof(marks[s])) <= 0) {
                    read_ok = 0;
                    break;
                }
                sum += atoi(marks[s]);
            }
            if (!read_ok) break;

            int per = sum / 5;
            total_percentage_sum += per;
            count++;

            printf("Student #%02d: %-12s | Eng: %2s | Math: %2s | Phy: %2s | Chem: %2s | Bio: %2s | Agg: %d%%\n",
                   count, student, marks[0], marks[1], marks[2], marks[3], marks[4], per);
        }

        printf("-----------------------------------------------------------------------\n");
        if (count > 0) {
            printf("Total Students Processed : %d\n", count);
            printf("Class Average Percentage  : %d%%\n", total_percentage_sum / count);
        }
        printf("-----------------------------------------------------------------------\n");

    } else {
        // Student view
        char student[64] = {0};
        char marks[5][16] = {{0}};

        if (read_token(sock, student, sizeof(student)) <= 0) {
            printf("[Client] No marks data received.\n");
            close(sock);
            return 0;
        }

        int mark_vals[5] = {0};
        int sum = 0;
        for (int s = 0; s < 5; s++) {
            if (read_token(sock, marks[s], sizeof(marks[s])) <= 0) break;
            mark_vals[s] = atoi(marks[s]);
            sum += mark_vals[s];
        }

        int max_val = mark_vals[0];
        int min_val = mark_vals[0];
        int max_sub = 0;
        int min_sub = 0;

        for (int s = 1; s < 5; s++) {
            if (mark_vals[s] > max_val) {
                max_val = mark_vals[s];
                max_sub = s;
            }
            if (mark_vals[s] < min_val) {
                min_val = mark_vals[s];
                min_sub = s;
            }
        }

        printf("====================================================\n");
        printf("             STUDENT ACADEMIC REPORT                \n");
        printf("====================================================\n");
        printf("Student Name         : %s\n", student);
        printf("Marks in Each Subject:\n");
        for (int s = 0; s < 5; s++) {
            printf("  - %-12s : %d / 100\n", SUBJECT_NAMES[s], mark_vals[s]);
        }
        printf("Aggregate Percentage : %d%%\n", sum / 5);
        printf("Highest Subject      : %s (%d / 100)\n", SUBJECT_NAMES[max_sub], max_val);
        printf("Lowest Subject       : %s (%d / 100)\n", SUBJECT_NAMES[min_sub], min_val);
        printf("====================================================\n");
    }

    close(sock);
    return 0;
}
