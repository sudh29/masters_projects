/*
 * Server side C program for ELL785 Computer Communication Networks Assignment 1
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
#include <netinet/in.h>
#include <signal.h>

#define DEFAULT_PORT 4080
#define BUFFER_SIZE 512

static int server_fd = -1;

void handle_signal(int sig) {
    (void)sig;
    if (server_fd >= 0) {
        close(server_fd);
    }
    printf("\n[Server] Shutting down gracefully.\n");
    exit(0);
}

/* Helper to resolve data files from multiple potential relative directories */
FILE *open_data_file(const char *filename) {
    char path[256];
    FILE *fp = fopen(filename, "r");
    if (fp) return fp;

    snprintf(path, sizeof(path), "data/%s", filename);
    fp = fopen(path, "r");
    if (fp) return fp;

    snprintf(path, sizeof(path), "../data/%s", filename);
    fp = fopen(path, "r");
    if (fp) return fp;

    snprintf(path, sizeof(path), "socket_programming/data/%s", filename);
    fp = fopen(path, "r");
    return fp;
}

/* Helper to read a single token delimited by newline/whitespace from stream */
static ssize_t read_token(int fd, char *buf, size_t maxlen) {
    size_t count = 0;
    while (count < maxlen - 1) {
        char c;
        ssize_t n = read(fd, &c, 1);
        if (n <= 0) break;
        if (c == '\n' || c == '\r' || c == '\0' || c == ' ') {
            if (count == 0) continue; // skip leading whitespace
            break;
        }
        buf[count++] = c;
    }
    buf[count] = '\0';
    return count;
}

int main(int argc, char *argv[]) {
    int port = DEFAULT_PORT;
    if (argc > 1) {
        port = atoi(argv[1]);
        if (port <= 0 || port > 65535) {
            fprintf(stderr, "Invalid port: %s. Using default %d\n", argv[1], DEFAULT_PORT);
            port = DEFAULT_PORT;
        }
    }

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    int opt = 1;
    struct sockaddr_in address;
    int addrlen = sizeof(address);

    // Creating socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("[Server] Socket creation failed");
        exit(EXIT_FAILURE);
    }

    // Forcefully attaching socket to the port (reuseaddr)
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("[Server] setsockopt failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("[Server] Bind failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd, 5) < 0) {
        perror("[Server] Listen failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    printf("====================================================\n");
    printf(" IIT Delhi - ELL785 Network Programming Server\n");
    printf(" Listening on 0.0.0.0:%d ...\n", port);
    printf("====================================================\n");

    while (1) {
        int new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen);
        if (new_socket < 0) {
            perror("[Server] Accept failed");
            continue;
        }

        char client_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &address.sin_addr, client_ip, sizeof(client_ip));
        printf("\n[Server] Connected to client %s:%d\n", client_ip, ntohs(address.sin_port));

        char user[64] = {0};
        char pwd[64] = {0};
        char user1[64] = {0};
        char pwd1[64] = {0};

        // Read username and password with framing
        if (read_token(new_socket, user, sizeof(user)) <= 0) {
            close(new_socket);
            continue;
        }

        if (read_token(new_socket, pwd, sizeof(pwd)) <= 0) {
            close(new_socket);
            continue;
        }

        printf("[Server] Login attempt - User: '%s'\n", user);

        int utype = (strcmp(user, "instructor") == 0) ? 1 : 2;
        int auth_success = 0;

        FILE *fptr = open_data_file("user_pass.txt");
        if (fptr == NULL) {
            fprintf(stderr, "[Server] Error opening user_pass.txt\n");
            char err_msg[] = "Server Error: Unable to access credentials database\n";
            send(new_socket, err_msg, strlen(err_msg), 0);
            close(new_socket);
            continue;
        }

        while (fscanf(fptr, "%63s %63s", user1, pwd1) != EOF) {
            if (strcmp(user1, user) == 0 && strcmp(pwd1, pwd) == 0) {
                auth_success = 1;
                break;
            }
        }
        fclose(fptr);

        if (!auth_success) {
            printf("[Server] Authentication FAILED for user: '%s'\n", user);
            char invalid_msg[] = "Invalid user credentials\n";
            send(new_socket, invalid_msg, strlen(invalid_msg), 0);
            close(new_socket);
            continue;
        }

        printf("[Server] Authentication SUCCESSFUL for user: '%s' (Type: %s)\n",
               user, (utype == 1) ? "Instructor" : "Student");

        if (utype != 1) {
            // Student logged in
            char print1[] = "Student logged in : Marks of the student only\n";
            send(new_socket, print1, strlen(print1), 0);
            usleep(20000); // 20ms pacing for TCP framing

            FILE *fptr1 = open_data_file("student_marks.txt");
            if (fptr1 == NULL) {
                fprintf(stderr, "[Server] Error opening student_marks.txt\n");
            } else {
                char student[64], m1[16], m2[16], m3[16], m4[16], m5[16];
                int found = 0;
                while (fscanf(fptr1, "%63s %15s %15s %15s %15s %15s", student, m1, m2, m3, m4, m5) != EOF) {
                    if (strcmp(student, user) == 0) {
                        found = 1;
                        dprintf(new_socket, "%s\n", student);
                        dprintf(new_socket, "%s\n", m1);
                        dprintf(new_socket, "%s\n", m2);
                        dprintf(new_socket, "%s\n", m3);
                        dprintf(new_socket, "%s\n", m4);
                        dprintf(new_socket, "%s\n", m5);
                        break;
                    }
                }
                fclose(fptr1);
                if (!found) {
                    printf("[Server] No marks record found for student: %s\n", user);
                }
            }
        } else {
            // Instructor logged in
            char print2[] = "Instructor logged in : Marks of all students\n";
            send(new_socket, print2, strlen(print2), 0);
            usleep(10000);

            FILE *fptr1 = open_data_file("student_marks.txt");
            if (fptr1 == NULL) {
                fprintf(stderr, "[Server] Error opening student_marks.txt\n");
            } else {
                char student[64], m1[16], m2[16], m3[16], m4[16], m5[16];
                while (fscanf(fptr1, "%63s %15s %15s %15s %15s %15s", student, m1, m2, m3, m4, m5) != EOF) {
                    dprintf(new_socket, "%s\n", student);
                    dprintf(new_socket, "%s\n", m1);
                    dprintf(new_socket, "%s\n", m2);
                    dprintf(new_socket, "%s\n", m3);
                    dprintf(new_socket, "%s\n", m4);
                    dprintf(new_socket, "%s\n", m5);
                }
                fclose(fptr1);
            }
        }

        close(new_socket);
        printf("[Server] Finished handling request for '%s'.\n", user);

        // In test mode or single-run flag if requested by argument:
        if (argc > 2 && strcmp(argv[2], "--single") == 0) {
            break;
        }
    }

    close(server_fd);
    return 0;
}
