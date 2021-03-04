# Library imported
import math
import numpy as np

# Function for DCT
def dctTransform(matrix):
    m, n = len(matrix), len(matrix[0])
    dct = np.zeros([m, n])
    for i in range(0, m):
        for j in range(0, n):
            if i == 0:
                ci = 1 / math.sqrt(m)
            else:
                ci = math.sqrt(2) / math.sqrt(m)
            if j == 0:
                cj = 1 / math.sqrt(n)
            else:
                cj = math.sqrt(2) / math.sqrt(n)
            summ = 0
            for k in range(0, m):
                for l in range(0, n):
                    cs1 = math.cos(((2 * k + 1) * i * math.pi) / (2 * m))
                    cs2 = math.cos(((2 * l + 1) * j * math.pi) / (2 * n))
                    summ = summ + (matrix[k][l] * cs1 * cs2)
            dct[i][j] = ci * cj * summ
    return dct


# Fuction for IDCT
def inverseDctTransform(matrix):
    m, n = len(matrix), len(matrix[0])
    inverseDct = np.zeros([m, n])
    for k in range(0, m):
        for l in range(0, n):
            summ = 0
            for i in range(0, m):
                for j in range(0, n):
                    if i == 0:
                        ci = 1 / math.sqrt(m)
                    else:
                        ci = math.sqrt(2) / math.sqrt(m)
                    if j == 0:
                        cj = 1 / math.sqrt(n)
                    else:
                        cj = math.sqrt(2) / math.sqrt(n)

                    cs1 = math.cos(((2 * k + 1) * i * math.pi) / (2 * m))
                    cs2 = math.cos(((2 * l + 1) * j * math.pi) / (2 * n))
                    summ = summ + (ci * cj * matrix[i][j] * cs1 * cs2)
            inverseDct[k][l] = summ
    return inverseDct


# Function to create transform matrix for DCT and IDCT
def TransformMat(m):
    T = np.zeros([m, m])
    for i in range(0, m):
        for j in range(0, m):
            cs1 = math.cos(((2 * j + 1) * i * math.pi) / (2 * m))
            if i == 0:
                T[i][j] = 1 / math.sqrt(m)
            else:
                T[i][j] = (math.sqrt(2) / math.sqrt(m)) * cs1
    return T


A = [
    [37, 39, 45, 57, 31, 35, 22, 18],
    [31, 38, 41, 41, 19, 30, 22, 18],
    [21, 39, 47, 33, 15, 27, 26, 23],
    [32, 48, 52, 31, 19, 27, 27, 22],
    [54, 57, 54, 31, 23, 25, 21, 14],
    [53, 55, 58, 42, 25, 19, 17, 17],
    [44, 46, 55, 46, 22, 17, 21, 29],
    [51, 41, 40, 34, 14, 19, 28, 35],
]

B = dctTransform(A)
print(B)
C = inverseDctTransform(B)
print(C)

T = TransformMat(8)
TT = np.transpose(T)
D = np.matmul(np.matmul(T, A), TT)
print(D)
E = np.matmul(np.matmul(TT, D), T)
print(E)
