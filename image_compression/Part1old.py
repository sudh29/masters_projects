import timeit
import random
import struct
import numpy as np


# Function to add error pattern uniformly with hamming weight d
def error_ran(s, w):
    n = len(s)
    s_list = []
    for i in range(n):
        s_list.append("0")
    for i in range(w):
        j = random.randint(0, n - 1)
        if s_list[j] == "0":
            s_list[j] = "1"
    list_str = "".join(map(str, s_list))
    return list_str


# Function to XOR two binary strings
def xor_string(a, b):
    m = len(a)
    n = len(b)
    result_str = ""
    if m == n:
        for i in range(m):
            if a[i] == b[i]:
                result_str += "0"
            else:
                result_str += "1"
    return result_str


# Fuction for string to 8bits binary list
def str_8bits_list(s=""):
    return [bin(ord(x))[2:].zfill(8) for x in s]


# Fuction to bits to character string
def bits2string(s):
    b = [(s[i : i + 8]) for i in range(0, len(s), 8)]
    return "".join([chr(int(x, 2)) for x in b])


# Function for percentage of character changed
def diff_letters(a, b):
    c = min(len(a), len(b))
    s = sum(a[i] != b[i] for i in range(c))
    d = len(a) - len(b)
    if d < 0:
        s = s
    else:
        s = s + d
    per_s = (float(s) / len(a)) * 100
    return per_s


# Function for creating chunks of size k
def chunk(m, k):
    chunk_list = [(m[i : i + k]) for i in range(0, len(m), k)]
    return chunk_list


# Function for Repetition Codes encoder
def repetitionCodes(data):
    res = ""
    r = 3
    for i in data:
        if i == "0":
            for j in range(r):
                res += "0"
        elif i == "1":
            for j in range(r):
                res += "1"
    return res


# Function for Repetition Codes decoder
def decodeRepetitionCodes(data):
    res = ""
    n = len(data)
    r = 3
    for i in range(0, n, r):
        if data[i] == "0":
            res += "0"
        elif data[i] == "1":
            res += "1"
    return res


def float_to_bin(num):
    return format(struct.unpack("!I", struct.pack("!f", num))[0], "032b")


def bin_to_float(binary):
    return struct.unpack("!f", struct.pack("!I", int(binary, 2)))[0]


def probrange(data):
    n = len(data)
    level = float(1 / n)
    str1 = list(data)
    str1.sort(reverse=True)
    d = {}
    for n in str1:
        keys = d.keys()
        if n in keys:
            d[n] = d[n] + level
        else:
            d[n] = level
    low = 0
    for k, v in d.items():
        d[k] = [low, low + v]
        low = d[k][1]
    val1 = d[" "]
    d[" "] = [val1[0], 1.0]
    # print(d)
    return d


def encodeAC(msg, freq):
    msglen = len(msg)
    low = 0
    high = 1
    for i in range(msglen):
        high1 = low + (high - low) * freq[msg[i]][1]
        low1 = low + (high - low) * freq[msg[i]][0]
        high = high1
        low = low1
    tag = (low + high) / 2
    taglen = len(str(tag)) - 2
    x = float_to_bin(tag)
    freqlist = []
    for k, v in freq.items():
        freqlist.append([k, v])
    # print("E", tag, x, taglen, msglen)
    return x, taglen, freqlist


def decodeAC(bincode, codelen, msglen, freq):
    tag = round(bin_to_float((bincode)), codelen)
    # print("D", tag, bincode, codelen)
    string = ""
    low = 0
    high = 1
    for i in range(msglen):
        t = (tag - low) / (high - low)
        for j in range(len(freq)):
            if t < freq[j][1][1] and t > freq[j][1][0]:
                string += freq[j][0]
                high = freq[j][1][1]
                low = freq[j][1][0]
                tag = t
                break
    # print(string)
    return string


################################################################

print("------------- Assignment1---------------")
print()
################## Open File #######################
try:
    with open("/home/sudhanshu/Downloads/4thSem/Multimedia/As1/data.txt", "r") as fpt:
        fdata = fpt.read()
except FileNotFoundError:
    fdata = None
# print("The string data from file : ", str(fdata))

################# String to binary ##################
fdata_bn = "".join(format(ord(i), "b").zfill(8) for i in fdata)
# print(fdata_bn)


#####################################################
print("Experiment-1: No Encoding")
# # Hamming weights
list_d = [10, 100, 200, 500, 5000]
e1 = []
# Generate error pattern of length M with hamming weight d
for i in range(0, 5):
    ran_str = error_ran(fdata_bn, int(list_d[i]))
    x_str = xor_string(fdata_bn, ran_str)  # XOR  with error pattern
    rx_fdata = bits2string(x_str)  # Binary to text
    print(
        "Percentage of modified characters with d :",
        int(list_d[i]),
        " is ",
        diff_letters(fdata, rx_fdata),
    )
    e1.append(diff_letters(fdata, rx_fdata))

######################################################
print()
print("Experiment-2: ")
print("Using Repetition coding")
e2 = []
c_data = chunk(fdata_bn, 4)  # chunk of size 4 for L2
encode_data2 = ""
for i in range(len(c_data)):
    x = str(c_data[i])
    encode_data2 += repetitionCodes(x)

# Generate error pattern of length M' with hamming weight d
for i in range(0, 5):
    ran_str1 = error_ran(encode_data2, int(list_d[i]))  # Uniform error
    x_str1 = xor_string(encode_data2, ran_str1)  # XOR
    rx_fdata1 = decodeRepetitionCodes(x_str1)  # Decode
    rx_txt1 = bits2string(rx_fdata1)
    print(
        "Percentage of modified characters with d :",
        int(list_d[i]),
        " is ",
        diff_letters(fdata, rx_txt1),
    )
    e2.append(diff_letters(fdata, rx_txt1))
#####################################################

print()
print("Using Arithmetic coding")

data = " .,zyxwvutsrqponmlkjihgfedcba"

x = probrange(data)
# for k, v in x.items():
#     print(k, v)

chunkdata = chunk(fdata, 4)
# print(chunkdata)

e3 = []
encodedata = []
for i in range(len(chunkdata) - 1):
    a, c, d = encodeAC(chunkdata[i], x)
    encodedata.append([a, c])

binstr = ""
for i in encodedata:
    binstr += i[0]


for i in range(0, 5):
    ran_str2 = error_ran(binstr, int(list_d[i]))  # Uniform error
    x_str2 = xor_string(binstr, ran_str2)  # XOR
    x_str2_list = chunk(x_str2, 32)
    output = ""
    for j in range(len(x_str2_list)):
        output += decodeAC(x_str2_list[j], encodedata[j][1], 4, d)

    print(
        "Percentage of modified characters with d :",
        int(list_d[i]),
        " is ",
        diff_letters(fdata, output),
    )
    e3.append(diff_letters(fdata, output))
    # print(output)
    output = ""


######################  Plot ############################
# importing matplotlib module
from matplotlib import pyplot as plt


# Function to plot
plt.plot(list_d, e1, label="No Encoding")
plt.plot(list_d, e2, label="Repetition code")
plt.plot(list_d, e3, label="Arithmetic code")

plt.xlabel("Value of d = {10, 100, 200, 500, 5000} ")
plt.ylabel("Percentage of modified characters")
plt.title("Assignment1")
plt.legend()
# function to show the plot
plt.show()

print()
print("-------End-------")
########################################################


