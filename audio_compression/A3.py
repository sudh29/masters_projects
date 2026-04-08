print(" ------------Multimedia Assignment 3---------")
print()
# Library imported
import time
import math
import wave
import struct
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt

# start timer to check time
start_time = time.monotonic()

####################################################

# New Quantizer Design for Defference value before Golam Encoder
def uniformQuantizer(Xdata, Bit, Delta):
    Nsteps = 2 ** (Bit - 1) - 1
    Q_Data = np.multiply(
        np.sign(Xdata).astype("int16"),
        np.minimum(np.floor(np.abs(Xdata) / Delta), Nsteps).astype("int16"),
    )
    return Q_Data


# Functon for predicting coefficients
def predictorCoeff(x, N):
    m = len(x)
    R = np.zeros((N, N))
    temp = []
    for k in range(N + 1):
        total = 0
        for i in range(1, m - k):
            total += x[i] * x[i + k]
        total = total / (m - k)
        temp.append(total)
    # print(temp, "temp")
    P = np.array(temp[1:])
    P.shape = N, 1
    # print(P)
    temp = temp[0:N]
    l = 0
    for i in range(len(R)):
        for j in range(len(R[0]) - l):
            R[i][j + l] = temp[j]
        l += 1
    # print(R)
    l = 1
    for i in range(1, len(R)):
        for j in range(l):
            R[i][j] = temp[l - j]
        l += 1
    # print(R)
    R_inv = np.linalg.inv(R)
    A = np.matmul(R_inv, P)
    # print(A)
    A = A.T[0]
    A = [round(i, 4) for i in A]
    return A


# Function for differnce using predictor
def predictorEncoder(A, x):
    N = len(A)
    pn_list = [0]
    diff_list = [x[0]]
    x_Q = [(x[0])]
    for n in range(1, len(x)):
        pn = 0
        for i in range(1, N + 1):
            if (n - i) < 0:
                xQ = 0
            else:
                xQ = x[n - i]
            pn += A[i - 1] * xQ
        pn_list.append(pn)
        diffQ = x[n] - pn
        diff_list.append(diffQ)
        xq = diffQ + pn
        x_Q.append(xq)
    return diff_list, pn_list


# Function for output using quantized differnece as input
def predictorDecoder(A, d_q):
    N = len(A)
    pn_list = [0]
    x_Q = [(d_q[0])]
    for n in range(1, len(d_q)):
        pn = 0
        for i in range(1, N + 1):
            if (n - i) < 0:
                xQ = 0
            else:
                xQ = d_q[n - i]
            pn += A[i - 1] * xQ
        pn_list.append(pn)
        x_Q.append(d_q[n] + pn)
    # print(x_Q)
    return x_Q, pn_list


# Function for encoder
def encoderDPCM(data):
    res = [data[0]]
    for i in range(1, len(data)):
        d_q = data[i] - data[i - 1]
        res.append(d_q)
    return res


# Function for decoder
def decoderDPCM(data):
    res = [(data[0])]
    for i in range(1, len(data)):
        temp = data[i] + res[i - 1]
        res.append(temp)
    return res


# Function for array to audio file
def write(data, path, s):
    sampleRate = s
    obj = wave.open(path, "w")
    obj.setnchannels(1)  # mono
    obj.setsampwidth(2)
    obj.setframerate(sampleRate)
    for i in data:
        d = struct.pack("<h", int(i * pow(2, 15)))
        obj.writeframesraw(d)
    obj.close()
    return


def SNR_PE(x, xi, p):
    m = len(x)
    num = 0
    deno = 0
    pe = 0
    for i in range(1, m):
        deno += pow((x[i] - xi[i]), 2)
        num += pow((x[i]), 2)
        pe += pow((x[i] - p[i]), 2)
    SNR = num / deno
    SPER = num / pe
    # print("SNR:", SNR, SPER, pe)
    return SNR, pe, SPER


def golombEncoder(n_list, m):
    golomb_code_list = []
    for n in n_list:
        n = int(n)
        q = n // m
        r = n % m
        quo = "1" * q + "0"
        b = math.floor(math.log2(m))
        k = 2 ** (b + 1) - m
        if r < k:
            rem = bin(r)[2:]
            l = len(rem)
            if l < b:
                rem = "0" * (b - l) + rem
        else:
            rem = bin(r + k)[2:]
            l = len(rem)
            if l < b + 1:
                rem = "0" * (b + 1 - l) + rem
        golomb_code = quo + rem
        golomb_code_list.append(golomb_code)
    # print(golomb_code_list)
    return golomb_code_list


def golombDecoder(code_list, m):
    s_list = []
    for code in code_list:
        k = math.ceil(math.log2(m))
        t = (2 ** k) - m
        s = 0
        for i in range(len(code)):
            if code[i] == "1":
                s += 1
            else:
                break
        x = code[i + 1 : i + k]
        x = int(x, 2)
        if x < t:
            s = s * m + x
            s_list.append(s)
        else:
            x = x * 2 + int(code[i + k :], 2)
            s = s * m + x - t
            s_list.append(s)
    # print(s_list)
    return s_list


def sign_of_data(data):
    if data > 0:
        return -1
    elif data == 0:
        return 0
    else:
        return 1


def outPredict(Encoded_data_val, Delta_val, pn_Tx_list, lenth_0f_dat):
    Quantized_out = np.zeros(lenth_0f_dat)
    D_rX = np.zeros(lenth_0f_dat)
    # D_rX= np.sum(np.multiply(Delta,Encoded_data_val),np.multiply(np.sign(Encoded_data_val),Delta/2))
    for i in range(1, lenth_0f_dat):
        D_rX[i] = (
            Delta_val * Encoded_data_val[i]
            + sign_of_data(Encoded_data_val[i]) * Delta_val / 2
        )
        Quantized_out[i] = pn_Tx_list[i] + D_rX[i]
    return Quantized_out


###################### Golomb Distributed Data Function #####################
def Gmap(Q_dif_value, lenth_0f_data):
    pDnch = np.zeros(lenth_0f_data)
    for i in range(len(Q_dif_value)):
        if Q_dif_value[i] < 0:
            pDnch[i] = 2 * abs(Q_dif_value[i]) - 1
        else:
            pDnch[i] = 2 * abs(Q_dif_value[i])
    return pDnch


def GmapInverse(Deco_out):
    lenth_0f_data = len(Deco_out)
    pDnch = np.zeros(lenth_0f_data)
    for i in range(len(Deco_out)):
        n = Deco_out[i]
        if n % 2 == 1:
            pDnch[i] = -(n + 1) / 2
        else:
            pDnch[i] = n / 2
    return pDnch


##################### Golomb Parameter M from given Data #####################
def GolamParameterM(D_data, length_of_data):
    A_Q_dif_value = np.unique(D_data)
    probaPV, bina = np.histogram(D_data, bins=A_Q_dif_value)
    prob_of_occurance = (probaPV) / (length_of_data)
    max_value = np.max(prob_of_occurance)
    find_index = np.where(np.isclose(prob_of_occurance, max_value))
    M = np.floor(-1 / math.log2(1 - prob_of_occurance[find_index[0][0] + 1])) + 1
    # print(M)
    return M


############## Reading audio file ###############
print("Choose the file:")
print("1: Music file")
print("2: Dialogue file")
print("3: Speech file")
print("4: Music file")

print("Enter input:")
x = input()
if x == "1":
    file = "0Music.wav"
elif x == "2":
    file = "1Dialogue.wav"
elif x == "3":
    file = "2Speech.wav"
elif x == "4":
    file = "3Music.wav"
else:
    print("Invalid option")

path = "/home/sudhanshu/Downloads/4thSem/Multimedia/As3/" + file
pathout = "/home/sudhanshu/Downloads/4thSem/Multimedia/As3/"

file = wave.open(path)
print("Channels:", file.getnchannels())
f = file.getframerate()
print("Framerate:", f)
duration = 1 / f
nf = file.getnframes()
t = nf / f
print("Time:", t)
t_seq = np.arange(0, t, duration)
data = file.readframes(-1)
wav_data_int16 = np.frombuffer(data, "Int16")
# wav_data_float32 = wav_data_int16.astype(np.float32)
# max_int16 = 2 ** 15  # Normalise float32 array so that values are between -1.0 and +1.0
# wav_data_float32_normalised = wav_data_float32 / max_int16
inputdata = wav_data_int16
print()

x = encoderDPCM(inputdata)
y = decoderDPCM(x)
# Function to plot
plt.plot(t_seq, wav_data_int16)
plt.xlabel("Time")
plt.ylabel("Xn")
plt.title("Audio signal")
plt.legend()
plt.show()


############## DPCM predictor coefficients  ###############

a1 = predictorCoeff(inputdata, 1)  # N=1
a2 = predictorCoeff(inputdata, 2)  # N=2
a4 = predictorCoeff(inputdata, 4)  # N=4

print("################ Prediction Coefficient  ################")
print("                                                           ")
print("Predictor coefficients for N=1:", a1)
print("Predictor coefficients for N=2:", a2)
print("Predictor coefficients for N=4:", a4)
print(" ")

###### Quantizing the Difference value Before Golam #####
SNR_1_list = []
SNR_2_list = []
SNR_4_list = []
Bits_list = []
for i in range(4, 12, 2):
    print("No_Bit_per_samples:", i)
    print("Bit Rate(kbps):", int((f * i) / 1000))
    print()
    No_Bit_per_samples = i
    Bits_list.append(No_Bit_per_samples)

    # N=1
    d1, pn1 = predictorEncoder(a1, inputdata)
    Delta1 = (2 * abs(max(d1))) / (2 ** No_Bit_per_samples)
    Q_dif_value1 = uniformQuantizer(d1, No_Bit_per_samples, Delta1)
    length_of_data1 = len(Q_dif_value1)
    D_data1 = Gmap(Q_dif_value1, length_of_data1)
    Parameter_M1 = GolamParameterM(D_data1, length_of_data1)
    # print("The Golomb Paraameter M for this Data N=1:", Parameter_M1)
    # Golomb Encoding
    g_D_data1 = golombEncoder(D_data1, int(Parameter_M1))
    # Golomb Decoding
    g_decoded1 = golombDecoder(g_D_data1, int(Parameter_M1))
    g_rescaled1 = GmapInverse(g_decoded1)
    op1, pn_op1 = predictorDecoder(a1, g_rescaled1)
    Quan_op1 = outPredict(Q_dif_value1, Delta1, pn1, length_of_data1)
    SNR_1, pe_1, SPER_1 = SNR_PE(inputdata, Quan_op1, pn_op1)
    SNR_1_list.append(SNR_1)

    # N=2
    d2, pn2 = predictorEncoder(a2, inputdata)
    Delta2 = (2 * abs(max(d2))) / (2 ** No_Bit_per_samples)
    Q_dif_value2 = uniformQuantizer(d2, No_Bit_per_samples, Delta2)
    length_of_data2 = len(Q_dif_value2)
    D_data2 = Gmap(Q_dif_value2, length_of_data2)
    Parameter_M2 = GolamParameterM(D_data2, length_of_data2)
    # print("The Golomb Paraameter M for this Data N=2:", Parameter_M2)
    # Golomb Encoding
    g_D_data2 = golombEncoder(D_data2, int(Parameter_M2))
    # Golomb Decoding
    g_decoded2 = golombDecoder(g_D_data2, int(Parameter_M2))
    g_rescaled2 = GmapInverse(g_decoded2)
    op2, pn_op2 = predictorDecoder(a2, g_rescaled2)
    Quan_op2 = outPredict(Q_dif_value2, Delta2, pn2, length_of_data2)
    SNR_2, pe_2, SPER_2 = SNR_PE(inputdata, Quan_op2, pn_op2)
    SNR_2_list.append(SNR_2)

    # N=4
    d4, pn4 = predictorEncoder(a4, inputdata)
    Delta4 = (2 * abs(max(d4))) / (2 ** No_Bit_per_samples)
    Q_dif_value4 = uniformQuantizer(d4, No_Bit_per_samples, Delta4)
    length_of_data4 = len(Q_dif_value4)
    D_data4 = Gmap(Q_dif_value4, length_of_data4)
    Parameter_M4 = GolamParameterM(D_data4, length_of_data4)
    # print("The Golomb Paraameter M for this Data N=4:", Parameter_M4)
    # Golomb Encoding
    g_D_data4 = golombEncoder(D_data4, int(Parameter_M4))
    # Golomb Decoding
    g_decoded4 = golombDecoder(g_D_data4, int(Parameter_M4))
    g_rescaled4 = GmapInverse(g_decoded4)
    op4, pn_op4 = predictorDecoder(a4, g_rescaled4)
    Quan_op4 = outPredict(Q_dif_value4, Delta4, pn4, length_of_data4)
    SNR_4, pe_4, SPER_4 = SNR_PE(inputdata, Quan_op4, pn_op4)
    SNR_4_list.append(SNR_4)

    print("################ The Prediction Error ################")
    print("Prediction Error for N=1:", pe_1)
    print("Prediction Error for N=2:", pe_2)
    print("Prediction Error for N=4:", pe_4)
    print()
    print("################ SNR(Signal to Noise Ratio) ################")
    print("SNR for N=1:", SNR_1)
    print("SNR for N=2:", SNR_2)
    print("SNR for N=4:", SNR_4)
    print()
    print("####### SPER( Signal-to-prediction-error ratio)###########")
    print("SPER for N=1:", SPER_1)
    print("SPER for N=2:", SPER_2)
    print("SPER for N=4:", SPER_4)
    print()

# Function to plot
font1 = {"family": "serif", "color": "blue", "size": 20}
font2 = {"family": "serif", "color": "darkred", "size": 15}
plt.plot(Bits_list, SNR_1_list, marker="o", label="N=1")
plt.plot(Bits_list, SNR_2_list, marker=".", label="N=2")
plt.plot(Bits_list, SNR_4_list, marker="*", label="N=4")
plt.xlabel("Bits", fontdict=font2)
plt.ylabel("SNR(dB)", fontdict=font2)
plt.title("SNR value for different predictor order N", fontdict=font1)
plt.legend()
plt.show()

# write(x, pathout + "out.wav", f)
####################################################
end_time = time.monotonic()
print()
print("Execution time: ", timedelta(seconds=end_time - start_time))
print()
####################################################
