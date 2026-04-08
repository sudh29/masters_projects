# ----------------Multimedia assignment 2-----------
print(" --------------Multimedia Assignment 2---------")
print()
# Library imported
import os
import cv2
import time
import numpy as np
from PIL import Image
from datetime import timedelta

# start timer to check time
start_time = time.monotonic()

path = "/home/sudhanshu/Downloads/4thSem/Multimedia/As2/1.bmp"
pathout = "/home/sudhanshu/Downloads/4thSem/Multimedia/As2/"

####################################################
# LWZ encoder function
def LZWencoder(seq, table):
    tsize = len(table) + 1
    encodedSeq = []
    prev_val = ""
    i = 0
    while i < len(seq):
        curr_val = str(seq[i])
        comp_val = prev_val + curr_val
        if comp_val in table:
            prev_val = comp_val
        elif len(table) == 512:
            encodedSeq.append(table[prev_val])
            prev_val = curr_val
        else:
            # print(table[prev_val])
            encodedSeq.append(table[prev_val])
            table[comp_val] = tsize
            tsize += 1
            prev_val = curr_val
        i += 1
    if i == len(seq):
        encodedSeq.append(table[prev_val])
    # print("Code length: ", len(encodedSeq))
    # print("Dictionary length: ", len(table))
    return encodedSeq, table


####################################################

# LWZ decoder function
def LZWdecoder(code, dic):
    decodeSeq = ""
    temp = ""
    for i in code:
        if len(dic) >= i:
            decodeSeq += dic[i - 1]
            temp += dic[i - 1][:1]
            if temp not in dic:
                dic.append(temp)
                temp = temp[-1]
            temp += dic[i - 1][1:]
    # print(decodeSeq)
    return decodeSeq


####################################################
# GIF encoder function
def GIFencoder(seq, table):
    table["clear"] = len(table)
    table["end"] = len(table)
    tsize = len(table)
    encodedSeq = [table["clear"]]
    prev_val = ""
    i = 0
    while i < len(seq):
        curr_val = str(seq[i])
        comp_val = prev_val + curr_val
        if comp_val in table:
            prev_val = comp_val
        elif len(table) == 4096:
            encodedSeq.append(table[prev_val])
            prev_val = curr_val
        else:
            encodedSeq.append(table[prev_val])
            table[comp_val] = tsize
            tsize += 1
            prev_val = curr_val
        i += 1
    if i == len(seq):
        encodedSeq.append(table[prev_val])
    encodedSeq.append(table["end"])
    # print("Code length: ", len(encodedSeq))
    # print("Dictionary length: ", len(table))
    return encodedSeq, table


####################################################
# GIF decoder function
def GIFdecoder(code, dic):
    dic.append("clear")
    dic.append("end")
    # print(dic)
    decodeSeq = ""
    temp = ""
    code = code[1:]
    for i in code:
        if len(dic) >= i:
            decodeSeq += dic[i - 1]
            temp += dic[i - 1][:1]
            if temp not in dic:
                dic.append(temp)
                temp = temp[-1]
            temp += dic[i - 1][1:]
    decodeSeq = decodeSeq[:-1]
    # print(decodeSeq)
    return decodeSeq


####################################################

dictionary = {"a": 1, "b": 2, "r": 3, "y": 4, ".": 5}
orignaldict = ["a", "b", "r", "y", "."]
inputstring = "a.bar.array.by.barrayar.bay."

print("Dictionary given: ", dictionary)
print("Input string:  ", inputstring)
a, b = LZWencoder(inputstring, dictionary)
c = LZWdecoder(a, orignaldict)
print("LWZ encoded val: ", a)
print("LWZ dictionary val: ", b)
print("Decoded string: ", c)
print()
####################################################

img = cv2.imread(path)  # read image
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
m, n = np.shape(img1)
img2 = img1.flatten().astype(str)  # convert to 1D
print("Image size: ", m * n)
print()
gray_dictionary = dict((str(i), i) for i in range(256))
compress_img, g_dic = LZWencoder(img2, gray_dictionary)
print("LWZ coded sequence size: ", len(compress_img))
print("LWZ dictionary size: ", len(g_dic))
print("Compression Ratio LWZ: ", len(img2) / len(compress_img))
print()
####################################################

compress_gif, gif_dic = GIFencoder(img2, dict((str(i), i) for i in range(256)))
print("GIF coded sequence size: ", len(compress_gif))
print("GIF dictionary size: ", len(gif_dic))
print("Compression Ratio GIF: ", len(img2) / len(compress_gif))
print()

####################################################
Image.open(path).save(pathout + "op.png")
s1 = os.path.getsize(path)
s2 = os.path.getsize(pathout + "op.png")
print("Orignal image size(bytes): ", s1)
print("PNG image size(bytes): ", s2)
print("Compression Ratio PNG: ", s1 / s2)
print()

####################################################
end_time = time.monotonic()
print("Execution time: ", timedelta(seconds=end_time - start_time))
print()
####################################################
