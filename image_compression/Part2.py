# Library imported
import cv2
import math
import time
import numpy as np
from scipy import fft
from PIL import Image
from datetime import timedelta

# start timer to check time
start_time = time.monotonic()

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


# Function for DCT
def dctTransform(T, TT, matrix):
    # DCT using T
    dctimag = np.matmul(np.matmul(T, matrix), TT)
    return dctimag


# Fuction for IDCT
def inverseDctTransform(T, TT, matrix):
    # inverse DCT using T
    idctimag = np.matmul(np.matmul(TT, matrix), T)
    return idctimag


# Function for MSE calculation per pixel
def MSE(mat1, mat2):
    m, n = np.shape(mat1)
    MSE = np.sum(np.square(np.subtract(mat1, mat2)))
    mseperpixel = MSE / (m * n)
    return mseperpixel


################## Main Program #####################
threshold = int(input("Enter energy threshold: "))
path = r"/home/sudhanshu/Downloads/4thSem/Multimedia/As1/cat.jpg"
pathout = "/home/sudhanshu/Downloads/4thSem/Multimedia/As1/"

img = cv2.imread(path)  # read image
wname = "imag"  # window name
cv2.imshow(wname, img)
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
# print(img1)
m, n = np.shape(img1)
T = np.array(TransformMat(8))  # tansformation matrix of size 8X8
TT = np.transpose(T)  # transform of T
energy = np.zeros([8, 8])  # create matrix to store total energy


# Run the program for with block size of 8X8 across image of 512 X 512
for i in range(0, m, 8):
    for j in range(0, n, 8):
        img2 = img1[i : i + 8, j : j + 8]
        dctimag = dctTransform(T, TT, img2)  # DCT
        energy += np.square(dctimag)  # save energy

# calculate the average energy for 4096 blocks of 8X8 size
print()
avgenergy = np.divide(energy, 4096)
print("Average energy at each location of the 8 X 8 block:")
print(np.array(avgenergy.astype(int)))
print()
# map matrix with 0 and 1 to show which
# coefficients to keep and which to  dicard
mapmat = np.zeros([8, 8])
for i in range(8):
    for j in range(8):
        if avgenergy[i][j] < threshold:  # avgenergy less than 350 discard
            avgenergy[i][j] = 0
        else:
            mapmat[i][j] = 1

print("Average energy after discarding less energy coefficients:")
print((avgenergy.astype(int)))  # avgenergy matrix after discarding
print()
# from map matrix we get masked matrix
print("Map matrix:")
Mask = mapmat.astype(int)  # map matrix
print(Mask)
print()


# Run the program for with block size of 8X8 across image of 512 X 512
for i in range(0, m, 8):
    for j in range(0, n, 8):
        img2 = img1[i : i + 8, j : j + 8]
        dctimag = dctTransform(T, TT, img2)  # DCT
        maskedimg = np.multiply(Mask, dctimag)  # mask the DCT coeff.
        idctimag = inverseDctTransform(T, TT, maskedimg)  # inverse DCT
        # join the block to form complete image
        if j == 0:
            hrow = idctimag
        else:
            hrow = np.hstack((hrow, idctimag))
    if i == 0 and j == n - 8:
        fimage = hrow
    else:
        fimage = np.vstack((fimage, hrow))

print()
finalimage = fimage.astype(np.uint8)  # final image with uint8 datatype
# print(finalimage)

img = Image.fromarray(finalimage, "L")  # create image from matrix
img.show()  # show compressed image
# img.save(pathout + "compressedcat.jpg")  # save compreesed image using T DCT
img.save(pathout + "compressedcat1.jpg")  # save compressed image

mse = MSE(img1, finalimage)
print("MSE/pixel: ", mse)
print()

end_time = time.monotonic()
print(
    "Execution time: ", timedelta(seconds=end_time - start_time)
)  # time to run the program

cv2.waitKey(0)
cv2.destroyAllWindows()

