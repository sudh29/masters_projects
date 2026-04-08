import os
import cv2
import numpy
from os.path import isfile, join
from scipy.stats import multivariate_normal


# Function to initialiaze and updating parameters
def intialiseUpdateParams(update):
    global mu, sigma, wt, bgGaussianNo, fgGaussianNo
    # Initialising 3 gasuuians
    gaussianOrderList = numpy.zeros((1, 3))
    # Initialising mean, variance and wetghtd
    if update == 0:
        for row in range(mu.shape[0]):
            for col in range(mu.shape[1]):
                wt[row, col] = [1 / 3] * 3
                mu[row, col] = numpy.array([[130, 130, 130]]) * 3
                sigma[row, col] = [36.0] * 3
    # updaing mean, var and weights
    elif update == 1:
        # bgGaussianNo = numpy.ones((frameHeight,frameWidth))
        for row in range(mu.shape[0]):
            for col in range(mu.shape[1]):
                bgGaussianNo[row][col] = -1
                # Finding ratio
                gaussianOrderList = wt[row, col] / numpy.sqrt(sigma[row, col])
                # Sorting mean, var and weights
                # as  per ratio obtained
                sortedIndices = numpy.array(numpy.argsort(gaussianOrderList)[::-1])
                wt[row][col] = wt[row][col][sortedIndices]
                mu[row][col] = mu[row][col][sortedIndices]
                sigma[row][col] = sigma[row][col][sortedIndices]
                totalweight = 0
                for g in range(gaussianNum):
                    totalweight = totalweight + wt[row][col][g]
                    # Checking for threshold condition
                    # and separating out foreground gaussiaans
                    if totalweight >= threshold:
                        bgGaussianNo[row][col] = g
                        break
                    if wt[row][col][g] < threshold:
                        fgGaussianNo[row][col] = g

                if bgGaussianNo[row][col] == -1:
                    bgGaussianNo[row][col] = 2

                if fgGaussianNo[row][col] == -1:
                    fgGaussianNo[row][col] = 0


def applyingGMM(grayFrame):
    global mu, sigma, wt
    # Initialising frame that separates foreground and background
    bgFrame = numpy.zeros((frameHeight, frameWidth))
    fgFrame = numpy.zeros((frameHeight, frameWidth))
    for row in range(mu.shape[0]):
        for col in range(mu.shape[1]):
            matchedGaussian = -1
            pixelX = grayFrame[row][col]
            for gsNum in range(gaussianNum):
                frameIntsy = pixelX - mu[row, col, gsNum]
                invSigma = numpy.linalg.inv(sigma[row, col, gsNum] * numpy.eye(3))
                muSigma = numpy.dot(invSigma, frameIntsy)
                curveRange = 6.25 * sigma[row, col, gsNum]
                pixelRange = numpy.dot(frameIntsy.T, muSigma)

                # Checking if the gaussian falls in
                # specified range
                if pixelRange < curveRange:
                    matchedGaussian = gsNum
                    break

            if matchedGaussian != -1:
                # Updating mean, var and weights
                # if gaussian is matched
                wt[row, col] = (1 - learningRate) * wt[row, col]
                wt[row, col, matchedGaussian] = (
                    wt[row, col, matchedGaussian] + learningRate
                )
                rho = learningRate * multivariate_normal.pdf(
                    pixelX, mu[row, col, matchedGaussian], numpy.linalg.inv(invSigma)
                )
                prod = rho * numpy.dot(
                    (pixelX - mu[row, col, matchedGaussian]).T,
                    (pixelX - mu[row, col, matchedGaussian]),
                )
                rhoC = 1 - rho
                sigma[row, col, matchedGaussian] = (
                    rhoC * sigma[row, col, matchedGaussian] + prod
                )
                mu[row, col, matchedGaussian] = (
                    rhoC * mu[row, col, matchedGaussian] + rho * pixelX
                )

                # Separating foreground pixels b making it white
                if matchedGaussian > bgGaussianNo[row, col]:
                    bgFrame[row, col] = 250

                if matchedGaussian > fgGaussianNo[row, col]:
                    fgFrame[row, col] = 250

                if matchedGaussian == fgGaussianNo[row, col]:
                    fgFrame[row, col] = pixelX

            # If gaussian is not matched it means foreground
            # and separating it out
            else:
                bgFrame[row, col] = 250
                fgFrame[row, col] = 250
                mu[row, col, -1] = grayFrame[row][col]

    return bgFrame, fgFrame


def extractFrames(path, vidFile, out, oppathIn):
    # global mu,sigma,wt
    # videoFileObj = cv2.VideoCapture(vidFile)
    videoFileObj = cv2.VideoCapture(path + vidFile)
    frameCount = 0
    nextFrame = 1
    update = 0
    if videoFileObj.isOpened():
        extract = True
        # setting frame height and frame width
        videoFileObj.set(
            cv2.CAP_PROP_FRAME_WIDTH, frameWidth
        )  # set attribute to assign object dimensions to width and height variable used in code.
        videoFileObj.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
        intialiseUpdateParams(update)

    while extract and nextFrame:
        update = 1
        nextFrame, videoFrame = videoFileObj.read()
        frameCount += 1
        if nextFrame and frameCount % 100 == 0:
            print(frameCount)
            # Converting RGB frame to Grey Frame
            grayFrame = cv2.cvtColor(videoFrame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(
                os.path.join(out, "frame" + str(frameCount) + ".jpg"), grayFrame
            )
            intialiseUpdateParams(update)
            # Applying Mixture of Gaussians to grey frame
            outputFrame, outputFgFrame = applyingGMM(grayFrame)

            # Writing Output Frame with separated foreground pixels
            cv2.imwrite(
                os.path.join(oppathIn, "frame" + str(frameCount) + ".jpg"), outputFrame
            )
            cv2.imwrite(
                os.path.join(oppathInFG, "frameFg" + str(frameCount) + ".jpg"),
                outputFgFrame,
            )


# Function to convert Frames to Video
def framesToVideo(In, out, fps, flag):
    frame_array = []
    if flag == 1:
        fnm = "frame"
    else:
        fnm = "frameFg"

    for i in range(0, 900, 100):
        filename = In + fnm + str(i + 100) + ".jpg"
        print(i, filename)
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        frame_array.append(img)
    # Writing frames to Video
    out = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"DIVX"), fps, size)
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()


# Initialising parameters
frameHeight = 240
frameWidth = 320
gaussianNum = 3
threshold = 0.58
learningRate = 0.1
fps = 1
# Path to input and output frames
# pathIn = "C:\\Users\\krishna\\Desktop\\ML_Assign\\"
# pathOut = "C:\\Users\\krishna\\Desktop\\ML_Assign\\Frames\\"
# oppathIn = "C:\\Users\\krishna\\Desktop\\ML_Assign\\demobg\\"
# oppathInFg = "C:\\Users\\krishna\\Desktop\\ML_Assign\\demofg\\"
# oppathOut = "C:\\Users\\krishna\\Desktop\\ML_Assign\\demobg\\output.avi"
# oppathOutFg = "C:\\Users\\krishna\\Desktop\\ML_Assign\\demofg\\output1.avi"

pathIn = "/home/sudhanshu/Downloads/4thSem/ML/A1/"
pathOut = "/home/sudhanshu/Downloads/4thSem/ML/A1/Frames/"
oppathInBG = "/home/sudhanshu/Downloads/4thSem/ML/A1/OpFramesBG/"
oppathInFG = "/home/sudhanshu/Downloads/4thSem/ML/A1/OpFramesFG/"
oppathOutBG = (
    "/home/sudhanshu/Downloads/4thSem/ML/A1/BG_2019JTM2674_2019JTM2207_2019JTM2088.avi"
)
oppathOutFG = (
    "/home/sudhanshu/Downloads/4thSem/ML/A1/FG_2019JTM2674_2019JTM2207_2019JTM2088.avi"
)

# Defining mean, var and weights
mu = numpy.zeros((frameHeight, frameWidth, gaussianNum, 3))
bgGaussianNo = numpy.zeros((frameHeight, frameWidth))
fgGaussianNo = numpy.zeros((frameHeight, frameWidth))
sigma = numpy.zeros((frameHeight, frameWidth, gaussianNum))
wt = numpy.zeros((frameHeight, frameWidth, gaussianNum))

# Main Function
if __name__ == "__main__":
    vidFile = "umcp.mpg"
    extract = True
    extractFrames(pathIn, vidFile, pathOut, oppathInBG)
    framesToVideo(oppathInBG, oppathOutBG, fps, 1)
    framesToVideo(oppathInFG, oppathOutFG, fps, 0)

