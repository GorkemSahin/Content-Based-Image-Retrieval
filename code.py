import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os, sys

def getImgDirectories(mainDir):
	print("Getting file paths...")
	imgTypes = os.listdir(mainDir)
	imgPaths = []
	for i in range(0, len(imgTypes)):
		imgTypePath = mainDir + "/" + imgTypes[i]
		for j in range(0, len(os.listdir(imgTypePath))):
			imgPaths.append(imgTypePath + "/" + os.listdir(imgTypePath)[j])
	return imgTypes, imgPaths

def prepareLbpHistogramData():
	print("Preparing LBP data...")
	imgTypes, imgPaths = getImgDirectories("CBIR")
	for i in range(0, len(imgPaths)):
		print("Creating and saving LBP data for: " + imgPaths[i].split("/", 1)[1])
		saveLbpHistogram(imgPaths[i])

def saveLbpHistogram(imgPath):
	img = cv.imread(imgPath, 0)
	img = cv.resize(img, (500, 500))
	histogram = createLbpHistogram(img)
	imgName = imgPath.rsplit("/", 1)[1]
	dataName = imgName.split(".", 1)[0] + ".txt"
	print("Saving LBP data... " + dataName)
	np.savetxt("LBPDATA/" + dataName, histogram)

def hasLessThanTwoTransitions(no):
	binary = '{0:08b}'.format(no)
	transitions = 0
	for i in range(0, len(binary)-1):
		if binary[i] != binary[i+1]:
			transitions += 1
	if transitions <= 2:
		return True
	else:
		return False

def createLbpHistogram(img):
	row, col = img.shape
	lbpValues = []
	for i in range(1, row-1):
		for j in range(1, col-1):
			pattern = 0
			if (img[i][j] > img[i-1][j-1]):
				pattern = pattern << 1
			else:
				pattern = pattern << 1
				pattern += 1
			if (img[i][j] > img[i][j-1]):
				pattern = pattern << 1
			else:
				pattern = pattern << 1
				pattern += 1
			if (img[i][j] > img[i+1][j-1]):
				pattern = pattern << 1
			else:
				pattern = pattern << 1
				pattern += 1
			if (img[i][j] > img[i+1][j]):
				pattern = pattern << 1
			else:
				pattern = pattern << 1
				pattern += 1
			if (img[i][j] > img[i+1][j+1]):
				pattern = pattern << 1
			else:
				pattern = pattern << 1
				pattern += 1
			if (img[i][j] > img[i][j+1]):
				pattern = pattern << 1
			else:
				pattern = pattern << 1
				pattern += 1
			if (img[i][j] > img[i-1][j+1]):
				pattern = pattern << 1
			else:
				pattern = pattern << 1
				pattern += 1
			if (img[i][j] > img[i-1][j]):
				pattern = pattern << 1
			else:
				pattern = pattern << 1
				pattern += 1
			lbpValues.append(pattern)
	lbpHistogram = np.zeros(256, dtype=int)
	for i in range(0, len(lbpValues)):
		if hasLessThanTwoTransitions(lbpValues[i]):
			lbpHistogram[lbpValues[i]] += 1
	lbpHistogram = normaliseHistogram(lbpHistogram, row*col)
	return lbpHistogram

def prepareHueHistogramData():
	imgTypes, imgPaths = getImgDirectories("CBIR")
	for i in range(0, len(imgPaths)):
		print("Creating and saving histogram data for: " + imgPaths[i].split("/", 1)[1])
		saveHueHistogram(imgPaths[i])

def saveHueHistogram(imgPath):
	img = cv.imread(imgPath)
	img = cv.resize(img, (500, 500))
	histogram = createHueHistogram(img)
	imgName = imgPath.rsplit("/", 1)[1]
	dataName = imgName.split(".", 1)[0] + ".txt"
	print("Saving Hue histogram data... " + dataName)
	np.savetxt("HUEDATA/" + dataName, histogram)

def getHue(b, g, r):
    r, g, b = r/255, g/255, b/255
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    h = int(h)
    return h

def normaliseHistogram(histogram, size):
	normalisedHistogram = []
	for i in range(0, len(histogram)):
		normalisedHistogram.append(histogram[i] / float(size))
	return normalisedHistogram

def createHueHistogram(img):
	row, col, channels = img.shape
	y = np.zeros(360, dtype=int)
	for i in range(0,row):
		for j in range(0,col):
			b, g, r = img[i, j]
			hue = getHue(b, g, r)
			y[hue] += 1
	y = normaliseHistogram(y, row*col)
	return y

def getHueHistogramData():
	print("Getting Hue histogram data...")
	dataPaths = os.listdir("HUEDATA")
	fileNames = []
	data = []
	for i in range(0, len(dataPaths)):
		fileNames.append(dataPaths[i].split(".", 1)[0])
		print("Reading... HUEDATA/" + dataPaths[i])
		data.append(np.loadtxt("HUEDATA/" + dataPaths[i]))
	return data, fileNames

def getLbpHistogramData():
	print("Getting LBP histogram data...")
	dataPaths = os.listdir("LBPDATA")
	fileNames = []
	data = []
	for i in range(0, len(dataPaths)):
		fileNames.append(dataPaths[i].split(".", 1)[0])
		print("Reading... LBPDATA/" + dataPaths[i])
		data.append(np.loadtxt("LBPDATA/" + dataPaths[i]))
	return data, fileNames

def calculateDifference(data, histogram):
	difference = 0
	for i in range(0, len(data)):
		difference += abs(data[i]-histogram[i])
	return difference

def classifyTestImgBasedOnHistogram(data, fileNames, histogram):
	differences = []
	for i in range(0, len(data)):
		differences.append(calculateDifference(data[i], histogram))
	closestFive = getClosestFive(differences)
	print("Closest images to the given test sample image: ")
	for i in range(0, len(closestFive)):
		print(fileNames[closestFive[i]])
	

def getClosestFive(differences):
	closest = sys.maxsize
	secondClosest = sys.maxsize
	thirdClosest = sys.maxsize
	fourthClosest = sys.maxsize
	fifthClosest = sys.maxsize
	for i in range(0, len(differences)):
		if differences[i] < closest:
			closest = differences[i]
		elif differences[i] < secondClosest:
			secondClosest = differences[i]
		elif differences[i] < thirdClosest:
			thirdClosest = differences[i]
		elif differences[i] < fourthClosest:
			fourthClosest = differences[i]
		elif differences[i] < fifthClosest:
			fifthClosest = differences[i]
	closestImgs = []
	closestImgs.append(differences.index(closest))
	closestImgs.append(differences.index(secondClosest))
	closestImgs.append(differences.index(thirdClosest))
	closestImgs.append(differences.index(fourthClosest))
	closestImgs.append(differences.index(fifthClosest))
	return closestImgs

def classifyTestImgBasedOnBothHistograms(hueData, lbpData, fileNames, hueHistogram, lbpHistogram):
	hueDifferences = []
	lbpDifferences = []
	totalDifference = []
	for i in range(0, len(hueData)):
		hueDifferences.append(calculateDifference(hueData[i], hueHistogram))
		lbpDifferences.append(calculateDifference(lbpData[i], lbpHistogram))
		totalDifference.append(hueDifferences[i] + lbpDifferences[i])
	closestFive = getClosestFive(totalDifference)
	print("Closest images to the given test sample image: ")
	for i in range(0, len(closestFive)):
		print(fileNames[closestFive[i]])

def main():
	#prepareHueHistogramData()
	#prepareLbpHistogramData()
	hueData, fileNames = getHueHistogramData()
	lbpData, fileNames = getLbpHistogramData()
	fileName = input("Path to the test image: ")
	img = cv.imread(fileName)
	if img is None:
		print("File not found.")
		return
	img = cv.resize(img, (500, 500))
	hueHistogram = createHueHistogram(img)
	img = cv.imread(fileName, 0)
	img = cv.resize(img, (500, 500))
	lbpHistogram = createLbpHistogram(img)
	print("Classifying the given image based on hue histogram data...")
	classifyTestImgBasedOnHistogram(hueData, fileNames, hueHistogram)
	print("Classifying the given image based on LBP histogram data...")
	classifyTestImgBasedOnHistogram(lbpData, fileNames, lbpHistogram)
	print("Classifying the given image based on both histograms...")
	classifyTestImgBasedOnBothHistograms(hueData, lbpData, fileNames, hueHistogram, lbpHistogram)
	input()
main()
