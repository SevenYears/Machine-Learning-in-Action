# -*- coding: UTF-8 -*-

from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
import os

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance**0.5
    sortedDistIndicies = distance.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append((listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print "预测结果 %d, 真实结果 %d" % (classifierResult, datingLabels[i])
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print "错误率 %f" % (errorCount/(float(numTestVecs)))
 

def classifyPerson():
    resList = ['not at all', 'in small does', 'in large does']
    percentTats = float(raw_input("游戏时间百分比:"))
    ffMiles = float(raw_input("每年旅行飞行公里数:"))
    iceCream = float(raw_input("每年吃的冰激凌(升):"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print "结论: ", resList[int(classifierResult) - 1]

def img2vector(filename):
    returnVec = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            returnVec[0, 32*i+j] = int(line[j])
    return returnVec
    

def classifyDigit():
    labels = []
    trainData = os.listdir('digits/trainingDigits')
    m = len(trainData)
    print "训练数据集大小: %d" % m
    trainMat = zeros((m, 1024))
    for i in range(m):
        fileName = trainData[i]
        fileNameWithoutExt = fileName.split('.')[0]
        classNum = int(fileNameWithoutExt.split('_')[0])
        labels.append(classNum)
        trainMat[i, :] = img2vector('digits/trainingDigits/%s' % fileName)
    testData = os.listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testData)
    print "测试数据集大小: %d" % mTest
    for i in range(mTest):
        fileName = testData[i]
        fileNameWithoutExt = fileName.split('.')[0]
        classNum = int(fileNameWithoutExt.split('_')[0])
        testVec = img2vector('digits/testDigits/%s' % fileName)
        classifierResult = classify0(testVec, trainMat, labels, 3)
        print "预测结果: %s, 真实结果: %s" % (classifierResult, classNum)
        if classifierResult != classNum:
            errorCount += 1.0
    print "错误率: %f" % (errorCount/float(mTest))

if __name__=='__main__':
    # classifyPerson()
    classifyDigit()

