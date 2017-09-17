# -*- coding: UTF-8 -*-

from math import log
import operator

def calcShannonEnt(dataSet):
    '''
    计算数据集的香农熵
    '''
    numEntries = len(dataSet)
    labelsCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelsCounts.keys():
            labelsCounts[currentLabel] = 0
        labelsCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelsCounts:
        prob = float(labelsCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def createDataSet():
    '''
    测试使用
    '''
    dataSet = [[1, 1, 'maybe'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    '''
    从数据集中获得 第axis个特征取值为value 子集
    '''
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    '''
    选择能使数据集熵降到最低的特征划分数据集
    '''
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestFeature = -1
    bestInfoGain = 0.0
    for i in range(numFeatures):
        # 获取第I个特征可以取的所有值
        featList = [data[i] for data in dataSet]
        uniqueVals = set(featList)
        tmpEntropy = 0.0
        for val in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, val)
            prob = len(subDataSet) / float(len(dataSet))
            tmpEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - tmpEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    '''
    多数表决法决定叶子节点分类结果
    '''
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    '''
    构造决策树
    '''
    classList = [example[-1] for example in dataSet]
    # 如果类别完全相同则递归结束，划分完成
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果所有特征都已经遍历完成，则采用多数法决定分类结果
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [data[bestFeat] for data in dataSet]
    uniqueVals = set(featValues)
    for val in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][val] = createTree(splitDataSet(dataSet, bestFeat, val), subLabels)
    return myTree

if __name__ == '__main__':
    myDat, labels = createDataSet()
    print calcShannonEnt(myDat)
    print chooseBestFeatureToSplit(myDat)
    print createTree(myDat, labels)