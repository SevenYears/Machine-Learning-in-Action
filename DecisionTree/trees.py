# -*- coding: UTF-8 -*-

from math import log
import operator
import matplotlib.pyplot as plt

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

def createTree(dataSet, oriLabels):
    '''
    构造决策树
    '''
    labels = oriLabels[:]
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

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, 
                            xy=parentPt, 
                            xycoords='axes fraction', 
                            xytext=centerPt, 
                            textcoords='axes fraction',
                            va="center",
                            ha="center",
                            bbox=nodeType,
                            arrowprops=arrow_args)

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    sencondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in sencondDict.keys():
        if type(sencondDict[key]).__name__ == 'dict':
            plotTree(sencondDict[key], cntrPt, str(key))  
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(sencondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD                          

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

def getNumLeafs(myTree):
    '''
    获取叶子节点个数
    '''
    numLeafs = 0
    firstStr = myTree.keys()[0]
    sencondDict = myTree[firstStr]
    for key in sencondDict.keys():
        if type(sencondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(sencondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    '''
    获取树的深度
    '''
    maxDepth = 0
    firstStr = myTree.keys()[0]
    sencondDict = myTree[firstStr]
    for key in sencondDict.keys():
        if type(sencondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(sencondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

def classify(inputTree, featLabels, testVec):
    '''
    使用建好的树分类
    '''
    firstStr = inputTree.keys()[0]
    sencondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in sencondDict.keys():
        if testVec[featIndex] == key:
            if type(sencondDict[key]).__name__ == 'dict':
                classLabel = classify(sencondDict[key], featLabels, testVec)
            else:
                classLabel = sencondDict[key]
    return classLabel

def storeTree(inputTree, filename):
    '''
    使用pickle存储树
    '''
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    '''
    从pickle文件中读取树
    '''
    import pickle
    fr = open(filename)
    return pickle.load(rf)

if __name__ == '__main__':
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    print lensesTree
    createPlot(lensesTree)
    print classify(lensesTree, lensesLabels, ['presbyopic', 'hyper', 'no', 'normal'])