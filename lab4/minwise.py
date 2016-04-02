#Tim Taylor and Taumer Anabtawi
#CSE 5243
#Lab 3, Classification with Naive Bayes

import math
import pickle
import sys
import time
import random

def read_in_preprocessed(fileName):
    dat_file_freq = open(fileName,"r")
    count_feature_vector = pickle.load(dat_file_freq)
    dat_file_freq.close()
    return count_feature_vector

def compute_jaccard(binaryString1, binaryString2):
    intersectionCount = 0
    unionCount = 0

    for i in range(len(binaryString1)):
        if (binaryString1[i] == "1") and (binaryString2[i] == "1"):    
            intersectionCount += 1
            unionCount += 1
        elif (binaryString1[i] == "1") or (binaryString2[i] == "1"):
            unionCount += 1
        
    return 1.0*intersectionCount / unionCount

def getOverallSimilarityEstimate(documentString1, documentString2, orderToAccessArrays):
    sum = 0
    for orderToAccessArray in orderToAccessArrays:
        sum += getSimilarityEstimate(documentString1, documentString2, orderToAccessArray)

    estimate = (1.0*sum) / len(orderToAccessArrays)

    return estimate


def getSimilarityEstimate(documentString1, documentString2, orderToAccessArray):
    hashIndex1 = compute_hash(documentString1, orderToAccessArray)
    hashIndex2 = compute_hash(documentString2, orderToAccessArray)

    similarityEstimate = 0
    if (hashIndex1 == hashIndex2):
        similarityEstimate = 1

    return similarityEstimate

def compute_hash(documentString, orderToAccessArray):
    for i in range(len(orderToAccessArray)):
        indexToAccess = orderToAccessArray[i]
        if (documentString[indexToAccess] == "1"):
            return indexToAccess

    return -1 #no documents with no words a safe assumption?

def getRandomPermutation(length):
    initialOrder = range(length)
    for i in range(length*5):
        #swap two random indices
        index1 = random.randint(0,length-1)
        index2 = random.randint(0,length-1)
        
        temp = initialOrder[index1]
        initialOrder[index1] = initialOrder[index2]
        initialOrder[index2] = temp

    return initialOrder


#############
#Main
#############

if not ((len(sys.argv) == 2) or (len(sys.argv) == 3)):
    print("please pass as parameters the input file name, and optionally pass the max number of feature vectors to use")
    quit()

k_values = [16, 32, 64, 128, 256]

inputFileName = str(sys.argv[1])

rawFeatureVectors = read_in_preprocessed(inputFileName)

topicFeatureVectors = []
bodyFeatureVectors = []

bodyVectorElements = []
topicVectorElements = []

if len(sys.argv) == 3:
    numFeatureVectors = int(sys.argv[2])
else:
    numFeatureVectors = len(rawFeatureVectors)

print("Preparing feature vectors")
for i in range(numFeatureVectors):
    topicFeatureVectors.append(dict())
    bodyFeatureVectors.append(dict())

    for word in rawFeatureVectors[i]:
        #get the value of the word to pass along
        featureVector = rawFeatureVectors[i]
        featureValue = featureVector[word]
        if (str.isupper(str(word[0]))):
            topicFeatureVector = topicFeatureVectors[i]
            topicFeatureVector[word] = featureValue

            if word not in topicVectorElements:
                topicVectorElements.append(word)

        else:
            bodyFeatureVector = bodyFeatureVectors[i]
            bodyFeatureVector[word] = featureValue

            if word not in bodyVectorElements:
                bodyVectorElements.append(word)

#make every body word appear in each body feature vector 
for word in bodyVectorElements:
    for bodyFeatureVector in bodyFeatureVectors:
        if word not in bodyFeatureVector:
            bodyFeatureVector[word] = 0

#make every body word appear in each body feature vector 
for word in topicVectorElements:
    for topicFeatureVector in topicFeatureVectors:
        if word not in topicFeatureVector:
            topicFeatureVector[word] = 0

#compute the true similarity baseline

#generate an array of binary strings to represent each feature vector
binaryRepresentedBodyFeatureVectors = []
print("Generating binary representation")
for featureVector in bodyFeatureVectors:
    binaryFeatureVectorString = ""
    for i in range(len(bodyVectorElements)):
        bodyWord = bodyVectorElements[i]

        if featureVector[bodyWord] > 0:
            binaryFeatureVectorString += "1"
        else:
            binaryFeatureVectorString += "0"
    
    binaryRepresentedBodyFeatureVectors.append(binaryFeatureVectorString)

#compare each document to each other document and get the true jaccard similarity
numFeatureVectorsToCompare = len(binaryRepresentedBodyFeatureVectors)
baselineJaccard = []
baselineTimeStart = time.time()
print("Computing Baseline Jaccard Similarity")
for i in range(numFeatureVectorsToCompare):
    baselineJaccardRow = []
    for j in range(numFeatureVectorsToCompare):
        if (i == j):
            baselineJaccardRow.append("1")

        elif (i < j):
            jaccard = compute_jaccard(binaryRepresentedBodyFeatureVectors[i], binaryRepresentedBodyFeatureVectors[j])
            baselineJaccardRow.append(str(jaccard))
        else:
            baselineJaccardRow.append("0")
        
    #print(str(baselineJaccardRow))
    baselineJaccard.append(baselineJaccardRow)

baselineTimeTaken = time.time()-baselineTimeStart

#compute minwise hashing based similarity
minwiseTimesTaken = []
minwiseSims = []
for k in k_values:
    #get k permutations to work with
    permutations = []
    minwiseHashTimeStart = time.time()
 
    print("Computing minwise estimate for k="+str(k))
  
    for g in range(k):
        permutations.append(getRandomPermutation(len(binaryRepresentedBodyFeatureVectors[0])))

    minwiseSim = []
    for i in range(numFeatureVectorsToCompare):
        minwiseSimRow = []
        for j in range(numFeatureVectorsToCompare):
            if (i == j):
                minwiseSimRow.append("1")
            elif (i < j):
                estimate = getOverallSimilarityEstimate(binaryRepresentedBodyFeatureVectors[i], binaryRepresentedBodyFeatureVectors[j], permutations)
                minwiseSimRow.append(estimate)
            else:
                minwiseSimRow.append("0")
        #print(str(minwiseSimRow))
        minwiseSim.append(minwiseSimRow)
    
    minwiseHashTimeTaken = time.time()-minwiseHashTimeStart
    minwiseTimesTaken.append(str(minwiseHashTimeTaken))

    minwiseSims.append(minwiseSim)


print("Computing mean square error")
MSEs = []
for k in range(len(minwiseSims)):
    mse = 0
    for i in range(len(baselineJaccard[0])):
        for j in range(len(baselineJaccard[0])):
            if (i < j):
                minwiseSim = minwiseSims[k]
                difference = float(baselineJaccard[i][j]) - float(minwiseSim[i][j])
                mse += pow(difference , 2)
    MSEs.append(mse)

print("Times taken:")
print("Baseline: " + str(baselineTimeTaken))
for i in range(len(k_values)):
    print("Estimate using " + str(k_values[i]) + " permutations took this long: " + str(minwiseTimesTaken[i]))

print("")

print("MSEs:")
for i in range(len(k_values)):
    print("Estimate using " + str(k_values[i]) + " permutations has mse of: " + str(MSEs[i]))





