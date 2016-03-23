#Tim Taylor and Taumer Anabtawi
#CSE 5243
#Lab 3, Classification with Naive Bayes

import math
import pickle
import sys
import time

def read_in_preprocessed(fileName):
    dat_file_freq = open(fileName,"r")
    count_feature_vector = pickle.load(dat_file_freq)
    dat_file_freq.close()
    return count_feature_vector

def convertIndicesToStringRepresentation(bodyIndex, topicIndex):
    inputKey = str(bodyIndex)+"~"+str(topicIndex)+"="
    return inputKey

def calculateProbability(pos, neg):
    prob = 0
    if (pos == 0) and (neg == 0):
        prob = 1 #unknown sample, try to be completely unbiased
    elif (neg != 0) or (pos != 0):
        prob = pos / (1.0*(pos + neg))

    return prob

#takes a feature vector and the prior probability dict and returns a dict for topic labels with predicted values. All topic labels
#not in this returned dict are assumed to be 0
def classifyFeatureVector(probabilityDict, bodyFeatureVector, topicVectorElements, bodyVectorElements):
    predictedFeatureVector = dict()
    for topicWord in topicVectorElements:
        predictedValue = classifyOneTopicElement(probabilityDict, bodyFeatureVector, topicVectorElements, bodyVectorElements, topicWord)
        predictedFeatureVector[topicWord] = predictedValue

    return predictedFeatureVector

#decide if, for a given body feature vector and topic label word, if that topic label word is more likely 1 or 0
def classifyOneTopicElement(probabilityDict, bodyFeatureVector, topicVectorElements, bodyVectorElements, topicWord):
    #get probability of topic element being 1
    probPos = getProbability(probabilityDict, bodyFeatureVector, topicVectorElements, bodyVectorElements, topicWord, 1)
    
    #get probability of topic element being 0
    probNeg = getProbability(probabilityDict, bodyFeatureVector, topicVectorElements, bodyVectorElements, topicWord, 0)
 
    #return 1 if it has a higher likelihood, 0 otherwise
    predictedValue = 0
    if probPos > probNeg:
        predictedValue = 1

    return predictedValue

def getProbability(probabilityDict, bodyFeatureVector, topicVectorElements, bodyVectorElements, topicWord, topicValue):
    probability = 1
    for bodyWord in bodyFeatureVector:
        #short circuit the calculations if it is alredy 0% chance
        if probability > 0:
            #get base key for querying probability dict
            bodyIndex = bodyVectorElements.index(bodyWord)
            topicIndex = topicVectorElements.index(topicWord)
            baseKey = convertIndicesToStringRepresentation(bodyIndex, topicIndex) 
            subProb = 0
            if (bodyFeatureVector[bodyWord] > 0):
                #get p(word=1 | topicWord = topicValue)
                subProb = probabilityDict[str(baseKey)+"1"+str(topicValue)]

            else:
                #get p(word=0 | topicWord = topicValue)
                subProb = probabilityDict[str(baseKey)+"0"+str(topicValue)]

            probability = probability * subProb
                
    #multiply by p(topicWord = topicValue)
    probability = probability * probabilityDict[str(topicWord+str(topicValue))] 
    return probability


#############
#Main
#############

if not (len(sys.argv) == 3 or len(sys.argv) == 4):
    print("please pass as parameters the input file name, and a float between and including 1 and 99 as the percent of the data to use as training data (all other data serves for testing). Optionally also pass the max number of feature vectors to profile across training and testing")
    quit()

inputFileName = str(sys.argv[1])
percentTrainingDocs = float(sys.argv[2])

#ensure that there are some training and some test documents
if (percentTrainingDocs < 1 or percentTrainingDocs > 99):
    print("Please enter a float from 1 to 99 for the second parameter")
    print(str(calculateProbability(1,1)))
    quit()

rawFeatureVectors = read_in_preprocessed(inputFileName)

#split up feature vectors into the topic label feature vectors and the body feature vectors
#also generate list of all topic label words and body words
if len(sys.argv) == 4:
    numFeatureVectors = int(sys.argv[3])
else:
    numFeatureVectors = len(rawFeatureVectors)

topicFeatureVectors = []
bodyFeatureVectors = []

bodyVectorElements = [] 
topicVectorElements = []
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


#Find the split point where feature vectors were for training but are now for testing
numTrainingDataFeatureVectors = int(percentTrainingDocs*.01*numFeatureVectors)
print("Training on "+str(numTrainingDataFeatureVectors+1)+" out of "+str(numFeatureVectors+1)+" documents")
startTrainingTime = time.time()

#calculate probability of each element given one feature vector topic label element
bodyAndLabelToProbabilityDict = dict()
for i in range(len(bodyVectorElements)):
    print("Calculating probability priors on body word " + str(i) +" of " + str(len(bodyVectorElements)))
    for j in range(len(topicVectorElements)):
        baseKeyString = convertIndicesToStringRepresentation(i, j)
        
        #find num of hits vs num of misses for feature i where j is 1 or 0 and sort accordingly
        #will use this data to come up with p(i|j), p(i|!j), p(!i|j), p(!i|!j)

        bodyWord = bodyVectorElements[i]
        topicWord = topicVectorElements[j]

        #counts for finding prob of a topic label
        posTopic = 0
        negTopic = 0

        #counts for finding prob of a word given a topic label
        posBodyPosTopic = 0
        posBodyNegTopic = 0
        negBodyPosTopic = 0
        negBodyNegTopic = 0

        for k in range(numTrainingDataFeatureVectors):
            bodyFeatureVetor = bodyFeatureVectors[k]
            topicFeatureVector = topicFeatureVectors[k]

            if (bodyFeatureVector[bodyWord] > 0) and (topicFeatureVector[topicWord] > 0):
                posBodyPosTopic += 1
                posTopic += 1
            elif (bodyFeatureVector[bodyWord] > 0) and (topicFeatureVector[topicWord] == 0):
                posBodyNegTopic += 1
                negTopic += 1
            elif (bodyFeatureVector[bodyWord] == 0) and (topicFeatureVector[topicWord] > 0):
                negBodyPosTopic += 1
                posTopic += 1
            else:
                negBodyNegTopic += 1
                negTopic += 1

        #p(i|j), p(!i|j)
        probPosPos = calculateProbability(posBodyPosTopic, negBodyPosTopic)
        probNegPos = 1-probPosPos
 
        #p(i|!j), p(!i|!j)
        probPosNeg = calculateProbability(posBodyNegTopic, negBodyNegTopic)
        probNegNeg = 1-probPosNeg

        #p(j), p(!j)
        probTopicPos = calculateProbability(posTopic, negTopic)
        probTopicNeg = 1-probTopicPos

        bodyAndLabelToProbabilityDict[baseKeyString+"11"] = probPosPos
        bodyAndLabelToProbabilityDict[baseKeyString+"10"] = probPosNeg
        bodyAndLabelToProbabilityDict[baseKeyString+"01"] = probNegPos
        bodyAndLabelToProbabilityDict[baseKeyString+"00"] = probNegNeg
        
        bodyAndLabelToProbabilityDict[topicWord+"1"] = probTopicPos
        bodyAndLabelToProbabilityDict[topicWord+"0"] = probTopicNeg 

stopTrainingTime = time.time()

#Test the classifier on the remaining documents
print("Testing on "+str(numFeatureVectors - numTrainingDataFeatureVectors)+" out of "+str(numFeatureVectors)+" documents")
startTestingTime = time.time()

numTruePositives = 0
numFalseNegatives = 0
numFalsePositives = 0
numTrueNegatives = 0

for k in range(numTrainingDataFeatureVectors, len(bodyFeatureVectors)):
    print("Testing model on feature vector " + str(k) + " of " + str(len(bodyFeatureVectors)))

    bodyFeatureVetor = bodyFeatureVectors[k]
    actualTopicFeatureVector = topicFeatureVectors[k]            
    predictedTopicFeatureVector = classifyFeatureVector(bodyAndLabelToProbabilityDict, bodyFeatureVector, topicVectorElements, bodyVectorElements)

    #all topicWords are in the predictedTopicFeatureVector
    for topicWord in predictedTopicFeatureVector:
        predictedValue = predictedTopicFeatureVector[topicWord]
        actualValue = 0
        if actualTopicFeatureVector[topicWord] > 0:
            #forcing it to 1 for simplicity
            actualValue = 1

        if (predictedValue == 1) and (actualValue == 1):
            numTruePositives += 1
        elif (predictedValue == 1) and (actualValue == 0):
            numFalsePositives += 1
        elif (predictedValue == 0) and (actualValue == 1):
            numFalseNegatives += 1
        else:
            #correctly classified as 0
            numTrueNegatives += 1

stopTestingTime = time.time()

timeTraining = -startTrainingTime + stopTrainingTime
timeTesting = -startTestingTime + stopTestingTime
numTestingDataFeatureVectors = int((100-percentTrainingDocs)*.01*numFeatureVectors)
avgTimeTesting = timeTesting / numTestingDataFeatureVectors


accuracy = (numTruePositives + numTrueNegatives) / (1.0*(numTruePositives + numTrueNegatives + numFalsePositives + numFalseNegatives))

print("Classification results:")
print("num true positives: " + str(numTruePositives))
print("num true negatives: " + str(numTrueNegatives))
print("num false positives: " + str(numFalsePositives))
print("num false negatives: " + str(numFalseNegatives))
print("accuracy: " + str(accuracy))
print("average time taken to classify an unknown tuple: " + str(avgTimeTesting))
print("time to build a model: " + str(timeTraining))
