#Tim Taylor and Taumer Anabtawi
#CSE 5243
#Lab 5: Classification by Association Rules

import math
import pickle
import sys
import time

def read_in_preprocessed(fileName):
    dat_file_freq = open(fileName,"r")
    count_feature_vector = pickle.load(dat_file_freq)
    dat_file_freq.close()
    return count_feature_vector

#checks if all the words in the list of words "words" are in the vector "vector"
def all_words_are_in_vector(words, vector):
    present = 1
    for word in words:
        if not (word in vector):
            present = 0

    return present

def sort_rules(confidenceRulesRates, rules):
    newRules = []
    unusedIndices = range(len(rules))
    while (len(unusedIndices) > 0):
        maxConf = 0
        nextIndex = 0
        for i in range(len(unusedIndices)):
            index = unusedIndices[i]
            conf = confidenceRulesRates[str(rules[index])]
            if (conf > maxConf):
                maxConf = conf
                nextIndex = index
                
        newRules.append(rules[nextIndex])
        unusedIndices.remove(nextIndex)

    return newRules

#############
#Main
#############

if not ((len(sys.argv) == 5) or (len(sys.argv) == 6)):
    print("please pass as parameters the input file name, the minimum support value, the minimum confidence value, the percent of documents to train on and optionally pass the max number of feature vectors to use")
    quit()

inputFileName = str(sys.argv[1])
rawFeatureVectors = read_in_preprocessed(inputFileName)

minSupport = float(sys.argv[2])
minConfidence = float(sys.argv[3])
trainPercent = float(sys.argv[4])

if (trainPercent <= 0 or trainPercent >= 100):
    print("Please enter a number between 0 and 100 for the training percent")
    quit()

topicFeatureVectors = []
bodyFeatureVectors = []

bodyVectorElements = []
topicVectorElements = []

if len(sys.argv) == 6:
    numFeatureVectors = int(sys.argv[5])-1
else:
    numFeatureVectors = len(rawFeatureVectors)

numDocsToTrainOn = int(1.0*trainPercent*numFeatureVectors/100)

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

#create list of all words across all topic and feature vectors
wordElements = []
for word in bodyVectorElements:
    if (word not in wordElements):
        tempList = []
        tempList.append(word)
        wordElements.append(tempList)

for word in topicVectorElements: 
    if (word not in wordElements):
        tempList = []
        tempList.append(word)
        wordElements.append(tempList)

timeBuildModelStart = time.time()

#start gathering sets of support words
allSupportWords = []
while (len(wordElements) > 0):
    supportWords = []
    for wordList in wordElements:
        support = 0
        for i in range(numDocsToTrainOn):
            bodyFeatureVector = bodyFeatureVectors[i]
            topicFeatureVector = topicFeatureVectors[i]

            if (all_words_are_in_vector(wordList, bodyFeatureVector)):
                support += 1

        supportPercent = 100.0*support / numDocsToTrainOn

        if (supportPercent > minSupport):
            print("Found high support for word " + str(wordList) + " with support of " + str(supportPercent))
            supportWords.append(wordList)
            allSupportWords.append(wordList)

    #generate new list of word groupings to use in the next pass
    wordElements = []
    for wordList1 in supportWords:
        for wordList2 in supportWords:
            if (wordList1 != wordList2):
                newList = list(set(wordList1+wordList2))
                if (newList not in wordElements):
                    wordElements.append(newList)

print("support words:")
print(str(allSupportWords))

#generate list of possible rules out of remaining support words
possibleRules = []
for wordList1 in allSupportWords:
    for topicWord in topicVectorElements:
        tempList = []
        tempSubList = []
        tempSubList.append(topicWord)

        #first may imply the second
        tempList.append(wordList1)
        tempList.append(tempSubList)
        possibleRules.append(tempList)

print("Num possible rules: " + str(len(possibleRules)))

confidentRules = []
confidenceRulesRates = dict()
for rule in possibleRules:
    antecedantWords = rule[0]
    consequentWords = rule[1]

    confidenceHits = 0
    confidenceMisses = 0

    #check if that rule has high enough confidence
    for i in range(numDocsToTrainOn):
        bodyFeatureVector = bodyFeatureVectors[i]
        topicFeatureVector = topicFeatureVectors[i]
        
        if (all_words_are_in_vector(antecedantWords, bodyFeatureVector)):
            if (all_words_are_in_vector(consequentWords, topicFeatureVector)):
                confidenceHits += 1
            else:
                confidenceMisses += 1

    confidencePercent = confidenceHits*100.0/(confidenceHits+confidenceMisses)
    if (confidencePercent >= minConfidence):
        confidentRules.append(rule)
        confidenceRulesRates[str(rule)] = confidencePercent


sortedConfidentRules = sort_rules(confidenceRulesRates, confidentRules)

timeBuildModelEnd = time.time()
timeBuildModel = timeBuildModelEnd - timeBuildModelStart

print("Time to build model: " + str(timeBuildModel))

for rule in sortedConfidentRules:
    #antecedant words taken together imply the consequent words
    antecedantWords = rule[0]
    consequentWords = rule[1]
    print("Rule: " + str(antecedantWords) + " implies " + str(consequentWords) + " with confidence of " + str(confidenceRulesRates[str(rule)]) + "%")


for i in range(numDocsToTrainOn, numFeatureVectors):
    #check performance on these remaining feature vectors
    bodyFeatureVector = bodyFeatureVectors[i]
    topicFeatureVector = topicFeatureVectors[i]

    #for rule in sortedConfidentRules:
        #rules are in sorted order, so see if this rule applies
        #to the body feature vector. If so, only apply this rule
        #otherwise check the next rule to see if it applies

