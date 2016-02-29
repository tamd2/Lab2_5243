#Tim Taylor and Taumer Anabtawi
#CSE 5243
#Lab 2, clustering with K Means and DBScan

import math
import pickle
import sys

def ManhattanDistance(featureVector1, featureVector2):
    manDist = 0 #this is distance from vector1 to vector2
    for word in featureVector1:
        #look at each word in vector1
        value_original = featureVector1[word] #value of the word from the feat    ure vector
        value_neighbor = 0 #0, unless it is in the neighbor
        if word in featureVector2:
            value_neighbor = featureVector2[word]
        manDist += abs(value_original - value_neighbor)

    #loop again for words in vector 2 and not in vector 1
    for word in featureVector2:
        if not word in featureVector1:
            manDist += abs(featureVector2[word])

    return manDist

def get_euc_distance(vector1, vector2):
    #each vector is a dictionary
    distance_to_neighbor = 0 #this is distance from vector1 to vector2
    for word in vector1:
        #look at each word in vector1
        value_original = vector1[word] #value of the word from the feat    ure vector
        value_neighbor = 0 #0, unless it is in the neighbor
        if word in vector2:
            value_neighbor = vector2[word]
        distance_to_neighbor += math.pow(value_original - value_neighbor , 2) #(x1-x2)^2

    #loop again for words in vector 2 and not in vector 1
    for word in vector2:
        if not word in vector1:
            distance_to_neighbor += math.pow(vector2[word],2)
    distance_to_neighbor = math.sqrt(distance_to_neighbor)
    return distance_to_neighbor

#returns the overall entropy of this model
def OverallEntropy(clusterLabels, featureVectors, numClusters):
    #sort feature vector into clusters based off of cluster labels
    #clusteredFeaturedVectors will contain a list of feature vector lists
    #indexed by the cluster they belong to
    clusteredFeatureVectors = []
    for i in range(numClusters):
        clusteredFeatureVectors.append([])

    for featureIndex in clusterLabels:
        clusterIndex = clusterLabels[featureIndex]
        singleClusterDict = featureVectors[featureIndex]
        clusteredFeatureVectors[clusterIndex].append(singleClusterDict)
    
    overallEntropy = 0
    for i in range(len(clusteredFeatureVectors)):
        overallEntropy += clusterEntropy(clusteredFeatureVectors[i], i)

    return overallEntropy

def clusterEntropy(subFeatureVectorList, clusterIndex):
    #print("Cluster topic labels:")
    #print(str(subFeatureVectorList))
    #print("")

    wordFrequency = dict()
    numWords = 0
    #generate word frequency dictionary
    for fVector in subFeatureVectorList:
        for word in fVector:
            if (word not in wordFrequency):
                wordFrequency[word] = fVector[word]
            else:
                wordFrequency[word] += fVector[word]
            numWords += fVector[word]

    entropy = 0
    for word in wordFrequency:
        #mult by 1.0 to avoid integer division
        freq = 1.0 * wordFrequency[word] / numWords
        if (freq != 0):
            entropy += freq * math.log(freq, 2)

    return -entropy

def calculateSkew(clusterLabels, numClusters):
    if (numClusters == 1):
        print("only one cluster, skew is 0")
        return 0

    countList = dict()
    for clusterIndex in clusterLabels:
        clusterLabel = clusterLabels[clusterIndex]
        if clusterLabel in countList:
            countList[clusterLabel] += 1
        else:
            countList[clusterLabel] = 1

    #calculate avg number of points per cluster
    runningAvg = 0
    numClusters = 0
    for index in countList:
        runningAvg += countList[index]
        numClusters += 1
    avgPointsPerCluster = 1.0 * runningAvg / numClusters

    #calculate standard deviation
    variance = 0
    for index in countList:
        variance += pow((countList[index] - avgPointsPerCluster),2)
    variance = 1.0 * variance / numClusters
    stdev = pow(variance, .5)

    if (stdev == 0):
        return 0

    #calculate skew
    skew = 0
    for index in countList:
        count = countList[index]
        skew += pow((count - avgPointsPerCluster), 3)
    skew = 1.0 * skew / ((numClusters-1) * pow(stdev, 3))

    return skew

def read_in_preprocessed(fileName):
    dat_file_freq = open(fileName,"r")
    count_feature_vector = pickle.load(dat_file_freq)
    dat_file_freq.close()
    return count_feature_vector

######
#Main
######
if len(sys.argv) != 5:
    print("please pass as parameters epsilon, minPoints, the input file name, and 1 for manhattan distance or 2 for euclidian distance")
    quit()

epsilon = int(sys.argv[1])
minPoints = int(sys.argv[2])
inputFileName = str(sys.argv[3])
manOrEuc = int(sys.argv[4])
featureVectors = read_in_preprocessed(inputFileName)

print("creating proximity matrix")

#Create a proximity matrix, fill it up
numOfFeatureVectors=len(featureVectors)
proximityMatrix = [[-1 for x in range(numOfFeatureVectors)] for x in range(numOfFeatureVectors)]
for i in range(numOfFeatureVectors):
    print("filling proximity matrix row " + str(i) + " of " + str(numOfFeatureVectors))
    for j in range(numOfFeatureVectors):
		if not i is j:
                    if (proximityMatrix[j][i] != -1):
                        proximityMatrix[i][j] = proximityMatrix[j][i]
                    else:
                        if (manOrEuc == 1):
			                proximityMatrix[i][j] = ManhattanDistance(featureVectors[i], featureVectors[j])
                        elif (manOrEuc == 2):
                            proximityMatrix[i][j] = get_euc_distance(featureVectors[i], featureVectors[j])
                        else:
                            print("invalid distance metric selected, please use 1 or 2. Quitting...")
                            quit()

print("Classifying point types for DBScan")

#classify each point as core (count >= minPoints), border (between 1 and minPoints), or noise (not in classificationDict)
classificationDict = dict()
for i in range(numOfFeatureVectors):
    print("filling classification row " + str(i) + " of " + str(numOfFeatureVectors))
    for j in range(numOfFeatureVectors):
        if (i != j) and (proximityMatrix[i][j] <= epsilon):
            if (i in classificationDict):
                classificationDict[i] = classificationDict[i] + 1
            else:
                classificationDict[i] = 1

corePoints = []
print("searching for border points")
for index in classificationDict:
    if (classificationDict[index] >= minPoints):
        corePoints.append(index)        


print("clustering")
clusterLabels = dict()
currentClusterLabel=0
for index in corePoints:
    #bool to say to increment or not at the end of the core point loop
    incrementCurrentClusterLabel = 0
    if index not in clusterLabels:
        clusterLabels[index] = currentClusterLabel
        incrementCurrentClusterLabel = 1

    for i in range(numOfFeatureVectors):
        if (i != index) and (proximityMatrix[index][i] <= epsilon):
            if (i not in clusterLabels):
                clusterLabels[i] = currentClusterLabel
    if (incrementCurrentClusterLabel == 1):
        currentClusterLabel += 1

#print("feature vectors")
#for i in range(len(featureVectors)):
#    print(featureVectors[i])
#print("")

#print("Proximity matrix:")
#for i in range(len(proximityMatrix[0])):
#    print(str(proximityMatrix[i]))
#print("")

print("Cluster labels:")
print(str(clusterLabels))
print("")

#split up feature vectors into just the topic labels (words in all caps)
topicLabelFeatureVectors = []
for i in range(len(featureVectors)):
    topicLabelFeatureVectors.append(dict())
    for word in featureVectors[i]:
        if (str.isupper(str(word[0]))):
            topicLabel = topicLabelFeatureVectors[i]
            featureLabel = featureVectors[i]
            temp = featureLabel[word]
            topicLabel[word] = temp
        

#print("Topic label feature vectors:")
#for i in range(len(topicLabelFeatureVectors)):
#    print(str(topicLabelFeatureVectors[i]))

print("Entropy:")
print(str(OverallEntropy(clusterLabels, topicLabelFeatureVectors, currentClusterLabel+1)))

print("Skew:")
print(str(calculateSkew(clusterLabels, currentClusterLabel+1)))

print("Number of Clusters:")
print(str(currentClusterLabel+1))
