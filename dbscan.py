#Tim Taylor and Taumer Anabtawi
#CSE 5243
#Lab 2, clustering with K Means and DBScan

from bs4 import BeautifulSoup
from nltk import PorterStemmer
from lxml import html
import urllib2
import math
import numpy
import pickle
import sys

def ManhattanDistance(featureVector1, featureVector2):
    manDist = 0
    wordVector1 = featureVector1.keys()
    wordVector2 = featureVector2.keys()

	#create list of all words across both vectors
    unionList = list(set(wordVector1) | set(wordVector2))

    for word in unionList:
        if (word in wordVector1):
            if (word in wordVector2):
                manDist += abs(featureVector1[word] - featureVector2[word])					
            else:
                manDist += abs(featureVector1[word])
        else: #it has to be in wordVector2 if it wasn't in wordVector1, otherwise it isn't in unionList
            if (word in wordVector1):
                manDist += abs(featureVector1[word] - featureVector2[word])
            else:
                manDist += abs(featureVector2[word])

    return manDist

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

#list of stop words
#found here: http://xpo6.com/list-of-english-stop-words/
stop_words = set(["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the", "reuter"])

num_pages = 22
num_documents = 21578 #found out by running code...better way to do this?

#master word list, containing all words from all docs without duplicates 
#stop words have been filtered out
master_word_list = list()

#words with a tf-idf score that is above the average tf-idf score (only relevant words)
relevant_word_list = list()

#word prescence feature vector
presence_feature_vector= [dict() for x in xrange(num_documents)] #creates a list of dictionaries, one row for each document (not page)

#word count feature vector
count_feature_vector= [dict() for x in xrange(num_documents)] #creates a list of dictionaries, one row for each document (not page)

#tf-idf feature vector
tf_idf_feature_vector= [dict() for x in xrange(num_documents)] #creates a list of dictionaries, one row for each document (not page)

#document list holder
document_word_lists = [[] for x in xrange(num_documents)] #creates a list of lists, one for each document

######
#Main
######
if len(sys.argv) != 4:
    print("please pass as parameters epsilon, minPoints, and the input file name")
    quit()

epsilon = int(sys.argv[1])
minPoints = int(sys.argv[2])
inputFileName = str(sys.argv[3])
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
			            proximityMatrix[i][j] = ManhattanDistance(featureVectors[i], featureVectors[j])

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
