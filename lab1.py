from bs4 import BeautifulSoup
from nltk import PorterStemmer
from lxml import html
import urllib2
import math
import numpy
import pickle
import random

#list of stop words
#found here: http://xpo6.com/list-of-english-stop-words/
stop_words = set(["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the", "reuter"])

num_pages = 22
num_documents = 1000#21578 #found out by running code...better way to do this?

#master word list, containing all words from all docs without duplicates 
#stop words have been filtered out
master_word_list = list()

#words with a tf-idf score that is above the average tf-idf score (only relevant words)
relevant_word_list = list()


#word prescence feature vector
#presence_feature_vector= [dict() for x in xrange(num_documents)] #creates a list of dictionaries, one row for each document (not page)
#presence_feature_vector = []

#word count feature vector
#count_feature_vector= [dict() for x in xrange(num_documents)] #creates a list of dictionaries, one row for each document (not page)
#count_feature_vector = []


#euclidean distance matrix
euc_distance_matrix = [list() for x in xrange(num_documents)] #creates a list of lists, one row for each document (not page)
#need to initialize to 0?


#read in pre-preprocessed feature vectors from dat files
def read_in_preprocessed():
	dat_file_freq = open("freq_dat","r")
	count_feature_vector = pickle.load(dat_file_freq)
	dat_file_freq.close()

	dat_file_pres = open("pres_dat","r")
	presence_feature_vector = pickle.load(dat_file_pres)
	dat_file_pres.close()

#cluster_list is the cluster with list values that are document numbers
def clusterEntropy(cluster_list, clusterIndex, input_feature_vector):
    #print("Cluster topic labels:")
    #print(str(subFeatureVectorList))
    #print("")

    wordFrequency = dict()
    numWords = 0
    #generate word frequency dictionary
    for doc in cluster_list:
	fVector = input_feature_vector[doc] #get the feature vector of this document
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

def calculateSkew(clusters, numClusters):
    if (numClusters == 1):
        print("only one cluster, skew is 0")
        return 0

    countList = dict()
    for index in xrange(numClusters):
        current_cluster = clusters[index]
	if len(current_cluster) > 0:
		countList[index] = len(current_cluster)

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

#helper function for k-means
def get_average_vector(list_of_documents, input_feature_vector):
	avg_dict = dict()
	number_dict = dict()
	for index in range(0,len(list_of_documents)):
		current_document_list = input_feature_vector[list_of_documents[index]]
		for k,v in current_document_list.iteritems():
		    #loop over every key and value in the list
		    #calculate mass sum of all frequencies of this word
		    if not k in avg_dict.keys():
			avg_dict[k] = 0
		    if not k in number_dict.keys():
			number_dict[k] = 0
		    avg_dict[k] += v
		    number_dict[k] += 1 #increase the count of this word
	#at this point, we have ran through each word and have a mass sum of all frequencies across all vectors in the list of documents given
	#find the average using the length
	for k,v in avg_dict.iteritems():
		avg_dict[k] = avg_dict[k] / float(number_dict[k])
	return avg_dict

#calculate euclidean distance
def get_euc_distance(vector1, vector2):
	#each vector is a dictionary
	distance_to_neighbor = 0 #this is distance from vector1 to vector2
	for word in vector1:
		#look at each word in vector1
		value_original = vector1[word] #value of the word from the feature vector
		value_neighbor = 0 #0, unless it is in the neighbor
		if word in vector2:
			value_neighbor = vector2[word]
		distance_to_neighbor += math.pow(value_original - value_neighbor , 2) #(x1-x2)^2

	#loop again for wrods in vector 2 and not in vector 1
	for word in vector2:
		if not word in vector1:
			distance_to_neighbor += math.pow(vector2[word],2)
	distance_to_neighbor = math.sqrt(distance_to_neighbor)
	return distance_to_neighbor

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


def k_means(num_centroid, input_feature_vector, distance_function, num_iterations):
	cluster_matrix = [list() for x in xrange(num_centroid)]

	#find centers
	if num_centroid > num_documents:
		num_centroid = num_documents #max centroids = number of documents
	#for now, choose random centroids?
	centroid_documents = random.sample(range(0,num_documents), num_centroid) #random number (unique) between 0 and num documents, we get num_centroid of these numbers


	#assign each centroid to a group
	#num_centroid = number of clusters
	for index in range(0,num_centroid):
		cluster_matrix[index].append(centroid_documents[index]) #centroid is first document in each cluster

	#NEED TO CALCUALTE EUCLIDEAN DISTANCES BETWEEN TWO POINTS.....
	#initialize centroid means
	mean_centroid_vectors =  [dict() for x in xrange(num_centroid)]
	for index in range(0,num_centroid):
		mean_centroid_vectors[index] = input_feature_vector[centroid_documents[index]] #grab the feature vector for this centroid document and put in order (it is of dict() format)


#first, go in and figure out based on current mean where each document fits (which cluster) based on centroid mean vector
#then, recalculate the mean
#then, re cluster each document and calculate mean again (dump all documents out of cluster and re calculate)
#keep going until mean converges (or for a number of iterations)
#after final iteration, see where each document is (in what cluster)
#that is the cluter that document belongs to

	for iteration in xrange(num_iterations):
		print("Iteration: ")
		print iteration
		#REPEAT for X iterations/convergence

		#reinitialize cluster list with only centroids
		cluster_matrix[:] = []
		cluster_matrix = [list() for x in range(0,num_centroid)]
		#for index in range(0,num_centroid):
			#cluster_matrix[index].append(centroid_documents[index])


		#process each list and cluster based on distance to centroid
		for index in range (0, num_documents):
			#if index not in centroid_documents:
				#no need to reprocess centroid
				distances_to_centroids = [-1]*num_centroid #length num_centroid
				for centroid in range(0,num_centroid):
					#for each centroid, calculate this documents distances to the centroid
					dis_to_centr = distance_function(input_feature_vector[index], mean_centroid_vectors[centroid])
					distances_to_centroids[centroid] = dis_to_centr
				#now have this documents distance to each centroid
				index_of_min = distances_to_centroids.index(min(distances_to_centroids)) # grab index of minimum
				#this index is the cluster number
				cluster_matrix[index_of_min].append(index) #add this document number into the cluster
	
		#recalculate cluster mean centroid distance
		for index in range(0,num_centroid):
			#calculate average vector from all documents in cluster
			average_vector = get_average_vector(cluster_matrix[index], input_feature_vector) #send in list of document numbers inside each cluster
			mean_centroid_vectors[index] = average_vector

		#print cluster_matrix

		#reinitialize cluster list with only centroids
		#if index != num_iterations:
			#if this is the final iteration, then what we have in cluster_matrix is what we want
		#	cluster_matrix[:] = []
		#	cluster_matrix = [list() for x in range(0,num_centroid)]
		#	for index in range(0,num_centroid):
		#		cluster_matrix[index].append(centroid_documents[index])
	
		#goto REPEAT	

	#calc SSE
	#do here because we have mean_vectors
	SSE = 0
	#iterate over every centroid (number of K = number of clusters)
	for index in range(0,num_centroid):
		#number of clusters
		current_cluster = cluster_matrix[index]
		#iterate over cluster and get distance to centroid, the current cluster is index
		for doc in current_cluster:
			#if doc != current_cluster[0]:
				distance_to_mean = get_euc_distance(mean_centroid_vectors[index], input_feature_vector[doc])
				#print distance_to_mean
				SSE += math.pow(distance_to_mean,2)

	print("K-means clustering complete")
	print("SSE value: ")
	print SSE



	return cluster_matrix
		

######END REPEAT
	#process using distance matrix other documents to fit inside clusters
	#for index in range(0, num_documents):
	#	if index not in centroid_documents:
	#		#only cluster this document if it is not a centroid itself
	#		document_distances = distance_matrix[index] #distances of this document to other documents
	#		distances_to_centroid = [-1 in range(0,num_centroids)] #initialize to -1			
	#		for centroid in range(0,num_centroids):
	#			current_centroid = centroid_documents[centroid] #centroid document number
	#			#need distance from current document to centroid
	#			#for euclidean, not all distances are stored in each list due to redudent calculations
	#			centroid_distances = distance_matrix[current_centroid] #distance list of the centroid document to other documents
	#			#look at the lower document number's list
	#			if current_centroid > index:
	#				#the correct distance is found by taking the desired document number - the current document number
					#this is because the list doesnt not contain all document numbers
					#in list 1 (2nd list), the 1st element represnets distance of DOC2 to DOC2
					#in the same list, the 2nd element represents distance of DOC2 to DOC3
	#				distances_to_centroid[centroid] = document_distances[centroid-index] #centroid here is an index. to find the document number of the centroid, look at centroid_documents[centroid]
	#			else:
					#if the distance is inside the centroid's distance matrix
	#				distances_to_centroid[centroid] = centroid_distances[centroid-index]

			#at this point, we have distances of this document to each centroid
			#place 

#calculate euclidean distance
#creates matrix in which each index brings up the distances for each document
#[[A][B][C]]
#[A] = {distance to a, distance to b, distance to c}
#[B] = {distance to b, distance to c}
def get_euc_data_matrix(input_feature_vector):
	#each list within matrix is the distance from that document to other documents
	#example list 0 has indicies 0-num_documents-1, each index holds value for distance to that document number
	for index in range (0,num_documents):	
		current_document_list = input_feature_vector[index]
		#there is no need to go back and calculate distance already calculated
		#example at index = 1, distance from 0 to 1 has been calculated. start calculating from 1 on, 
		#no need to look at 1 to 0 distance
		#include index to have 0 (distance of matrix to itself)
		for neighbor in range (index, num_documents):
			neighbor_list = input_feature_vector[neighbor] #grab neighbor list
			distance_to_neighbor = 0 #this is distance from current_document_list to neighbor
			for word in current_document_list:
				#look at each word to our current list
				value_original = current_document_list[word] 
				value_neighbor = 0 #0, unless it is in the neighbor
				if word in neighbor_list:
					value_neighbor = neighbor_list[word]
				distance_to_neighbor += math.pow(value_original - value_neighbor , 2) #(x1-x2)^2
			distance_to_neighbor = math.sqrt(distance_to_neighbor)
			#print euc_distance_matrix[index]
			euc_distance_matrix[index].insert(neighbor, distance_to_neighbor)
			#euc_distance_matrix[index][neighbor] = distance_to_neighbor #set matrix to this value
				


#main goes here	
print("Reading in feature vectors from pre-processing...")
dat_file_freq = open("datafiles/freq_dat_5000","r")
count_feature_vector = pickle.load(dat_file_freq)
dat_file_freq.close()

#dat_file_pres = open("datafiles/pres_dat","r")
#presence_feature_vector = pickle.load(dat_file_pres)
#dat_file_pres.close()
print("Reading in complete")

print("Euc matrix")
#get_euc_data_matrix(count_feature_vector)
#print get_euc_distance(count_feature_vector[1], count_feature_vector[1])
print("Ready to run K-Means clustering")
num_centroid = input("Enter desired K value:\n")
num_iter = input("Enter desired iterations:\n")
#k_means_result = k_means(num_centroid, count_feature_vector, get_euc_distance, num_iter)
k_means_result = k_means(num_centroid, count_feature_vector, ManhattanDistance, num_iter)
#print k_means_result
print ("Number of clusters: ")
clusters_num = sum(1 for non_empty in k_means_result if non_empty) #only non empty clusters count as clusters
print (clusters_num)

#entropy
print ("Entropy: ")
entropy = 0
for index in range(0,len(k_means_result)):
	if len(k_means_result[index]) > 0:
		entropy += clusterEntropy(k_means_result[index], index, count_feature_vector)
print (entropy)

print ("Sew: ")
print (calculateSkew(k_means_result, clusters_num))



