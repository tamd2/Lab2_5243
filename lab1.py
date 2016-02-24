from bs4 import BeautifulSoup
from nltk import PorterStemmer
from lxml import html
import urllib2
import math
import numpy
import pickle

#list of stop words
#found here: http://xpo6.com/list-of-english-stop-words/
stop_words = set(["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the", "reuter"])

num_pages = 22
num_documents = 4#21578 #found out by running code...better way to do this?

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


#calculate euclidean distance
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
dat_file_freq = open("freq_dat","r")
count_feature_vector = pickle.load(dat_file_freq)
dat_file_freq.close()

dat_file_pres = open("pres_dat","r")
presence_feature_vector = pickle.load(dat_file_pres)
dat_file_pres.close()
print("Reading in complete")

print("Euc matrix")
get_euc_data_matrix(count_feature_vector)
print euc_distance_matrix


