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


#split block of text into words by space
def split_words(text):
	return text.split()

#clean up <D> tag and </D> form places and topics
#start tag is either <topics> or <places>
def clean_up_d_tag(text, start_tag, end_tag):
	new_list = list()
	#right now, text is just line with all tags
	#remove start tag and corresponding endtag
	start_tag_len = len(start_tag) #grab length of tag
	start_index = str(text).find(start_tag)	#find index of start tag
	end_index = str(text).find(end_tag)	#index of end tag
	removed_start_end_tag = str(text)[start_index+start_tag_len:end_index] #take off start and end tag in substring (topic/place)
	with_end_tag = removed_start_end_tag.split("<d>") #remove <d>
	for word in with_end_tag:
		#error with splitting into a space, remove it
		if len(word) > 0: 
			position = word.find("</d>")
			new_list.append(word[:position]) #split word up to position of end tag, substring essentially
	return new_list
		

#filter out stop words
def filter_out_words(unfiltered_set):
	return [word for word in unfiltered_set if word not in stop_words]
	#new_list = list()
	#for word in unfiltered_set:
	#	if word not in stop_words:
	#		new_list.append(word)
	#return new_list

#take each word and make it lower case
def make_lower_case(list_of_words):
	new_list = list()
	for word in list_of_words:
		word = word.lower()
		new_list.append(word)
	list_of_words[:] = [] #delete list
	return new_list

#take each word and make it upper case
def make_upper_case(list_of_words):
	new_list = list()
	for word in list_of_words:
		word = word.upper()
		new_list.append(word)
	list_of_words[:] = [] #delete list
	return new_list

#filter words by their stem only
def get_word_stems(list_of_words):
	new_list = list()
	for word in list_of_words:
		word = PorterStemmer().stem_word(word) #find stem of word
		new_list.append(word) #readd word as just stem
	list_of_words[:] = [] #delete list
	return new_list

#remove numbers from list
def remove_numbers(list_of_words):
	new_list = list()
	for word in list_of_words:
		if not word[0].isdigit():
			#if the first character is a number, disregard this word
			#I can forsee a problem if the name of a company starts with a number
			#but that is rare
			new_list.append(word)
	list_of_words[:] = [] #delete list
	return new_list

#grab tf for each word inside the world list and place in a dictionary 
def tf_dict(list_of_words):
	tf_dictionary = dict()
	for word in list_of_words:
		tf_dictionary[word] = term_freq(word, list_of_words)
	return tf_dictionary

#filter by term freq, if tf is < average, remove
def filter_by_tf(list_of_words, tf_dictionary):
	new_list = list()
	average = avg_tf(tf_dictionary) #use dictionary because need single instance and value representing all instances of that word
	std = numpy.std(tf_dictionary.values())
	for word in list_of_words:
		if tf_dictionary[word] >= average + std :
			#if the word has a tf value greater than the average tf value
			#it appears a lot in the document, keep it
			new_list.append(word)
	list_of_words[:] = [] #delete list	
	return new_list
	
#********************

#go through each word in word list for that document
#count up and store in its own row
#no need to look at "master" word list (all worrds in all documents)
#if the word is present in the document, it will be found in its own set of words
#checking the master word list would be inefficent and uneeded
#documents words is "words" dataset
def process_word_count_prescence(current_doc, words):
	for word in words:
		occurrences = words.count(word)
		#current page is index of row
		#word brings up the dictionary value for the word, will be a number
		count_feature_vector[current_doc][word] = occurrences	#can be 0
		presence_feature_vector[current_doc][word] = 1 #word is present
		#print current_doc

#pass in the prescence or count matrix for processing
#value of 0 for both prescence and count at this point
def process_final_matrix(matrix):
	for row in matrix:
	#row referes to each document aka each document's dictionary
		for word in master_word_list:
			if word not in row:
				#if the word from the master list is not in the dictionary 
				row[word] = 0
	
#*******************

#calculate frequency of a word inside each document from master word list
def process_word_freq():
	#loop through each word list stored in seperate 2d matrix
	for list_id in range(0,num_documents):
		#loop through words in complete word list		
		for word in master_word_list:
			#look at each list, count up how often the word occurres in that list aka frequency in document
			occurrences = document_word_lists[list_id].count(word)
			count_feature_vector[list_id][word] = occurrences #set value of word to how many times it occurres in the document, this is a dictionary data structure

		document_word_lists[list_id] = [] #delete list after use to save memory

#creates vector with frequency of each word in the document where it was found
def process_freq():
	for list_id in range(0,num_documents):
		print "Processing List %d" % (list_id)
		for word in document_word_lists[list_id]:
			occurrences = document_word_lists[list_id].count(word)
			count_feature_vector[list_id][word] = occurrences
			presence_feature_vector[list_id][word] = 1
		#TEST
		#print "Processing Master Word List for List %d" % (list_id)
		#for word_m in master_word_list:
		#	count_feature_vector[list_id][word_m] = 0
		#	presence_feature_vector[list_id][word_m] = 0
		
				


#can now also calculate tf-ifd
#http://stevenloria.com/finding-important-words-in-a-document-using-tf-idf/ is a good reference

#calculate term frequency for each document
def term_freq(word, list_of_words):
	occurrences = list_of_words.count(word) #how many times it appears in the list
	tot_words = len(list_of_words) #total number of words in this doc
	return float(occurrences)/float(tot_words)

#check how many documents contain the word
def num_docs_containing(word):
	count = 0	
	for doc_list in document_word_lists:
		if word in doc_list:
			count = count + 1
	return count

#calculate idf using wikipedia formula
#lower idf = more common (inverse)
def idf(word):
	log_len = math.log(len(document_word_lists))
	denom = 1 + num_docs_containing(word)
	return log_len / denom

#calculate tfidf
#pass in the word and list of words for that document
def tf_idf(word, list_of_words):
	return term_freq(word, list_of_words) * idf(word)

#find average tf_idf from all documents
def avg_tf_idf():
	total = 0
	total_num_words = 0
	for doc_list in tf_idf_feature_vector:
		for word in doc_list:
			total_num_words = total_num_words + 1 
			total = total + doc_list[word]
	return float(total)/float(total_num_words)

#find average tf from dictionary tf object

def avg_tf(dict_of_words):
	total = 0
	for word in dict_of_words:
		total = total + dict_of_words[word]
	if len(dict_of_words) != 0:
		avg = float(total)/len(dict_of_words)
	else:
		avg = 0
	return avg

def print_preprocessed_dat_files(num_in_dat_file):
	count_feature_dat = count_feature_vector[0:num_in_dat_file]
	pres_feature_dat = presence_feature_vector[0:num_in_dat_file]

	dat_file_freq = open("freq_dat","wb")
	pickle.dump(count_feature_dat,dat_file_freq)
	dat_file_freq.close()

	dat_file_pres = open("pres_dat","wb")
	pickle.dump(pres_feature_dat,dat_file_pres)
	dat_file_pres.close()


#want to loop through each segment
#use URL as there is no need to save data files locally

#initialize counters
#will go from 0 - num_docs-1
document_number = 0
topics_counter = 0
places_counter = 0

#loop through url index 0 - 21
for current_page in range(0,num_pages):
	print "Scanning Page %d of 22" % (current_page + 1)
	number = '000'
	if current_page < 10:
		number = '00' + str(current_page)
	else:
		number = '0' + str(current_page)
	url= 'http://web.cse.ohio-state.edu/~srini/674/public/reuters/reut2-'+number+'.sgm'
	page = urllib2.urlopen(url)

	#make soup from entire page
	soup = BeautifulSoup(page.read(),'lxml')

	#find text tag aka separate by document, each document has one text tag
	contents = soup.findAll('text')
	topics = soup.findAll('topics')
	places = soup.findAll('places')
	#can grab other tags here if needed

	#print len(contents)
	for body in contents:	    
		#grab list of words from body tag
		words = split_words(body.text) #split words from each individual body
		words = make_lower_case(words) #make all words lowercase
	   	words = filter_out_words(words) #filter out stop words
		words = remove_numbers(words) #remove numbers from word list	
		words = get_word_stems(words) #create list of words by stem only
	    	
		#filter words by term frequency
		tf_dictionary = tf_dict(words) #returns dictionary list with value as tf
		#if a word has a term frequency is in above the average + 1std, keep it
		words = filter_by_tf(words, tf_dictionary)
		tf_dictionary.clear() #delete dictionary to save memory, no longer needed

		master_word_list.extend(words) #add these words to the master word
	
		#store word list for each document in its own list
		document_word_lists[document_number] = words #set words as the dictionary for this row (document)

		document_number = document_number + 1 #doc has been processed, increment

	#process class labels as TOPICS and PLACES tags
	#each doc has topics and places tags even if empty, so counters match up
	#keep all upper case to differentiate
	#no filtering needed here
	#topics and places are seprated by <D> tag
	for topic in topics:
		#print topics_counter
		labels = clean_up_d_tag(topic, "<topics>","</topics>")#just scraped part wth tags#split_words(topic.text)
		labels = make_upper_case(labels)
		word_list = document_word_lists[topics_counter]
		word_list.extend(labels) #add to word list for this document
		master_word_list.extend(labels) #add to master word list
		topics_counter = topics_counter + 1 

	for place in places:
		labels = clean_up_d_tag(place, "<places>","</places>")#split_words(place.text)
		labels = make_upper_case(labels)
		word_list = document_word_lists[places_counter]
		word_list.extend(labels) #add to word list for this document
		master_word_list.extend(labels) #add to master word list
		places_counter = places_counter + 1 

		


#all individual document processing is done
#go through each document and process its dictionary
#process_final_matrix(count_feature_vector)
#process_final_matrix(presence_feature_vector)

#convert to set and back to list to remove duplicates which GREATLY reduces number of words in the list
#do so after finding all words, saves A LOT of time than doing it every page iteration
#takes 3 minutes 30 seconds to get 77k words
master_word_list = list(set(master_word_list)) 

#have all the words from all the documents
#calculate tf-idf for each word in each document 
#store in seperate vector 
#for list_id in range(0,num_documents):
#	print list_id
#	word_list = document_word_lists[list_id]
#	for word in word_list:
		#calculate tf-idf value for each word in a document from the word list
		#store this value into a tf-idf feature vector
#		score = tf_idf(word,word_list)
#		tf_idf_feature_vector[list_id][word] = score
	
#grab all words that are above the average tf_idf score from all documents
#average_score = avg_tf_idf() #from all words in all docs
#for doc_list in tf_idf_feature_vector:
#	for word in doc_list:
#		if doc_list[word] >= average_score:
			#if the tf-idf score of this word is above the average across all words
			#add to relevant words list
#			relevant_word_list.extend(word)

print "Number of releveant words sifted out %d" % (len(master_word_list))

#after have entire word list from all docs, go in and process each seperate doc list			
#process_word_freq()
print "Processing feature vectors"
process_freq()


#print information to file
all_word_file = open("all_words.txt","wb")
for word in sorted(master_word_list):
	all_word_file.write(word)
	all_word_file.write("\n")
all_word_file.close()



#dat_file = open("freq_dat","r")
#count_feature_vector = pickle.load(dat_file)
#dat_file.close()


print "Writing feature vectors to files"

count_vector_file = open("count_fv.txt","wb")
for index in range(0,num_documents):
	word_list = count_feature_vector[index]
	count_vector_file.write("DOCUMENT ")
	count_vector_file.write(str(index))
	count_vector_file.write(": \n\n")#extra space
	for word, count in word_list.iteritems():
		count_vector_file.write(word)
		count_vector_file.write(" : ")
		count_vector_file.write(str(count)) #should be number, only one attribute per word
		count_vector_file.write("\n")
	count_vector_file.write("\n")
count_vector_file.close()

presence_vector_file = open("pres_fv.txt","wb")
for index in range(0,num_documents):
	word_list = presence_feature_vector[index]
	presence_vector_file.write("DOCUMENT ")
	presence_vector_file.write(str(index))
	presence_vector_file.write(": \n\n")#extra space
	for word, count in word_list.iteritems():
		presence_vector_file.write(word)
		presence_vector_file.write(" : ")
		presence_vector_file.write(str(count)) #should be number, only one attribute per word
		presence_vector_file.write("\n")
	presence_vector_file.write("\n")
presence_vector_file.close()

num_in_dat = input("Enter number of documents to store in data file\n")
print "Creating dat files"
print_preprocessed_dat_files(num_in_dat)
print "Complete"





#print count_feature_vector
#print presence_feature_vector[0]
#print len(master_word_list)
#print tf_idf_feature_vector[0]


		#as of now, getting all words from each document (reuters tags)
		#all words between text tags
		#when counting words and master words, same number
		#when filtering out duplicates, less
		#but numbers are correct before removing duplicates
		#use this in report
		



