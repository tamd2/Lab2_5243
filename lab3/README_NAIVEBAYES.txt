Tim Taylor
CSE 5243
Lab 3

How to run the Naive Bayes python script:

    $ pyton naiveBayesClassifier.py <input file name> <percent of docs to train on> [optional: <max number of documents to train on>]

so for example, here is one valid way to run the script:

    $ python naiveBayesClassifier.py datafiles/freq_dat_21578 80

The above would run the classifier on all 21578 documents and will
train on 80% of those, and test on the remaining 20%

This would be incredibly slow however, so a better idea is to do this
script for a smaller number of documents:

    $ python naiveBayesClassifier.py datafiles/freq_dat_21578 80 1000

The above would run the classifier on 1000 documents, training on 
80% of them and testing on 20% of them.

    $ python naiveBayesClassifier.py datafiles/freq_dat_21578 50 1000

The above will run the classifier on 1000 documents, training on 50%
of them and testing on the remaining 50%

For the input files, please only use the input files provided in the
datafiles folder in this submission.

The program will output at the end the following information:

Overall accuracy
Number of true positives (correct predictions of 1)
Number of true negatives
Number of false positives
Number of false negatives
Time taken to build the model (in seconds)
Average time taken to classify an unknown feature vector (in seconds)

