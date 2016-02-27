Tim Taylor
CSE 5243
Lab 2

This README will tell you how to run dbscan from our files as well as what results to expect, and how to vary the epsilon, minPoints, and
input file parameters.

The basic command format for how to run dbscan is as follows:

python dbscan.py <epsilon> <minPoints> <inputFile name for feature vectors file> <1 for Manhattan or 2 for euclidian>

so, for example, if you wanted to run dbscan with epsilon of 10, minPoints of 5, using manhattan for distance, and using the input file
"inputFile" you would use the command:

python dbscan.py 10 5 inputFile 1

For the sake of saving time, input files of feature vectors have already been generated at several sizes (5,000 docs, 10,000 docs, 15,000 
docs, and the full 21578 docs) for immediate use. All of these files are stored in the datafiles folder in this directory. 
So a proper usage of our input files will look like:

python dbscan.py 10 5 datafiles/freq_dat_5000 1
  (for smallest load of 5000 docs)
python dbscan.py 10 5 datafiles/freq_dat_10000 1
  (for next smallest load of 10,000 docs)
.
.
.
and so on
  
These input files are saved and read through the Pickle library of python, so do not try to use your own input files.

