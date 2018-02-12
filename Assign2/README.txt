3.1 ColorHistogram.py
This program assumes there is an images directory with sub-directories aurora and none.  It will go through both sub-directories, reading each file, generating a histogram, and outputting the results into aurora_histogram.csv

Usage:
python ColorHistogram.py

Outputs:
Generates a CSV file with all AURORA and NOT AURORA histograms, 1 image per row.  
For details on this file, see writeup in section 2.2.

3.2 preceptron.py
Yes, this is named incorrectly, should be perceptron.py.  I left it as this because all my other code is assuming this name.

This program takes three parameters, and assumes an input file exists in the same directory named “aurora_histogram.csv”.

Usage:
python perceptron.py <learn> <iterations> <batchsize>

•	learn:		The learning rate (Mu)
•	Iterations:	Number of batches to run
•	BatchSize:	Number of random rows to select from the Training or Validation set during Training or Validation respectively.

Outputs:
See perceptron section 2.3.1 in the writeup for file details.

A printout to the screen of the overall (averaged) error for Training and Validation is printed to the screen.
The input parameters are printed.
The time information about runtime is printed.

3.3 runall.cmd
A simple windows batch file to run preceptron.py multiple times and in parallel.

Usage:
runall.cmd
