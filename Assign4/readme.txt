Files:
 1. join_file_data.py - Joins the S##.txt files into a single numpy array and saves it
 2. pre_processing.py - Runs FFT, PSD, and PCA on the numpy array file and saves as pca.npy
 3. clustering.py - opens pca.npy and runs clustering



1. join_file_data.py
Joins the S##.txt files into a single numpy array and saves it
	usage: python join_file_data.py
	assumptions:	data structure is ./ClusteringData/data/a##/p#/s##.txt
	input: 125 rows of 45 columns in each S##.txt file
	output:	data_combined.npy

2. pre_processing.py
Runs FFT, PSD, and PCA on the numpy array file and saves as pca.npy
	usage: python pre_processing.py
	assumptions:	You are using 64 bit python, otherwise, you need to run FFT and PSD seperate from PCA.
	input:	data_combined.npy
	output:	pca.npy

3. clustering.py
opens pca.npy and runs clustering
	usage: python clustering.py
	input: pca.npy
	output:	screen printed mean entropy, timing, coherence, and separation.

	Detailed Usage: 
		place a (numpy array) file called pca.npy in the same directory as clustering.py
		edit clustering.py to for the number of clusters you want to use.
		python clustering.py


