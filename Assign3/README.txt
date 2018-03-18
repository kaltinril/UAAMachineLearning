Jeremy Swartwood
UAA Machine Learning
Spring 2018
Assignment 3
 - 1. [csv_to_svm_converter] Use SVM-Light to classify aurora Borealis images as having or not having aurora. 
 - 2. [pca_reduction] Use PCA (Sklearn) to reduce the feature space from 768 down to N features
 - - Where N encapsulates 95% of the variance of the data.

csv_to_svm_converter.py
    Take a CSV formatted list of features, where the first row is the prediction and the last row is the filename,
    and convert to the SVM-Light required format.
     
    Usage:
            python csv_to_svm_converter.py [-p prefix] [-i filename]
            
    Example:
            python csv_to_svm_converter.py -p ah_pca -i aurora.csv
    
    Options:
            -p, --prefix
                Default: ah
                The value to be perpended to the _training.csv and _validate.csv output files.
            
            -i, --infile
                Default: aurora_histogram.csv
                The source filename used as input to this tool
            
            -h, --help
                Print the usage and options about this file.
                
    input file:
            <prediction>,<value1>,<value2,...<valuen>,<filename>
            1,4,767,....,2323,c:\path\to\filename1.png
            0,6,823,....,3623,c:\path\to\filename2.png

    output files**:
            <prediction> <feature>:<value> <feature>:<value> ... <feature>:<value> # <filename>
            Where Feature is an incrementing number for each feature
            Value is the value pulled from the input CSV file in order

            1 1:4 2:767 ... 768:2323 # c:\path\to\filename1.png
           -1 1:5 2:823 ... 768:4623 # c:\path\to\filename2.png

            (**Two files are produced, one for Training, and one for validation)
 
pca_reduction.py
    Take an input file of features, predictions, and associated notes and collapse the features.
    Each row should be <prediciton "Y" value>, <one or more features>, <any notes about this row like filename>

    Usage:
            python pca_reduction.py -i input_filename -o output_filename

    Example:
            python pca_reduction.py -i aurora.csv -o aurora_pca.csv
            
    Usage as Import:
            # To use this in another python library:
            import cv2
            import pca_reduction

            pca_reduction.run_pca("input_filename.csv", "output_filename.svm")
            
    Options:
            -o, --outfile
                Default: ./aurora_hist_pca.csv
                The reduced dimension (feature) file.
            
            -i, --infile
                Default: ./aurora_histogram.csv
                The source filename used as input to this tool
            
            -h, --help
                Print the usage and options about this file.
                
            -d, --debug
                Enable debug printing mode

    input file:
            <prediction>,<value1>,<value2,...<valuen>,<filename or notes>
            1,4,767,....,2323,c:\path\to\filename1.png
            0,6,823,....,3623,c:\path\to\filename2.png

    output file:
            Output file is the same format as the input file, 
            except with potentially less columns (Features).