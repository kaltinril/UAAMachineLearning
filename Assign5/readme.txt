Uses Naive Bayes to classify emails as SPAM or nonSPAM based on data from Dr Cenek's output from Spam Assasin

Usage:
  cd /script/location/
  python naive_bayes.py

Assumptions:
  Requires file SpamInstances.txt to be in the same directory as the script.  Execute the script from it's directory.
  64-bit python is installed, otherwise errors occur on the large data file.

Output:
  Two plotted charts are displayed
    1. Accuracy vs iterations
    2. ROC graph