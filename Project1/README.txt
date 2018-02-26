Jeremy Swartwood
Machine Learning Spring 2018 UAA
Project1 (Letter OCR)

Expectations:
	Input file has already had the A-Z replaced with 1-26  (Not 0-25)
	No parameters are required, uses the default values in () below if no parameters are passed.

Output:
	Prints the configuration used to the screen.
	Prints the time details of the run.
	Prints the Final Training and Validation accuracy rates.
	Creates 3 files on completion.
		1. Accuracy Rate vs Epoch
		2. Confusion Matrix (Actual vs Predicted)
		3. Error Rate vs Epoch

Usage:
	python letter_ocr.py [learn] [epochs] [batch_size] [hidden_nodes] [find_optimal]

	learn = (0.1) Learning rate for weight adjustment
	epochs = (500) # of complete runs through the Training set
	batch_size = (20) # of rows to batch through the matrix process at a time
	hidden_nodes = (100) # of nodes in the hidden layer
	find_optimal = (0) 0 = False, 1 = True.  Run from nodes 1-100, testing 3 times on each node, 
		       getting the average accuracy, and printing the best node count

Example Usage:
	- Run with Defaults
	python letter_ocr.py

	- Run with Defaults, but change the learning rate
	python letter_ocr.py 0.001

	- Run with Defaults, but change the learn, and epochs
	python letter_ocr.py 0.01 1000

	- Run with all parameters
	python letter_ocr.py 0.1 500 100 17 0
