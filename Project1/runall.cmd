set nodes=%1


echo "Starting 0.5 with 500 epochs"
start python letter_ocr.py 0.5 500 20 %nodes%
start python letter_ocr.py 0.5 500 50 %nodes%
start python letter_ocr.py 0.5 500 100 %nodes%
python letter_ocr.py 0.5 500 500 %nodes%
start python letter_ocr.py 0.5 500 1000 %nodes% 

echo "Starting 0.1 with 500 epochs"
start python letter_ocr.py 0.1 500 20 %nodes%
start python letter_ocr.py 0.1 500 50 %nodes%
start python letter_ocr.py 0.1 500 100 %nodes%
python letter_ocr.py 0.1 500 500 %nodes%
start python letter_ocr.py 0.1 500 1000 %nodes%

echo "Starting 0.01 with 500 epochs"
start python letter_ocr.py 0.01 500 20 %nodes%
start python letter_ocr.py 0.01 500 50 %nodes%
start python letter_ocr.py 0.01 500 100 %nodes%
python letter_ocr.py 0.01 500 500 %nodes%
start python letter_ocr.py 0.01 500 1000 %nodes% 

echo "Starting 0.001 with 500 epochs"
start python letter_ocr.py 0.001 500 20 %nodes%
start python letter_ocr.py 0.001 500 50 %nodes%
start python letter_ocr.py 0.001 500 100 %nodes%
python letter_ocr.py 0.001 500 500 %nodes%
start python letter_ocr.py 0.001 500 1000 %nodes%