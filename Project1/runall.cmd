
echo "Starting 0.5 with 500 epochs"
start python letter_ocr.py 0.5 500 20 100
start python letter_ocr.py 0.5 500 50 100
start python letter_ocr.py 0.5 500 100 100
python letter_ocr.py 0.5 500 500 100 > .\runs\0.5_500_500.log
start python letter_ocr.py 0.5 500 1000 100 

echo "Starting 0.1 with 500 epochs"
start python letter_ocr.py 0.1 500 20 100
start python letter_ocr.py 0.1 500 50 100
start python letter_ocr.py 0.1 500 100 100
python letter_ocr.py 0.1 500 500 100  > .\runs\0.1_500_500.log
start python letter_ocr.py 0.1 500 1000 100

echo "Starting 0.01 with 500 epochs"
start python letter_ocr.py 0.01 500 20 100
start python letter_ocr.py 0.01 500 50 100
start python letter_ocr.py 0.01 500 100 100
python letter_ocr.py 0.01 500 500 100 > .\runs\0.01_500_500.log
start python letter_ocr.py 0.01 500 1000 100 

echo "Starting 0.001 with 500 epochs"
start python letter_ocr.py 0.001 500 20 100
start python letter_ocr.py 0.001 500 50 100
start python letter_ocr.py 0.001 500 100 100
python letter_ocr.py 0.001 500 500 100 > .\runs\0.001_500_500.log
start python letter_ocr.py 0.001 500 1000 100