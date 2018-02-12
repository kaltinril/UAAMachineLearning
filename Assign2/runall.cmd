echo "Starting 0.1 with 100 iterations, 500 batch"
start python preceptron.py 0.1 100 500 ^> .\runs\0.1_100_500.log

echo "Starting 0.01 with 100 iterations, 500 batch"
start python preceptron.py 0.01 100 500 ^> .\runs\0.01_100_500.log

echo "Starting 0.001 with 100 iterations, 500 batch"
start python preceptron.py 0.001 100 500 ^> .\runs\0.001_100_500.log

echo "Starting 1.5 with 100 iterations, 500 batch"
python preceptron.py 1.5 100 500 > .\runs\1.5_100_500.log

echo "Starting 0.1 with 1000 iterations"
start python preceptron.py 0.1 1000 20 ^> .\runs\0.1_1000_20.log
start python preceptron.py 0.1 1000 50 ^> .\runs\0.1_1000_50.log
python preceptron.py 0.1 1000 100 > .\runs\0.1_1000_100.log
start python preceptron.py 0.1 1000 500 ^> .\runs\0.1_1000_500.log
start python preceptron.py 0.1 1000 1000 ^> .\runs\0.1_1000_1000.log

echo "Starting 0.01 with 1000 iterations"
python preceptron.py 0.01 1000 20  > .\runs\0.01_1000_20.log
start python preceptron.py 0.01 1000 50  ^> .\runs\0.01_1000_50.log
python preceptron.py 0.01 1000 100  > .\runs\0.01_1000_100.log
start python preceptron.py 0.01 1000 500  ^> .\runs\0.01_1000_500.log
start python preceptron.py 0.01 1000 1000  ^> .\runs\0.01_1000_1000.log

echo "Starting 0.001 with 1000 iterations"
python preceptron.py 0.001 1000 20  > .\runs\0.001_1000_20.log
start python preceptron.py 0.001 1000 50  ^> .\runs\0.001_1000_50.log
python preceptron.py 0.001 1000 100  > .\runs\0.001_1000_100.log
start python preceptron.py 0.001 1000 500  ^> .\runs\0.001_1000_500.log
start python preceptron.py 0.001 1000 1000  ^> .\runs\0.001_1000_1000.log

echo "Starting 1.5 with 1000 iterations"
python preceptron.py 1.5 1000 20  > .\runs\1.5_1000_20.log
start python preceptron.py 1.5 1000 50  ^> .\runs\1.5_1000_50.log
python preceptron.py 1.5 1000 100  > .\runs\1.5_1000_100.log
start python preceptron.py 1.5 1000 500  ^> .\runs\1.5_1000_500.log
start python preceptron.py 1.5 1000 1000  ^> .\runs\1.5_1000_1000.log
