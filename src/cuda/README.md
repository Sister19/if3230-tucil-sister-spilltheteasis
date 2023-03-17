# CUDA version of DFT and FFT
## Introduction
This program is an implementation of Discrete Fourier Transform (DFT) and Fast Fourier Transform (FFT) using NVIDIA's CUDA parallel computing platform. CUDA is a parallel computing platform that allows for the acceleration of complex computations by utilizing the power of a computer's graphics processing unit (GPU). This program performs a 2-dimensional discrete Fourier transform (DFT) and Fast Fourier Transform (FFT) on a square matrix using CUDA to parallelize the calculation. It reads a square matrix from the input and outputs its frequency domain representation to the console.

## How to run program - Open MP version
1. Clone this repository.
2. Go to `src/cuda` directory.
3. Run `nvcc cuda-dft.cu -o cuda-dft` to compile the dft source code or run `gcc nvcc cuda-fft.cu -o cuda-fft` to compile the fft source code.
4. Create empty file output.txt
5. Run `time ./cuda-dft < [input file].txt > output.txt` or `time ./cuda-fft < [input file].txt > output.txt` to run the program