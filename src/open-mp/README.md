# Open MP version of DFT
## Introduction
This program is an implementation of Discrete Fourier Transform (DFT) and Fast Fourier Transform (FFT) using OpenMP, a library for shared memory multiprocessing programming in C. This program performs a 2-dimensional discrete Fourier transform (DFT) and Fast Fourier Transform (FFT) on a square matrix using OpenMP to parallelize the calculation. It reads a square matrix from the input and outputs its frequency domain representation to the console.
The program uses the dft function to calculate the frequency domain representation for each element of the output matrix. The dft function uses a double loop to iterate over the input matrix and calculate the DFT using the formula for discrete Fourier transform.
The loops in the DFT calculation are parallelized using OpenMP directives. The number of threads used by the program can be set using the omp_set_num_threads() function.
The following OpenMP directives are used in the program:
- #pragma omp parallel for: parallelizes the loop
- #pragma omp parallel for private(k, l) shared(source, freq_domain): declares the loop variables k and l as private and the source and freq_domain matrices as shared between threads.
- #pragma omp parallel for reduction(+:element): declares the variable element as reduction, which allows each thread to have a private copy of the variable and accumulate its value in the reduction operation.

## How to run program - Open MP version
1. Clone this repository
2. Go to `src/open-mp` directory
3. Run `gcc open-mp-dft.c --openmp -o mp` to compile the source code that using fft or run `gcc open-mp-dft.c --openmp -o mp` to compile the source code that using dft
4. Run `./mp < 128.txt > out.txt` to run the program
