# Open MPI version of DFT
## Introduction
This is an implementation of DFT using Open MPI. The program is written in C and compiled using mpicc. The program is tested on Ubuntu 20.04.2 LTS with Open MPI 4.0.5.
This program performs a parallel Discrete Fourier Transform (DFT) of a matrix using the Message Passing Interface (MPI) library.
The matrix is read in by the master process, which then broadcasts it to all the other processes. Each slave process calculates the DFT for a certain range of rows in the matrix and stores the results in a local array. These local arrays are then gathered by the master process into a single array, which is used to construct the frequency domain matrix. The resulting frequency domain matrix is then printed out by the master process.
The program parallelizes the computation of the DFT by dividing the matrix rows among the different processes. This reduces the computation time, as each process only has to calculate the DFT for a portion of the matrix, rather than the entire matrix.
The program also uses the MPI library to facilitate the communication between processes, allowing for efficient exchange of data between processes.
In summary, the program parallelizes the computation of the DFT by dividing the matrix rows among the different processes and uses the MPI library to facilitate the communication between processes.
## How to run program - Open MPI version
1. Clone this repository
2. Go to `src/open-mpi` directory
3. Run `mpicc open-mpi.c -o mpi` to compile the source code
4. Run `mpirun -np <number of processes> ./dft <input file> <output file>` to run the program
5. Another way to run the program is by using `mpirun --hostfile hostfile ./mpi < 128.txt > out.txt`