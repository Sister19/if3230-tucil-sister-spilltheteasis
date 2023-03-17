// TUBES SISTER 13520002 CUDA
// cuda-fft.cu
 
// how to run
// > nvcc cuda-fft.cu -o cuda-fft
// > ./cuda-fft
 
// how to measure time
// ex testcase.txt already created, create empty output.txt
// > time ./cuda-fft < 128.txt > output.txt

#include <cuComplex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_N 512
#define BLOCK_SIZE 16
#define CU_MPI make_cuDoubleComplex(M_PI, 0.0)

struct Matrix {
    int    size;
    double mat[MAX_N*MAX_N];
};

struct FreqMatrix {
    int    size;
    cuDoubleComplex mat[MAX_N*MAX_N];
};

void readMatrix(struct Matrix *m) {
    scanf("%d", &(m->size));
    for (int i = 0; i < m->size; i++)
        for (int j = 0; j < m->size; j++)
            scanf("%lf", &(m->mat[i*m->size+j]));
}

__host__ __device__ cuDoubleComplex _cuCexp (cuDoubleComplex arg)
{
   cuDoubleComplex res;
   double s, c;
   double e = exp(arg.x);
   sincos(arg.y, &s, &c);
   res.x = c * e;
   res.y = s * e;
   return res;
}

__global__ void fft(double *d_mat, cuDoubleComplex *d_freq, int size) {
    // get index of thread
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y * blockDim.y + threadIdx.y;

    if (k < size && l < size) {
        // Rumus 2D FFT
        // F[k,l] = 1/MN * { 
        // sum(sum(f[m,n] * e^((-2*pi*i) * (k*m/M + l*n/N))) +                              for calculate even row and even column
        // sum(sum(f[m,n] * e^((-2*pi*i) * (k*m/M + l*n/N)) * e^((-2*pi*i) * (l/N))) +      for calculate even row and odd column
        // sum(sum(f[m,n] * e^((-2*pi*i) * (k*m/M + l*n/N)) * e^((-2*pi*i) * (k/M))) +      for calculate odd row and even column
        // sum(sum(f[m,n] * e^((-2*pi*i) * (k*m/M + l*n/N)) * e^((-2*pi*i) * ((l+k)/M)))    for calculate odd row and odd column
        // }

        // make size to cuDoubleComplex
        cuDoubleComplex sizeSquare      = make_cuDoubleComplex(size*size, 0.0);
        // initialize element in cuDoubleComplex format = 0 + 0i
        cuDoubleComplex element         = make_cuDoubleComplex(0.0, 0.0);  
        // make -2 to cuDoubleComplex
        cuDoubleComplex var_exp         = make_cuDoubleComplex(0.0, -2.0);

        // arg for even row and even column
        // 0
        // arg for even row and odd column
        cuDoubleComplex arg_even_odd    = make_cuDoubleComplex(l / (double) size, 0.0);
        // arg for odd row and even column
        cuDoubleComplex arg_odd_even    = make_cuDoubleComplex(k / (double) size, 0.0);
        // arg for odd row and odd column
        cuDoubleComplex arg_odd_odd     = make_cuDoubleComplex((k + l) / (double) size, 0.0);

        // var for even row and even column
        // 1
        // var for even row and odd column
        cuDoubleComplex var_even_odd    = _cuCexp(cuCmul(cuCmul(var_exp, CU_MPI), arg_even_odd));
        // var for odd row and even column
        cuDoubleComplex var_odd_even    = _cuCexp(cuCmul(cuCmul(var_exp, CU_MPI), arg_odd_even));
        // var for odd row and odd column
        cuDoubleComplex var_odd_odd     = _cuCexp(cuCmul(cuCmul(var_exp, CU_MPI), arg_odd_odd));

        for (int m = 0; m < size/2; m++) {
            for (int n = 0; n < size/2; n++) {
                // calculate e^((-2*pi*i) * (k*m/M + l*n/N))
                cuDoubleComplex arg             = make_cuDoubleComplex(((k*m / (double) size) + (l*n / (double) size)), 0.0);
                cuDoubleComplex exponent        = _cuCexp(cuCmul(cuCmul(var_exp, CU_MPI), arg));
                // make element in matrix to cuDoubleComplex
                cuDoubleComplex el_even_even    = make_cuDoubleComplex(d_mat[(2*m)*size+(2*n)],0.0);
                cuDoubleComplex el_even_odd     = make_cuDoubleComplex(d_mat[(2*m)*size+(2*n+1)],0.0);
                cuDoubleComplex el_odd_even     = make_cuDoubleComplex(d_mat[(2*m+1)*size+(2*n)],0.0);
                cuDoubleComplex el_odd_odd      = make_cuDoubleComplex(d_mat[(2*m+1)*size+(2*n+1)],0.0);
                // add each result to element
                element                         = cuCadd(element, cuCmul(el_even_even, exponent));                          // even row and even column
                element                         = cuCadd(element, cuCmul(el_even_odd, cuCmul(var_even_odd, exponent)));     // even row and odd column
                element                         = cuCadd(element, cuCmul(el_odd_even, cuCmul(var_odd_even, exponent)));     // odd row and even column
                element                         = cuCadd(element, cuCmul(el_odd_odd, cuCmul(var_odd_odd, exponent)));       // odd row and odd column
            }
        } 
        // set element to d_freq
        d_freq[k*size+l] = cuCdiv(element, sizeSquare);   
    }
}


int main(void) {
    struct Matrix     source;
    struct FreqMatrix freq_domain;
    double *d_mat;
    cuDoubleComplex *d_freq;

    readMatrix(&source);
    freq_domain.size = source.size;

    // allocate memory in device
    cudaMalloc((void **) &d_mat, source.size * source.size * sizeof(double));
    cudaMalloc((void **) &d_freq, source.size * source.size * sizeof(cuDoubleComplex));

    // copy data from host to device
    cudaMemcpy(d_mat, source.mat, source.size * source.size * sizeof(double), cudaMemcpyHostToDevice);

    // set block size (16 blocks) and grid size (matrix size/block size)
    dim3 block(source.size/BLOCK_SIZE, source.size/BLOCK_SIZE, 1);    
    dim3 grid(BLOCK_SIZE, BLOCK_SIZE, 1);
    // call kernel
    fft<<<block, grid>>>(d_mat, d_freq, source.size);

    // copy data from device to host
    cudaMemcpy(freq_domain.mat, d_freq, source.size * source.size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    // free memory in device
    cudaFree(d_mat);
    cudaFree(d_freq);

    cudaDeviceSynchronize();
    
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex size = make_cuDoubleComplex(source.size, 0.0);
    for (int k = 0; k < source.size; k++) {
        for (int l = 0; l < source.size; l++) {
            cuDoubleComplex el = freq_domain.mat[k*freq_domain.size+l];
            printf("(%lf, %lf) ", cuCreal(el), cuCimag(el));
            sum = cuCadd(sum, el);
        }
        printf("\n");
    }
    
    sum = cuCdiv(sum, size);
    printf("Average : (%lf, %lf)\n", cuCreal(sum), cuCimag(sum));

    return 0;
}
