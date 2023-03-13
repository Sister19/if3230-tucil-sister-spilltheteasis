// TUBES SISTER 13520002 CUDA
// cuda-dft.cu
 
// how to run
// > nvcc cuda-dft.cu -o cuda-dft
// > ./cuda-dft
 
// how to measure time
// ex testcase.txt already created, create empty output.txt
// > time ./cuda-dft < 128.txt > output.txt

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

__global__ void dft(double *d_mat, cuDoubleComplex *d_freq, int size) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y * blockDim.y + threadIdx.y;

    if (k < size && l < size) {
        cuDoubleComplex sizeSquare      = make_cuDoubleComplex(size*size, 0.0);
        cuDoubleComplex element         = make_cuDoubleComplex(0.0, 0.0);  
        cuDoubleComplex var_exp         = make_cuDoubleComplex(0.0, -2.0);

        for (int m = 0; m < size; m++) {
            for (int n = 0; n < size; n++) {
                cuDoubleComplex arg             = make_cuDoubleComplex(((k*m / (double) size) + (l*n / (double) size)), 0.0);
                cuDoubleComplex exponent        = _cuCexp(cuCmul(cuCmul(var_exp, CU_MPI), arg));
                cuDoubleComplex el              = make_cuDoubleComplex(d_mat[m*size+n],0.0);
                element                         = cuCadd(element, cuCmul(el, exponent));
            }
        } 

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

    cudaMalloc((void **) &d_mat, source.size * source.size * sizeof(double));
    cudaMalloc((void **) &d_freq, source.size * source.size * sizeof(cuDoubleComplex));

    cudaMemcpy(d_mat, source.mat, source.size * source.size * sizeof(double), cudaMemcpyHostToDevice);

    dim3 block(source.size/BLOCK_SIZE, source.size/BLOCK_SIZE, 1);    
    dim3 grid(BLOCK_SIZE, BLOCK_SIZE, 1);
    dft<<<block, grid>>>(d_mat, d_freq, source.size);

    cudaMemcpy(freq_domain.mat, d_freq, source.size * source.size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    cudaFree(d_mat);
    cudaFree(d_freq);
    
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex size = make_cuDoubleComplex(source.size, 0.0);
    for (int k = 0; k < source.size; k++) {
        for (int l = 0; l < source.size; l++) {
            // printf("%d %d\n", k, l);
            cuDoubleComplex el = freq_domain.mat[k*freq_domain.size+l];
            printf("(%lf, %lf) ", cuCreal(el), cuCimag(el));
            sum = cuCadd(sum, el);
        }
        printf("\n");
    }
    
    sum = cuCdiv(sum, size);
    printf("Average : (%lf, %lf)\n", cuCreal(sum), cuCimag(sum));

    cudaDeviceSynchronize();
    return 0;
}
