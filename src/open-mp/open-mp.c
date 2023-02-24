// TUBES SISTER 13520002 OPENMP
// openmp.c
 
// how to run
// > gcc mp.c --openmp -o mp -lm
// > ./mp
 
// how to measure time
// ex testcase.txt already created, create empty output.txt
// > time ./mp < 128.txt > output.txt

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <time.h>

#define MAX_N 512

struct Matrix {
    int    size;
    double mat[MAX_N][MAX_N];
};

struct FreqMatrix {
    int    size;
    double complex mat[MAX_N][MAX_N];
};

void readMatrix(struct Matrix *m) {
    scanf("%d", &(m->size));
    for (int i = 0; i < m->size; i++)
        for (int j = 0; j < m->size; j++)
            scanf("%lf", &(m->mat[i][j]));
}

double complex fft(struct Matrix *mat, int k, int l) {

    // Rumus 2D FFT
    // F[k,l] = 1/MN * { 
    // sum(sum(f[m,n] * e^((-2*pi*i) * (k*m/M + l*n/N))) +                              for calculate even row and even column
    // sum(sum(f[m,n] * e^((-2*pi*i) * (k*m/M + l*n/N)) * e^((-2*pi*i) * (l/N))) +      for calculate even row and odd column
    // sum(sum(f[m,n] * e^((-2*pi*i) * (k*m/M + l*n/N)) * e^((-2*pi*i) * (k/M))) +      for calculate odd row and even column
    // sum(sum(f[m,n] * e^((-2*pi*i) * (k*m/M + l*n/N)) * e^((-2*pi*i) * ((l+k)/M)))    for calculate odd row and odd column
    // }
    // arg for even row and even column
    // 0
    // arg for even row and odd column
    double complex arg_even_odd = (l / (double)mat->size);
    // arg for odd row and even column
    double complex arg_odd_even = (k / (double)mat->size);
    // arg for odd row and odd column
    double complex arg_odd_odd  = ((k + l) / (double)mat->size);

    // var for even row and even column
    // 1
    // var for even row and odd column
    double complex var_even_odd = cexp(-2.0I * M_PI * arg_even_odd);
    // var for odd row and even column
    double complex var_odd_even = cexp(-2.0I * M_PI * arg_odd_even);
    // var for odd row and odd column
    double complex var_odd_odd  = cexp(-2.0I * M_PI * arg_odd_odd);

    double complex element = 0.0;

    int m, n;
    // Parallelize the loop
    // Set reduction for element
    #pragma omp parallel for reduction(+:element)
    for (int m = 0; m < mat->size/2; m++)
    {
        for (int n = 0; n < mat->size/2; n++)
        {
            double complex arg = (k * m / ((double)mat->size/2)) + (l * n / ((double)mat->size/2));
            double complex exponent = cexp(-2.0I * M_PI * arg);
            element +=  (mat->mat[2*m][2*n] * exponent) +                       // even row and even column
                        (mat->mat[2*m][2*n+1] * exponent * var_even_odd) +      // even row and odd column
                        (mat->mat[2*m+1][2*n] * exponent * var_odd_even) +      // odd row and even column
                        (mat->mat[2*m+1][2*n+1] * exponent * var_odd_odd);      // odd row and odd column
        }

    }

    return element / (double)(mat->size * mat->size);
}

void printFreqMatrix(struct FreqMatrix *m)
{
    double complex sum = 0.0;
    for (int i = 0; i < m->size; i++)
    {
        for (int j = 0; j < m->size; j++)
        {
            double complex el = m->mat[i][j];
            printf("(%lf, %lf) ", creal(el), cimag(el));
            sum += el;
        }
        printf("\n");
    }
    sum /= m->size;
    printf("Average : (%lf, %lf)\n", creal(sum), cimag(sum));
    
}

int main(void) {
    struct Matrix     source;
    struct FreqMatrix freq_domain;

    readMatrix(&source);
    freq_domain.size = source.size;
    
    // Set the number of threads
    omp_set_num_threads(4);

    // Parallelize the loop 
    // Set private variables and shared variables
    int k, l;
    #pragma omp parallel for private(k, l) shared(source, freq_domain)
    for (k = 0; k < source.size; k++) {
        for (l = 0; l < source.size; l++) {
            freq_domain.mat[k][l] = fft(&source, k, l);  
        }
    }

    printFreqMatrix(&freq_domain);

    return 0;
}