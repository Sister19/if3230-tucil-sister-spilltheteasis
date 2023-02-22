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

double complex dft(struct Matrix *mat, int k, int l) {
    double complex element = 0.0;
    int m, n;
    // Parallelize the loop
    // Set reduction for element
    #pragma omp parallel for reduction(+:element)
    for (m = 0; m < mat->size; m++) {
        for (n = 0; n < mat->size; n++) {
            double complex arg      = (k*m / (double) mat->size) + (l*n / (double) mat->size);
            double complex exponent = cexp(-2.0I * M_PI * arg);
            element += mat->mat[m][n] * exponent;
        }
    }
    return element / (double) (mat->size*mat->size);
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
            freq_domain.mat[k][l] = dft(&source, k, l);  
        }
    }

    printFreqMatrix(&freq_domain);

    return 0;
}