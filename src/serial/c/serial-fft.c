// TUBES SISTER 13520002 OPENMP
// serial.c
 
// how to run
// > gcc serial.c -o serial -lm
// > ./serial
 
// how to measure time
// ex testcase.txt already created, create empty output.txt
// > time ./serial < 128.txt > output.txt

#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
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

double complex fft(struct Matrix *mat, int k, int l) 
{
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
    for (int m = 0; m < mat->size/2; m++)
    {
        for (int n = 0; n < mat->size/2; n++)
        {
            double complex arg = (k * m / ((double)mat->size/2)) + (l * n / ((double)mat->size/2));
            double complex exponent = cexp(-2.0I * M_PI * arg);
            element +=  (mat->mat[2*m][2*n] * exponent) +                       // calculate even row and even column
                        (mat->mat[2*m][2*n+1] * exponent * var_even_odd) +      // calculate even row and odd column 
                        (mat->mat[2*m+1][2*n] * exponent * var_odd_even) +      // calculate odd row and even column
                        (mat->mat[2*m+1][2*n+1] * exponent * var_odd_odd);      // calculate odd row and odd column
        }

    }

    return element / (double)(mat->size * mat->size);
}

int main(void) {
    struct Matrix     source;
    struct FreqMatrix freq_domain;

    readMatrix(&source);
    freq_domain.size = source.size;
    
    for (int k = 0; k < source.size; k++)
        for (int l = 0; l < source.size; l++)
            freq_domain.mat[k][l] = fft(&source, k, l);
    
    double complex sum = 0.0;
    for (int k = 0; k < source.size; k++) {
        for (int l = 0; l < source.size; l++) {
            double complex el = freq_domain.mat[k][l];
            printf("(%lf, %lf) ", creal(el), cimag(el));
            sum += el;
        }
        printf("\n");
    }
    
    sum /= source.size;
    printf("Average : (%lf, %lf)\n", creal(sum), cimag(sum));

    return 0;
}