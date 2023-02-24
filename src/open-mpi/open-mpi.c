#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
 
#define MAX_N 512
 
struct Matrix
{
    int size;
    double mat[MAX_N][MAX_N];
};
 
struct FreqMatrix
{
    int size;
    double complex mat[MAX_N][MAX_N];
};
 
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
 
void readMatrix(struct Matrix *m)
{
    scanf("%d", &(m->size));
    for (int i = 0; i < m->size; i++)
        for (int j = 0; j < m->size; j++)
            scanf("%lf", &(m->mat[i][j]));
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
 
void copyMatrix(double complex *m, struct FreqMatrix *n)
{
    for (int i = 0; i < n->size; i++)
    {
        for (int j = 0; j < n->size; j++)
        {
            n->mat[i][j] = m[i * n->size + j];
        }
    }
}
 
int main(int argc, char **argv)
{
    struct Matrix source;
    struct FreqMatrix freq_domain;
    int world_rank, world_size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
 
    if (world_rank == 0)
    {
        readMatrix(&source);
        freq_domain.size = source.size;
    }
    
    MPI_Bcast(&source, sizeof(struct Matrix), MPI_BYTE, 0, MPI_COMM_WORLD);
 
    /*
     * SLAVE FUNCTION
     */
 
    // array 2d buffer, hitung DFT per baris dengan ukuran local_size
    // ex: world size 8 with 4 process, local size is 2
    int local_size = source.size / world_size;
    double complex local_arr[local_size][source.size];
    int lower_bound = local_size * world_rank;
    int upper_bound = local_size * (world_rank + 1);
 
    // asign local array, 0-2, 2-4, 4-6, 6-8
    for (int k = lower_bound; k < upper_bound; k++)
    {
        for (int l = 0; l < source.size; l++)
        {
            local_arr[k % local_size][l] = fft(&source, k, l);
        }
    }
 
    double complex *all_arr = NULL;
 
    if (world_rank == 0) {
        all_arr = (double complex *) malloc(sizeof(double complex) * source.size * source.size);
    }
    
    // gather the array from local_arr in slave process to mpi gather to all_arr in master process
    MPI_Gather(local_arr, local_size * source.size, MPI_C_DOUBLE_COMPLEX, all_arr, local_size * source.size, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
 
    MPI_Finalize();
 
    if (world_rank == 0)
    {
        freq_domain.size = source.size;
        copyMatrix(all_arr, &freq_domain);
        printFreqMatrix(&freq_domain);
        free(all_arr);
    }
 
    return 0;
}
