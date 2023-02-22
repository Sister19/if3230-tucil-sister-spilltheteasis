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
 
double complex dft(struct Matrix *mat, int k, int l)
{
    double complex element = 0.0;
    for (int m = 0; m < mat->size; m++)
    {
        for (int n = 0; n < mat->size; n++)
        {
            double complex arg = (k * m / (double)mat->size) + (l * n / (double)mat->size);
            double complex exponent = cexp(-2.0I * M_PI * arg);
            element += mat->mat[m][n] * exponent;
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
            local_arr[k % local_size][l] = dft(&source, k, l);
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
