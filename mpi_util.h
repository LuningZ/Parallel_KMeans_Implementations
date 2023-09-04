#ifndef MPI_UTIL_H
#define MPI_UTIL_H

#include<mpi.h>

// Functions from mpi_io.c
float** file_read(char* filename, int* N, int* dims);
float** mpi_read(char *filename, int *N, int *dims, MPI_Comm comm);
int mpi_write(char *filename, float **objects, int K, int local_N, int dims, float **clusters, int *membership, MPI_Comm comm, int verbose);
int elkans_write(char *filename, float **objects, int K, int local_N, int dims, float **clusters, int *membership, MPI_Comm comm, int verbose);

// Functions from mpi_kmeans.c
float calculate_dist(int dims, float *p1, float *p2);
int find_nearest_cluster(int K, int dims, float *object, float **clusters);
int mpi_kmeans(float **objects, int dims, int local_N, int K, float threshold, int *membership, float **clusters, MPI_Comm comm);

// Functions from elkans_kmeans.c
int elkans_kmeans(float **objects, int dims, int local_N, int K, float threshold, int *membership, float **clusters, MPI_Comm comm, int local_dist_cal, float **dist_c1c2, float **lower, float *upper, int verbose);

#endif // MPI_UTIL_H
