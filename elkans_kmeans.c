#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include<mpi.h>
#include "mpi_util.h"

#define max_it 1000

float **allocate_2D_float_array(int rows, int cols);
void reset_2D_float_array(float **array, int rows, int cols);
void free_2D_float_array(float **array, int rows);

/**
 * @brief Function to perform Elkan's K-means algorithm
 * 	Return an array of cluster centers of size [K][dims] (MPI Version)
 *
 * @param[in] objects		Pointer to the matrix of data points
 * @param[in] dims		Dimension of data points
 * @param[in] local_N		Number of data points for local rank
 * @param[in] K			Number of clusters
 * @param[in] threshold		objects change membership
 * @param[in] membership	the cluster id for each data object
 * @param[in] clusters		Pointer to the matrix of cluster centroids
 * @param[in] comm		Communicator
 * @param[in] local_dist_cal	Record the num of dist calculations
 * @param[in] dist_c1c2		Pointer to the matrix of dist between centers
 * @param[in] lower		Lower bound l(x,c)
 * @param[in] upper		Upper bound u(x)
 * @param[in] verbose		Flag for quiet mode
 * 
 * @return 0	If the function run sucessfully
 *         -1	If there is something wrong
 */
int elkans_kmeans(float **objects, int dims, int local_N, int K, float threshold, int *membership,
        float **clusters, MPI_Comm comm, int local_dist_cal, float **dist_c1c2,
        float **lower, float *upper, int verbose){

    float pre_SSE = FLT_MAX, SSE = 0.0;

    float **newClusters;    // Local sum of coords of the points assigned to a center
    int *newClusterSize;    // Local num of points assigned to a center
    float **globalClusters;    // Global sum of coords of the points assigned to a center
    int *globalClusterSize;    // Global num of points assigned to a center
    float dist_xc1;        // Dist between a point and its assigned center c1
    float dist_xc;        // Dist between a point and its new center
    float *dist_cc;        // Dist between old and new center

    float **centersNew = allocate_2D_float_array(K, dims);
    float *s_of_centroid = (float*)malloc(K * sizeof(float)); // half of the d(c1,c2)
    int *r_of_x;        // Flag to check d(x,c(x)) is outdated or not
    r_of_x = (int*)malloc(local_N * sizeof(int));
    for(int i=0; i<local_N; i++)
        r_of_x[i] = 1;

    float **swap;
    int converge=0, it=0;
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Allocate memory for the variables
    newClusters = allocate_2D_float_array(K, dims);
    newClusterSize = (int *)calloc(K, sizeof(int));
    globalClusters = allocate_2D_float_array(K, dims);
    globalClusterSize = (int *)calloc(K, sizeof(int));
    dist_cc = (float*)calloc(K, sizeof(float));

    // Start the elkans iteration loop
    while(!converge && it < max_it){
	SSE = 0.0;
        // Reinitialize the values to 0
        reset_2D_float_array(newClusters, K, dims);
        memset(newClusterSize, 0, K * sizeof(int));
        reset_2D_float_array(globalClusters, K, dims);
        memset(globalClusterSize, 0, K * sizeof(int));

        // Step 1: Calculate dist d(c1,c2) for all centers
        for(int i=0; i<K; i++){
            float min_dist_c1c2 = FLT_MAX;
            for(int j=0; j<K; j++){
                if(i!=j){
                    dist_c1c2[i][j] = sqrt(calculate_dist(dims, clusters[i], clusters[j]));
                    local_dist_cal ++;
                    min_dist_c1c2 = fminf(min_dist_c1c2, dist_c1c2[i][j]);;
                }
            }
            s_of_centroid[i] = (0.5) * min_dist_c1c2;
        }

        // Step 2: Skip calculation for these points x such that u(x) <= s(c(x))
        for(int i=0; i<K; i++){
            for(int j=0; j<local_N; j++){
                if(upper[j] > s_of_centroid[membership[j]]){
                    // Step3: If point x satisfies given 3 conditinos
                    if( i!=membership[j] && upper[j] > lower[j][i]
                        && upper[j] > 0.5*dist_c1c2[membership[j]][i]){
                        // Step3a: If all conditions satisfied and flag r(x) is true, find d(x,c(x))
                        if(r_of_x[j] == 1){
                            dist_xc1= sqrt(calculate_dist(dims, objects[j], clusters[membership[j]]));

                            // Set upper bound each time d(x,c(x)) is calculated
                            upper[j] = dist_xc1;
                            r_of_x[j] = 0;
                            local_dist_cal ++;
                        }
                        else{
                            // If flag r(x) is false, use previous value stored in u(x) as d(x,c(x))
                            dist_xc1 = upper[j];
                        }

                        // Step3b: If d(x,c(x)) > l(x,c) or 0.5* d(c(x),c), find d(x,c)
                        if(dist_xc1 > lower[j][i] || dist_xc1 > (0.5*dist_c1c2[membership[j]][i])){
                            dist_xc = sqrt(calculate_dist(dims, objects[j], clusters[i]));

                            // Set lower bound each time d(x,c) is calculated.
                            lower[j][i] = dist_xc;
                            local_dist_cal ++;

                            // Reassign x to new centroid if the new dist is smaller
			    if(dist_xc < dist_xc1){
                                membership[j] = i;
                                r_of_x[j] = 1;	// set r(x) to 1 if d(x,c(x)) is outdated
			    }
                        }
                    }
                }
            }
        }

        // Step 4: Calculate new value of centroid as weighted mean of points assigned to the centroid
        // Calculate sum of point coordinates and number of points assigned to each centroid
        for(int i=0; i<local_N; i++){
            for(int j=0; j<dims; j++){
                newClusters[membership[i]][j] += objects[i][j];
            }
            newClusterSize[membership[i]] ++;
	    
	    // Calculate the squared error for the point and its cluster
	    float dist_to_center = sqrt(calculate_dist(dims, objects[i], clusters[membership[i]]));
	    SSE += dist_to_center * dist_to_center;
	}

	// Gather the SSEs computed by all processes
	float global_SSE = 0.0;
	MPI_Allreduce(&SSE, &global_SSE, 1, MPI_FLOAT, MPI_SUM, comm);
	SSE = global_SSE;

        // Calculate global sums of point coordinates and number of points for each centroid
        for(int i=0; i<K; i++){
            for(int j=0; j<dims; j++){
                MPI_Allreduce(&(newClusters[i][j]), &(globalClusters[i][j]), 1, MPI_FLOAT, MPI_SUM, comm);
            }
            MPI_Allreduce(&(newClusterSize[i]), &(globalClusterSize[i]), 1, MPI_INT, MPI_SUM, comm);
        }

        // Use global sums to compute new centroids, store in centroidsNew
        for(int i=0; i<K; i++){
            for(int j=0; j<dims; j++){
                if(globalClusterSize[i] != 0){
                    centersNew[i][j] = globalClusters[i][j] / globalClusterSize[i];
                }
                else{ // If no points in the cluster, re-initialize centroid
                    centersNew[i][j] = 0;
                }
            }

            // Calculate dist between the old and the new values of each centroid
            dist_cc[i] = sqrt(calculate_dist(dims, centersNew[i], clusters[i]));
            local_dist_cal ++;
        }
        
        // Step 5: Calcualte new lower bound l(x,c) as max{ l(x,c) - d(c, m(c)), 0}
        for(int i=0; i<local_N; i++){
            for(int j=0; j<K; j++){
                if(lower[i][j] - dist_cc[j] > 0)
                    lower[i][j] -= dist_cc[j];
                else
                    lower[i][j] = 0;
            }
        
            // Step 6: Calculate new upper bound u(x) as u(x) + d(c(x), m(c(x)))
            upper[i] += dist_cc[membership[i]];
            r_of_x[i] = 1;
        }

        // Step 7: Replace each center c by m(c):
        swap = clusters; clusters = centersNew; centersNew = swap;

        // Print New Centers for each iteration
        if(rank == 0 && verbose == 1){
            printf("\nCentroids in Iteration %d:\n", it);
            for(int i=0; i<K; i++){
                printf("Centroid %d : ", i);
                for(int j=0; j<dims; j++){
                    printf("\t%f", clusters[i][j]);
                }
                printf("   : Number of Points assigned to C%d = %d\n", i, globalClusterSize[i]);
            }
        }

        // Check convergence with SSE
	if(fabs((pre_SSE - SSE)/pre_SSE) < threshold) {
		converge = 1;
		if(rank == 0)
			printf("\nSSE = %f, Convergence Reached at Iteration %d!\n", SSE, it);
	}

	pre_SSE = SSE;
        it++;

    }
    if(it == max_it){
        if(rank == 0)
            printf("Maximum iterations reached without convergence.\n");
    }

    // Free allocated memory and swap pointers
    free_2D_float_array(newClusters, K);
    free_2D_float_array(globalClusters, K);
    free(newClusterSize);
    free(globalClusterSize);
    free(dist_cc);
    free(s_of_centroid);
    free(r_of_x);

    return 0;
}


float **allocate_2D_float_array(int rows, int cols) {
    float **array = (float **)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++) {
        array[i] = (float*)malloc(cols * sizeof(float));
    }
    return array;
}

void reset_2D_float_array(float **array, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        memset(array[i], 0, cols * sizeof(float));
    }
}

void free_2D_float_array(float **array, int rows) {
    for (int i = 0; i < rows; i++) {
        free(array[i]);
    }
    free(array);
}

