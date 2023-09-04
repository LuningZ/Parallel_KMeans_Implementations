#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <float.h>


/**
 * @brief Calculate square of Euclidean distance between two multi-dimensional points.
 *
 * @param[in] dims	Dimension of the data points
 * @param[in] p1	Pointer to the first point
 * @param[in] p2	Pointer to the second point
 *
 * @return dist 	The square of Euclidean distance between p1 and p2
 */
float calculate_dist(int dims, float *p1, float *p2){
	float dist= 0.0;	// Euclidean distance
	for(int i=0; i<dims; i++){
		dist += (p1[i]-p2[i]) * (p1[i]-p2[i]);
	}
	return dist;
}


/**
 * @brief Find the nearest cluster for a data point
 *
 * @param[in] K		Number of clusters
 * @param[in] dims	Dimension of data points
 * @param[in] object	Pointer to the data point
 * @param[in] clusters	Pointer to the matrix of cluster centroids
 *
 * @return index	The nearest cluster id
 */
int find_nearest_cluster(int K, int dims, float *object, float **clusters){
	int index = 0;	// store the closest cluster id
	float dist;	// Euclidean distance

	/* set the initial min_dist as the dist to the cluster[0] */
	float min_dist = calculate_dist(dims, object, clusters[0]);

	/* find the cluster id that has min distance to object */
	for(int i=1; i<K; i++){
		dist = calculate_dist(dims, object, clusters[i]);
		if(dist < min_dist){
			min_dist = dist;
			index =i;
		}
	}
	return(index);
}

/**
 * @brief Return an array of cluster centers of size [K][dims]
 *
 * @param[in] objects		Pointer to the matrix of data points
 * @param[in] dims		Dimension of data points
 * @param[in] N			Number of data points
 * @param[in] K			Number of clusters
 * @param[in] threshold		objects change membership
 * @param[in] membership	the cluster id for each data object
 * @param[in] clusters		Pointer to the matrix of cluster centroids
 * 
 * @return 0	If the function run sucessfully
 *         -1	If there is something wrong
 */
int seq_kmeans(float **objects, int dims, int N, int K, float threshold, int *membership, float **clusters){
	
	int index, it=0;
	int *newClusterSize; // objects assigned in each new cluster
	float SSE, pre_SSE = FLT_MAX;	// initialize previousSSE to a large value

	float **newClusters; // size: [K][dims]

	/* initialize membership array to -1 */
	for(int i=0; i<N; i++)
		membership[i] = -1;

	/* need to initialize newClusterSize and newClusters[0] to all 0 */
	newClusterSize = (int *)calloc(K, sizeof(int));
	if(newClusterSize == NULL){
		printf("Error: Unable to allocate memory for newClusterSize!\n");
		return -1;
	}

	newClusters = (float**)malloc(K * sizeof(float*));
	if(newClusters == NULL){
		printf("Error: Unable to allocate memory for newClusters!\n");
		free(newClusterSize);
		return -1;
	}

	newClusters[0] = (float*)calloc(K * dims, sizeof(float));
	if(newClusters[0] == NULL) {
		printf("Error: Unable to allocate memory for newClusters[0]!\n");
		free(newClusters);
		free(newClusterSize);
		return -1;
	}

	for(int i=1; i<K; i++)
		newClusters[i] = newClusters[i-1] + dims;

	do{
		SSE = 0.0;

		for(int i=0; i<N; i++){
			/* find the array index of nearest cluster center */
			index = find_nearest_cluster(K, dims, objects[i], clusters);

			/* Calculate SSE */
			float dist = calculate_dist(dims, objects[i], clusters[index]);
			SSE += dist;

			/* update the membership*/
			membership[i] = index;

			/* update new cluster center: sum of objects located within */
			newClusterSize[index]++;
			for(int j=0; j<dims; j++)
				newClusters[index][j] += objects[i][j];
		}

		/* average the sum and replace old cluster center with newClusters */
		for(int i=0; i<K; i++){
			for(int j=0; j<dims; j++){
				if (newClusterSize[i] >0)
					clusters[i][j] = newClusters[i][j] / newClusterSize[i];
				newClusters[i][j] = 0.0; // set back to 0 
			}
			newClusterSize[i] = 0; // set back to 0
		}
		
		if ((fabs(pre_SSE - SSE)/pre_SSE) < threshold) break;
		pre_SSE = SSE;
	} while (it++ < 1000);

	if(it < 1000)
		printf("Clustering converged at iteration %d.\n", it);
	else
		printf("Clustering reached the maximum number of iterations 1000.\n");

	printf("SSE = %f, threshold = %f, it = %d\n", SSE, threshold, it);


	free(newClusters[0]);
	free(newClusters);
	free(newClusterSize);

	return 0;

}	
