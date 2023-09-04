#ifndef SEQ_KMEANS_H
#define SEQ_KMEANS_H

/**
 * @brief Calculate square of Euclidean distance between two multi-dimensional points.
 *
 * @param[in] dims	Dimension of the data points
 * @param[in] p1	Pointer to the first point
 * @param[in] p2	Pointer to the second point
 *
 * @return dist 	The square of Euclidean distance between p1 and p2
 */
float calculate_dist(int dims, float *p1, float *p2);

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
int find_nearest_cluster(int K, int dims, float *object, float **clusters);

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
int seq_kmeans(float **objects, int dims, int N, int K, float threshold, int *membership, float **clusters);
	
#endif /* SEQ_KMEANS_H */
