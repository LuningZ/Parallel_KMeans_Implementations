/**
 * @file mpi_main.c
 * @brief A parallel implementation of K-means clustering using MPI.
 * 	Allows the user to specify the data file, number of clusters, and initial centers.
 * 	The result, which includes cluster centers and memberships, is saved to a file.
 * @author Luning
 * @version 3.0
 * @date 2023-08-15
 */

#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<mpi.h>
#include "mpi_util.h"

void read_k_objects(char* filename, int K, int dims, float** clusters);
int check_repeated_clusters(int K, int dims, float** clusters);

/* Print the usage of the program */
void PrintUsage(char *argv0, float threshold){
	char *help = 
		"Usage: %s [switches] -f filename -n num_clusters\n"
		"       -f filename    : file containing data to be clustered\n"
		"       -c centers     : file containing initial centers. default: filename\n"
		"       -k K	       : number of clusters (K must > 1)\n"
		"       -t             : print timing results (default no)\n"
		"       -q             : quiet mode\n"
		"       -h             : print this help information\n";
	fprintf(stderr, help, argv0, threshold);
	exit(-1);
}

int main(int argc, char **argv){

	// Variables initialization and setup
	char *filename = NULL;		// Original data file name
	char *center_filename = NULL;	// Initial centers file name
	int K = 2;			// Number of clusters (default: 2)
	int print_timing = 0;		// Option to print timings (default: not print)
	int verbose = 1;		// Option to set quiet mode (default: print)
	float threshold = 1e-8;

	int dims;			// dims: number of coordinates
	int N;				// N: number of data points
	int local_N;			// local_N: number of data on each procs
	int *membership;		// membership: the cluster id for each data object
	float **objects;		// objects: 2d array to record all the data points
	float **clusters;		// clusters: 2d array to record cluster centers
	double input_timing, clustering_timing, output_timing;	// record the timings

	int rank, nproc;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);

	// Parse command line arguments
	int opt;
	while((opt = getopt(argc, argv, "f:c:k:tqh")) != -1){
		switch(opt){
			case 'f':
				filename = optarg; break;
			case 'c':
				center_filename = optarg; break;
			case 'k':
				K = atoi(optarg); 
				if(K <= 1){
					fprintf(stderr, "Number of clusters should be more than 1.\n");
				} break;
			case 't':
				print_timing = 1; break;
			case 'q':
				verbose = 0; break;
			case 'h':
				if(rank==0) PrintUsage(argv[0], threshold);
				return 0; break;
			default:
				fprintf(stderr, "Invalid option given\n");
				if(rank==0) PrintUsage(argv[0], threshold);
				return -1;
		}
	}
	if(filename == NULL){
		if(rank==0) PrintUsage(argv[0], threshold);
		MPI_Finalize();
		return -1;
	}
	if(center_filename == NULL)	// Set default center_filename to filename
		center_filename = filename;

	/* Read data from file */
	MPI_Barrier(MPI_COMM_WORLD);
	input_timing = MPI_Wtime();	// start the timer
	if(rank == 0)
		printf("reading data points from file %s\n", filename);

	objects = mpi_read(filename, &local_N, &dims, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);

	// test the data was read in correctly
	/*if(rank == 2){
		printf("Process %d read the following data:\n", rank);
		for (int i = 0; i < local_N; i++) {
			printf("Data point %d: ", i);
			for (int j = 0; j < dims; j++) {
				printf("%f ", objects[i][j]);
			}
			printf("\n");
		}
	}*/

	/* Get the total number of data points N and check validity */
	MPI_Allreduce(&local_N, &N, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	if(N < K){
		if(rank == 0)
			printf("Error: number of clusters must be less than the number of data points to be clustered. \n");
		free(objects[0]);
		free(objects);
		MPI_Finalize();
		return -1;
	}

	/* Allocate memory for cluster centers (coordinates of cluster centers) */
	clusters = (float**)malloc(K * sizeof(float*));
	clusters[0] = (float*)malloc(K * dims * sizeof(float));
	for(int i=1; i<K; i++)
		clusters[i] = clusters[i-1] + dims;

	/* Set the initial cluster centers on rank 0 */
	if(rank == 0){
		if(center_filename != filename || local_N < K){
			// Use the specified points as the initial cluster centers
			printf("reading initial %d centers from file %s\n", K, center_filename);
			read_k_objects(center_filename, K, dims, clusters);
		}
		else{	// Use the first K points from the file as the initial cluster centers
			printf("selecting the first %d elements as initial centers\n", K);
			for(int i=0; i<K; i++)
				for(int j=0; j<dims; j++)
					clusters[i][j] = objects[i][j];
		}
	}
	/* Broadcast initial cluster centers to all processes */
	MPI_Bcast(clusters[0], K*dims, MPI_FLOAT, 0, MPI_COMM_WORLD);

	/* Check initial cluster centers for repeatition */
	if(check_repeated_clusters(K, dims, clusters) == 0){
		printf("Error: some initial clusters are repeated. Please select distinct initial centers\n\n");
		MPI_Finalize();
		return -1;
	}

	input_timing = MPI_Wtime() - input_timing;	// Calculate input timing

	/* Start the timer for the clustering computation */
	clustering_timing = MPI_Wtime();

	membership = (int*)malloc(local_N * sizeof(int));

	/* MPI K-Means */
	mpi_kmeans(objects, dims, local_N, K, threshold, membership, clusters, MPI_COMM_WORLD);

	clustering_timing = MPI_Wtime() - clustering_timing; // Calculate clustering timing

	/* Print the results (cluster centers and membership) to files */
	output_timing = MPI_Wtime();

	mpi_write(filename, objects, K, local_N, dims, clusters, membership, MPI_COMM_WORLD, verbose);

	output_timing = MPI_Wtime() - output_timing;


	free(membership);
	free(clusters[0]);
	free(clusters);
	free(objects[0]);
	free(objects);

	/* Print the timing performance to the screen */
	if(print_timing) {
		double max_input_timing, max_clustering_timing, max_output_timing;

		/* Get the max timing measured among all processes */
		MPI_Reduce(&input_timing, &max_input_timing, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&clustering_timing, &max_clustering_timing, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&output_timing, &max_output_timing, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

		/* Print timing results */
		if(rank == 0){
			printf("\nPerforming **** MPI Parallel Kmeans ****\n");
			printf("Input file: 	%s\n", filename);
			printf("N = 	%d\n", N);
			printf("dims = 	%d\n", dims);
			printf("K= 	%d\n", K);
			printf("threshold = 	%.6f\n", threshold);

			printf("I/O time (Input) = 	%10.4f sec\n", input_timing);
			printf("Computation timing = 	%10.4f sec\n", clustering_timing);
			printf("I/O time (Output) = 	%10.4f sec\n", output_timing);
		}
	}

	MPI_Finalize();
	return 0;
}

/* Check initial cluster centers for repeatition */
int check_repeated_clusters(int K, int dims, float** clusters){
	for(int i=0; i<K; i++){
		for(int j=i+1; j<K; j++){
			int are_equal = 1;
			for(int dim=0; dim<dims; dim++){
				if(clusters[i][dim] != clusters[j][dim]){
					are_equal=0;
					break;
				}
			}
			if(are_equal){
				return 0;
			}
		}
	}
	return 1;
}

/* Read the first K data points into the clusters array */
void read_k_objects(char* filename, int K, int dims, float** clusters){
	FILE *file = fopen(filename, "r");
	if(file == NULL){
		fprintf(stderr, "Could not open file %s\n", filename);
		return;
	}

	 for(int i=0; i<K; i++){
		 for(int j=0; j<dims; j++){
			 if(fscanf(file, "%f", &clusters[i][j]) != 1){
				 fprintf(stderr, "Error reading data point %d, dimension %d\n", i, j);
				 fclose(file);
				 return;
			 }
		 }
	 }
	 fclose(file);
}

