/**
 * @file seq_main.c
 * @brief A sequential implementation of K-means clustering.
 * 	Allows the user to specify the data file, number of clusters, and initial centers.
 * 	The result, which includes cluster centers and memberships, is saved to a file.
 * @author Luning
 * @version 1.0
 * @date 2023-07-17
 */

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<getopt.h>
#include<sys/time.h>
#include <sys/types.h>

#include"seq_kmeans.h"

double wtime();
float** file_read(char* filename, int* N, int* dims);
void read_k_objects(char* filename, int K, int dims, float** clusters);
int file_write(char *filename, int K, int N, int dims, float **clusters, int *membership, float **objects, int verbose);
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
	float threshold = 1e-6;

	int dims;			// dims: number of coordinates
	int N;				// N: number of data points
	int *membership;		// membership: the cluster id for each data object
	float **objects;		// objects: 2d array to record all the data points
	float **clusters;		// clusters: 2d array to record cluster centers
	double io_timing, clustering_timing;	// record the timings

	// Parse command line arguments
	int opt;
	while((opt = getopt(argc, argv, "f:c:k:tqh")) != -1){
		switch(opt) {
			case 'f': 
				filename = optarg; break;
		        case 'c': 
				center_filename = optarg; break;
			case 'k': 
				K = atoi(optarg); 
				if(K <= 1){
					fprintf(stderr, "Number of clusters should be more than 1.\n");
					return -1;
				} break;
			case 't': 
				print_timing = 1; break;
			case 'q': 
				verbose = 0; break;
			case 'h':
				PrintUsage(argv[0], threshold); return 0; break;
			default: 
				fprintf(stderr, "Invalid option given\n");
				PrintUsage(argv[0], threshold); 
				return -1;
		}
	}
	if(filename == NULL) PrintUsage(argv[0], threshold);
	if (center_filename == NULL){	// Set default center_filename to filename
		center_filename = filename;
	}

	/* Read data from file */
	io_timing = wtime();	// Start the timer
	printf("Reading data points from file %s\n", filename);

	objects = file_read(filename, &N, &dims);
	if(objects == NULL) return -1;
	if(N < K){	// Check the validity of N
		printf("Error: number of clusters must be less than the number of data points to be clustered.\n");
		free(objects[0]);
		free(objects);
		return -1;
	}

	/* Allocate memory for cluster centers (coordinates of cluster centers) */
	clusters = (float**) malloc(K * sizeof(float*));
	clusters[0] = (float*) malloc(K * dims * sizeof(float));
	if(!clusters || !clusters[0]){
		fprintf(stderr, "Error allocating memory for clusters.\n");
		return -1;
	}
	for(int i=1; i<K; i++)
		clusters[i] = clusters[i-1] + dims;

	/* Set the initial cluster centers */
	if(center_filename != filename){	
		// Use the specified points as the initial cluster centers
		printf("reading initial %d centers from file %s\n", K, center_filename);
		read_k_objects(center_filename, K, dims, clusters);
	}
	else {	// Use the first K points from the file as the initial cluster centers
		printf("selecting the first %d elements as initial centers\n", K);
		for(int i=0; i<K; i++){
			for(int j=0; j<dims; j++){
				clusters[i][j] = objects[i][j];
			}
		}
	}

	/* Check initial cluster centers for repeatition */
	if(check_repeated_clusters(K, dims, clusters) == 0){
		printf("Error: some initial clusters are repeated. Please select distinct initial centers\n");
		return -1;
	}

	/* Print the initial centers */
	if(verbose){
		printf("The Sorted initial cluster centers:\n");
		for(int i=0; i<K; i++){
			printf("clusters[%d]=", i);
			for(int j=0; j<dims; j++){
				printf("%6.2f", clusters[i][j]);
			}
			printf("\n");
		}
	}

	io_timing = wtime() - io_timing;	// Calculate io timing


	/* Start the timer for the clustering computation */
	clustering_timing = wtime();
	membership = (int *)malloc(N * sizeof(int));
	if(!membership){
		fprintf(stderr, "Error allocating memory for membership.\n");
		return -1;
	}
	/* Sequential K-Means */
	seq_kmeans(objects, dims, N, K, threshold, membership, clusters);
	clustering_timing = wtime() - clustering_timing; // Calculate clustering timing

	/* Print the results (cluster centers and membership) to files */
	file_write(filename, K, N, dims, clusters, membership, objects, verbose);

	free(membership);
	free(clusters[0]);
	free(clusters);
	free(objects[0]);
	free(objects);

	/* Print the timing performance to the screen */
	if(print_timing){
		printf("\nPerforming **** Sequential Kmeans ****\n");

		printf("Input file:	%s\n", filename);
		printf("N =	%d\n", N);
		printf("dims = 	%d\n", dims);
		printf("K =	%d\n", K);
		printf("threshold =	%.4f\n", threshold);

		printf("I/O time = 	%10.4f sec\n", io_timing);
		printf("Computation timing = 	%10.4f sec\n", clustering_timing);
	}

	return 0;
}	

/* Function to measure wall clock time */
double wtime(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + (double)tv.tv_usec / 1000000;
}

/* Function to read data points from the file 
 * 	The number of lines is the total number of data points, 
 * 	and the number of values on each line is the number of dimensions. 
 */
float** file_read(char* filename, int* N, int* dims){
	FILE *file = fopen(filename, "r");
	if(file == NULL){
		fprintf(stderr, "Could not open file %s\n", filename);
		exit(-1);
	}

	char *line = NULL;
	size_t len = 0;
	ssize_t read;

	float** data = NULL;
	*N = 0;
	*dims = 0;

	/* Count the lines (objects) and the number of coords (dims) in the first line */
	while((read = getline(&line, &len, file)) != -1){
		if(*dims == 0){	// Only need to calculate dims once
			char *token = strtok(line, " ");
			while(token){
				(*dims)++;
				token = strtok(NULL, " ");
			}
		}
		(*N)++;
	}

	/* Allocate memory for data points */
	data = (float**)malloc((*N)*sizeof(float*));
	if(!data){
		fprintf(stderr, "Error allocating memory for data.\n");
		exit(-1);
	}
	for(int i=0; i<*N; i++){
		data[i] = (float*)malloc((*dims)*sizeof(float));
		if(!data[i]){
			fprintf(stderr, "Error allocating memory for data.\n");
			exit(-1);
		}
	}

	/* Go back to the start of the file to read data */
	fseek(file, 0, SEEK_SET);
	int i=0;
	while((read = getline(&line, &len, file)) != -1){
		int j=0;
		char *token = strtok(line, " ");
		while(token){
			data[i][j] = atof(token);
			j++;
			token = strtok(NULL, " ");
		}
		i++;
	}

	free(line);
	fclose(file);
	return data;
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

/* Function to write the coordinates of the cluster centers to a file */
int file_write(char* filename, int K, int N, int dims, float** clusters, int *membership, float** objects, int verbose){
	/* Create a filename for the cluster centers */
	char outfile[1024];
	sprintf(outfile, "%.*s_centers.txt", (int)(strlen(filename)-4), filename);

	/* Open file for writing cluster centers */
	FILE *f_centers = fopen(outfile, "w");
	if(!f_centers){
		fprintf(stderr, "Error: Failed to open output file %s for cluster centers.\n", outfile);
		return -1;
	}

	/* Write the the coordinates of the cluster centers to the file */
	if(verbose){
		    printf("Here are the centers after clustering:\n");
	}
	for(int i=0; i<K; i++){
		for(int j=0; j<dims; j++){
			fprintf(f_centers, "%.6f ", clusters[i][j]);
			// If verbose is set to 1, print the coordinates to the screen as well
			if(verbose){
				printf("%.6f ", clusters[i][j]);
			}
		}
		fprintf(f_centers, "\n");
		if(verbose){
			printf("\n");
		}
	}
	fclose(f_centers);
	if(verbose){
		printf("Cluster centers saved to %s\n", outfile);
	}

	/* Create a filename for the cluster membership */
	sprintf(outfile, "%.*s_membership.txt", (int)(strlen(filename)-4), filename);

	/* Open file for writing cluster membership */
	FILE *f_membership = fopen(outfile, "w");
	if(!f_membership){
		fprintf(stderr, "Error: Failed to open output file %s for cluster membership.\n", outfile);
		return -1;
	}

	/* Write data points and cluster membership to the file */
	for(int i=0; i<N; i++){
		// Write the data points
		for(int j=0; j<dims; j++){
			fprintf(f_membership, "%.6f ", objects[i][j]);
		}
		// Write the membership
		fprintf(f_membership, "%d\n", membership[i]);
	}
	fclose(f_membership);

	if(verbose){
		printf("Data points with cluster membership saved to %s\n", outfile);
	}

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
