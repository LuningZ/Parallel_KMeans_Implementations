#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include<mpi.h>
#include "mpi_util.h"

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


/*mpi_read*/
float** mpi_read(char *filename, int *local_N, int *dims, MPI_Comm comm){
	
	float **objects = NULL;	// Whole data set
	float **local_objs; 	// Data set for each process
	int N;			// Total number of data points

	int rank, nproc;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nproc);

	/* Only rank 0 reads the entire data set */	
	if(rank == 0){
		// Only rank 0 reads the entire data set
		objects = file_read(filename, &N, dims);
	}

	/* Broadcast dims and N to all processes */
	MPI_Bcast(dims, 1, MPI_INT, 0, comm);
	MPI_Bcast(&N, 1, MPI_INT, 0, comm);

	int base = N / nproc;	// Base number of data points Each process will get
	int extra = N % nproc;	// Number of processes that will get 1 extra data point

	/* Calculate the number of data points (local_N) for each process */
	*local_N = rank < extra ? base+1 : base;
	
	if(*local_N < nproc){
		printf("Error: number of data points must be larger than the number of MPI processes.\n");
		MPI_Finalize();
	        exit(-1);
	}
	
	/* Allocate memory for local data on each process */
	local_objs = (float**)malloc((*local_N) * sizeof(float*));
	for (int i=0; i<(*local_N); i++) {
		local_objs[i] = (float*)malloc((*dims) * sizeof(float));
	}

	if(rank == 0){
		// Rank 0 distributes the data to all other procs
		int offset = *local_N;
		for(int i=1; i<nproc; i++){
			int num_points = i<extra ? base+1 : base;

			for(int j=0; j<num_points; j++){
				MPI_Send(objects[offset+j], *dims, MPI_FLOAT, i, j, comm);
			}
			offset += num_points;
		}

		// Copy the data for rank 0
		for(int i=0; i<(*local_N); i++){
			memcpy(local_objs[i], objects[i], (*dims)*sizeof(float));
		}

		// Cleanup the complete data
		for(int i=0; i<N; i++)
			free(objects[i]);
		free(objects);
	}
	else{
		// All other ranks receive their respective chunks of data from rank 0
		for(int i=0; i<*local_N; i++){
		MPI_Recv(local_objs[i], *dims, MPI_FLOAT, 0, i, comm, MPI_STATUS_IGNORE);
		}
	}

	return local_objs;	
}

/*mpi_write*/
int mpi_write(char *filename, float **objects, int K, int local_N, int dims, float **clusters, int *membership, MPI_Comm comm, int verbose){
	
	MPI_File fh;
	MPI_Status status;

	int rank, nproc, err;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nproc);

	// Dynamically allocate file name buffers
	size_t filename_len = strlen(filename);
	char *centers_file_name = malloc(filename_len + 16);
	char *memb_file_name = malloc(filename_len + 16);
	if(!centers_file_name || !memb_file_name){
		MPI_Abort(comm, MPI_ERR_OTHER);
		return -1;
	}
    	sprintf(centers_file_name, "%.*s_centers_mpi.txt", (int)(strlen(filename)-4), filename);
        sprintf(memb_file_name, "%.*s_membership_mpi.txt", (int)(strlen(filename)-4), filename);

	/* Output the coordinates of the cluster centres by rank 0 */
	if(rank == 0){
		if(verbose){
			printf("Writing coordinates of K=%d cluster centers to file \"%s\"\n", K, centers_file_name);
		}
		err = MPI_File_open(MPI_COMM_SELF, centers_file_name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
		if (err != MPI_SUCCESS){
			MPI_Abort(comm, err);
			return err;		
		}

		for(int i=0; i<K; i++){
			for(int j=0; j<dims; j++){
				int count = snprintf(NULL, 0, "%f ", clusters[i][j]);
				char *str = malloc(count+1);
				if(!str){
					MPI_File_close(&fh);
					MPI_Abort(comm, MPI_ERR_OTHER);
					return -1;
				}
				snprintf(str, count+1, "%f ", clusters[i][j]);
				MPI_File_write(fh, str, count, MPI_CHAR, &status);
				free(str);
			}
			MPI_File_write(fh, "\n", 1, MPI_CHAR, &status);
		}
		MPI_File_close(&fh);
	}

	// Output the coordinates of each data object followed by its membership 
	if(verbose && rank == 0){
		printf("Writing coordinates of each data object followed by its membership to file \"%s\"\n", memb_file_name);
	}

	err = MPI_File_open(comm, memb_file_name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
	if (err != MPI_SUCCESS){
		MPI_Abort(comm, err);
		return err;
	}

	for(int i=0; i<local_N; i++) {
		// Calculate the required memory for object_line dynamically
		int required_size = 0;
		for(int j=0; j<dims; j++){
			required_size += snprintf(NULL, 0, "%f ", objects[i][j]);
		}
		required_size += snprintf(NULL, 0, "%d\n", membership[i]);

		char *object_line = malloc(required_size + 1);
		if(!object_line){
			MPI_File_close(&fh);
			MPI_Abort(comm, MPI_ERR_OTHER);
			return -1;
		}

		int offset = 0;
		for(int j=0; j<dims; j++) {
			offset += sprintf(object_line + offset, "%f ", objects[i][j]);
			
		}
		offset += sprintf(object_line + offset, "%d\n", membership[i]);


		MPI_File_write_ordered(fh, object_line, strlen(object_line), MPI_CHAR, &status);
		free(object_line);
	}

	MPI_File_close(&fh);
	free(centers_file_name);
	free(memb_file_name);

	return 0;
                                    
}

/*elkans_write*/
int elkans_write(char *filename, float **objects, int K, int local_N, int dims, float **clusters, int *membership, MPI_Comm comm, int verbose){
	
	MPI_File fh;
	MPI_Status status;

	int rank, nproc, err;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nproc);

	// Dynamically allocate file name buffers
	size_t filename_len = strlen(filename);
	char *centers_file_name = malloc(filename_len + 16);
	char *memb_file_name = malloc(filename_len + 16);
	if(!centers_file_name || !memb_file_name){
		MPI_Abort(comm, MPI_ERR_OTHER);
		return -1;
	}
    	sprintf(centers_file_name, "%.*s_centers_elkans.txt", (int)(strlen(filename)-4), filename);
        sprintf(memb_file_name, "%.*s_membership_elkans.txt", (int)(strlen(filename)-4), filename);

	/* Output the coordinates of the cluster centres by rank 0 */
	if(rank == 0){
		if(verbose){
			printf("Writing coordinates of K=%d cluster centers to file \"%s\"\n", K, centers_file_name);
		}

		err = MPI_File_open(MPI_COMM_SELF, centers_file_name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
		if (err != MPI_SUCCESS){
			MPI_Abort(comm, err);
			return err;		
		}

		for(int i=0; i<K; i++){
			for(int j=0; j<dims; j++){
				int count = snprintf(NULL, 0, "%f ", clusters[i][j]);
				char *str = malloc(count+1);
				if(!str){
					MPI_File_close(&fh);
					MPI_Abort(comm, MPI_ERR_OTHER);
					return -1;
				}
				snprintf(str, count+1, "%f ", clusters[i][j]);
				MPI_File_write(fh, str, count, MPI_CHAR, &status);
				free(str);
			}
			MPI_File_write(fh, "\n", 1, MPI_CHAR, &status);
		}
		MPI_File_close(&fh);
	}

	// Output the coordinates of each data object followed by its membership 
	if(verbose && rank == 0){
		printf("Writing coordinates of each data object followed by its membership to file \"%s\"\n", memb_file_name);
	}

	err = MPI_File_open(comm, memb_file_name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
	if (err != MPI_SUCCESS){
		MPI_Abort(comm, err);
		return err;
	}


	for(int i=0; i<local_N; i++) {
		// Calculate the required memory for object_line dynamically
		int required_size = 0;
		for(int j=0; j<dims; j++){
			required_size += snprintf(NULL, 0, "%f ", objects[i][j]);
		}
		required_size += snprintf(NULL, 0, "%d\n", membership[i]);

		char *object_line = malloc(required_size + 1);
		if(!object_line){
			MPI_File_close(&fh);
			MPI_Abort(comm, MPI_ERR_OTHER);
			return -1;
		}

		int offset = 0;
		for(int j=0; j<dims; j++) {
			offset += sprintf(object_line + offset, "%f ", objects[i][j]);
			
		}
		offset += sprintf(object_line + offset, "%d\n", membership[i]);


		MPI_File_write_ordered(fh, object_line, strlen(object_line), MPI_CHAR, &status);
		free(object_line);
	}

	MPI_File_close(&fh);
	free(centers_file_name);
	free(memb_file_name);

	return 0;
}
