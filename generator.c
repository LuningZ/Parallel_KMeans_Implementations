//#define _POSIX_C_SOURCE 2
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<unistd.h>

#define M_PI 3.1415926

void print_usage(){
	printf("Usage: ./generate [-d dimensions] [-n num_points] [-m min_range] [-M max_range] [-f filename] [-s use_fixed_seed] [-c num_centroids] [-v variance]\n");
}

typedef struct {
	float *coords;
} Centroid;

int main(int argc, char **argv){
	int DIMENSIONS = 2;
	int NUM_POINTS = 1000;
	float MIN_RANGE = -10.0;
	float MAX_RANGE = 10.0;
	char *FILENAME = "datafile.txt";
	int USE_FIXED_SEED = 0;	// Flag for fixed seed
	int USE_CENTROIDS = 0;	// Flag to check if centroids should be used
	int NUM_CENTROIDS = 5;  // Number of centroids
	float VARIANCE = 2.0;   // Variance around the centroids
	Centroid* centroids;
	
	int opt;
	while((opt = getopt(argc, argv, "d:n:m:M:f:hsc:v:")) != -1){
		switch(opt){
			case 'd':
				DIMENSIONS = atoi(optarg); 
				if(DIMENSIONS <= 0){
					fprintf(stderr, "Number of dimensions should be positive.\n");
					return -1;
				} break;
			case 'n':
				NUM_POINTS = atoi(optarg);
				if(NUM_POINTS <= 0){
					fprintf(stderr, "Number of points should be positive.\n");
					return -1;
				} break;
			case 'm':
				MIN_RANGE = atof(optarg); break;
			case 'M':
				MAX_RANGE = atof(optarg); 
				if(MAX_RANGE <= MIN_RANGE){
					fprintf(stderr, "Max range should be greater than min range.\n");
					return -1;
				} break;
			case 'f':
				FILENAME = optarg; break;
			case 'h':
				print_usage();
				return 0;
			case 's':
				USE_FIXED_SEED = 1; break;
			case 'c':
				NUM_CENTROIDS = atoi(optarg);
				if(NUM_CENTROIDS <= 0){	                    
					fprintf(stderr, "Number of centroids should be positive.\n");
					return -1;
				}
				USE_CENTROIDS = 1;
				break;
			case 'v':
				VARIANCE = atof(optarg);
				USE_CENTROIDS = 1;
				break;
			default:
				fprintf(stderr, "Invalid option given\n");
				print_usage();
				return -1;
		}
	}

	FILE *file = fopen(FILENAME, "w");
	if(file == NULL){
		fprintf(stderr, "Could not open file %s\n", FILENAME);
		return -1;
	}

	// Select the seeding method based on '-s' option
	if(USE_FIXED_SEED){
		srand(12345);	// Use a fixed seed for reproducible results
	}
	else{
		srand(time(NULL));	// Use current time for varied results
	}

	// Generate the centroids
	if(USE_CENTROIDS){
		// Generate Centroids
		centroids = (Centroid *)malloc(NUM_CENTROIDS * sizeof(Centroid));
		printf("Generated Centroids:\n");
		for(int i = 0; i < NUM_CENTROIDS; i++){
			centroids[i].coords = (float *)malloc(DIMENSIONS * sizeof(float));
			printf("Centroid %d: ", i+1);
			for(int j = 0; j < DIMENSIONS; j++){
				centroids[i].coords[j] = MIN_RANGE + ((float)rand() / RAND_MAX) * (MAX_RANGE - MIN_RANGE);
				printf("%.6f", centroids[i].coords[j]);
				if(j < DIMENSIONS -1)
					printf(", ");
			}
			printf("\n");
		}
	}
	
	for(int i=0; i<NUM_POINTS; i++){
		if(USE_CENTROIDS) {
			Centroid chosenCentroid = centroids[rand() % NUM_CENTROIDS];
			for(int j=0; j<DIMENSIONS; j++){
				// Box-Muller transform
				float u1 = (float)rand() / RAND_MAX;
				float u2 = (float)rand() / RAND_MAX;
				float z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

				float varied_variance = z * VARIANCE;
				float num = chosenCentroid.coords[j] + varied_variance;
				fprintf(file, "%.6f", num);
				if(j < DIMENSIONS - 1)
					fprintf(file, " ");
			}
		} else{
			for(int j=0; j<DIMENSIONS; j++){
				float num = MIN_RANGE + ((float)rand() / RAND_MAX) * (MAX_RANGE - MIN_RANGE);
				fprintf(file, "%.6f", num);
				if(j < DIMENSIONS - 1)
					fprintf(file, " ");
			}
		}
		fprintf(file, "\n");
	}

	fclose(file);
	printf("%s has been written with %d random points.\n", FILENAME, NUM_POINTS);
	printf("Details:\n");
	printf("\tDimensions: %d\n", DIMENSIONS);
	printf("\tMinimum Range: %.2f\n", MIN_RANGE);
	printf("\tMaximum Range: %.2f\n", MAX_RANGE);

	if(USE_CENTROIDS){
		for(int i = 0; i < NUM_CENTROIDS; i++){
			free(centroids[i].coords);
		}
		free(centroids);
	}
	return 0;
}
