CC = gcc
MPICC = mpicc

CFLAGS = -g -Wall -Wextra -std=c99 -D_POSIX_C_SOURCE=200809L -lm

EXECS = seq_main  mpi_main elkans_main generator
all: $(EXECS)

seq_main: seq_main.c seq_kmeans.o
	$(CC) $(CFLAGS) -o $@ $^
seq_kmeans.o: seq_kmeans.c
	$(CC) $(CFLAGS) -c $<

mpi_main: mpi_main.c mpi_kmeans.o mpi_io.o
	$(MPICC) $(CFLAGS) -o $@ $^
mpi_kmeans.o: mpi_kmeans.c
	$(MPICC) $(CFLAGS) -c $<
mpi_io.o: mpi_io.c
	$(MPICC) $(CFLAGS) -c $<
elkans_main: elkans_main.c mpi_kmeans.o mpi_io.o elkans_kmeans.o
	$(MPICC) $(CFLAGS) -o $@ $^
elkans_kmeans.o: elkans_kmeans.c
	$(MPICC) $(CFLAGS) -c $<
generator: generator.c
	$(CC) $(CFLAGS) -o $@ $^

.PHONY: clean

clean:
	rm -f *.o *.txt $(EXECS)
