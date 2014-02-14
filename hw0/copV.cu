#include <time.h>
#include <stdio.h>
#include <stdlib.h>	//rand, atoi
#include <sys/time.h>

#define MAX_N 10000000

//Determines if A & B are co-prime using the Euclidean subtraction  Algorithm.
unsigned int __cop(unsigned int A, unsigned int B){
	while(A != B){
		if(A > B){
			A = A - B;
		}else{
			B = B - A;
		}
	}
	return A == 1;
}

//Finds the number of co-prime  numbers in two vectors A and B
unsigned int copV(unsigned int N, unsigned int * A, unsigned int * B){
	unsigned int numCop = 0;
	for(int i = 0 ; i < N; i++){
		numCop += __cop(A[i], B[i]);
	}
	return numCop;
}

__global__ void copV_GPU(unsigned int N, unsigned int * A, unsigned int * B, unsigned int * numCop){
	unsigned int n = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(n >= N)
		return;
		
	//Performance without these temp integers?
	int a = A[n];
	int b = B[n];
	
	while(a != b){
		if(a > b){
			a = a-b;
		}else{
			b = b-a;
		}
	}
	
	if( a == 1 )	//They are co-prime
		atomicAdd(numCop,2);
}

int main(int argc, char * argv[]){

	//Create an array on N
	unsigned int N;
	size_t sz;
	unsigned int * h_A;
	unsigned int * h_B;
	
	srand(time(NULL));
	
	if(argc < 2){
		//Fill the arrays with 1000 random numbers
		N = 1000;
		sz = sizeof( unsigned int ) * N;
		h_A = (unsigned int *)malloc( sz );
		h_B = (unsigned int *)malloc( sz );
		
		printf("Argc=%d Filling array A & B with %d integers\n", argc-1, N);
		for(unsigned int i = 0 ; i < N; i++){
			h_A[i] = rand() % MAX_N;
			h_B[i] = rand() % MAX_N;
			//printf("%d\n%d\n",h_A[i],h_B[i]);
		}	
	}else if(argc == 2){
		//Fill the arrays with random numbers
		N = (unsigned int) atoi( argv[1] );
		sz = sizeof( unsigned int ) * N;
		h_A = (unsigned int *)malloc( sz );
		h_B = (unsigned int *)malloc( sz );
	
		printf("\nArgc=%d Filling array A & B with %d integers\n\n", argc-1, N);
		for(unsigned int i = 0 ; i < N; i++){
			h_A[i] = rand() % MAX_N;
			h_B[i] = rand() % MAX_N;
			//printf("%d\n%d\n",h_A[i],h_B[i]);
		}
	}else if ( argc > 2){
	
		//Read from file
		printf( "Reading input from file %s\n",argv[2]);
		
		
		FILE *fr;
		fr = fopen(argv[2], "r");
		if( !fr ){
			fprintf(stderr, "Can't open input file!\n");
			exit(1);
		}
		
		char oneword[100];
		int count = 0;
		
		//First count the number of entries in the file
		while(fscanf(fr,"%s",oneword) != EOF ){
			count ++;
		}
		N = count/2;
		printf("Starting %d\n", N);
		rewind(fr);
		//Now re-allocate a new array
		sz = sizeof( unsigned int ) * N;
		h_A = (unsigned int *)malloc( sz );
		h_B = (unsigned int *)malloc( sz );
		
		count = 0;
		while(fscanf(fr,"%s",oneword) != EOF ){
		
			//sprintf("%s\n",oneword);
			if(count % 2){
				h_B[count/2] = atoi( oneword );
				//printf("%d\n",h_B[count/2]);
			}
			else{
				h_A[count/2] = atoi( oneword );
				//printf("%d ",h_A[count/2]);
			}
			count ++;
		}
		
		
		fclose(fr);
	}
	
	//Serial Version
	struct timeval t0, t1, t2;
	gettimeofday(&t0,0);
	unsigned int numCoPrimes = copV(N, h_A, h_B);
	gettimeofday(&t1,0);
	float timdiff1 = (1000000.0*(t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec)) / 1000000.0;
	printf ("\nDone: time taken for serial version is %3.1f s\n", timdiff1);
	printf("Number co-primes found by CPU = %d\n\n", numCoPrimes);
	
	//Allocate vectors and count in device memory
	unsigned int * d_A;
	cudaMalloc(&d_A,sz);
	unsigned int * d_B;
	cudaMalloc(&d_B,sz);
	unsigned int * d_numCop;
	cudaMalloc((void **) & d_numCop, sizeof(unsigned int));
	
	//for( int nthr = 32; nthr <= 512 ; nthr*=2){
	gettimeofday(&t1,0);
	
	//Copy vectors from host to device memory
	cudaMemcpy(d_A,h_A,sz,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,h_B,sz,cudaMemcpyHostToDevice);
	cudaMemset((void *) d_numCop, 0, sizeof(unsigned int));
	
	unsigned int nThreads = 128;
	unsigned int nBlocks = (N + nThreads - 1)/nThreads;
	
	copV_GPU<<< nBlocks, nThreads>>>(N,d_A,d_B,d_numCop);

	
	cudaDeviceSynchronize();
	
	unsigned int h_numCop;
	
	cudaMemcpy( (void *) & h_numCop,
				(void *) d_numCop,
				sizeof(unsigned int),
				cudaMemcpyDeviceToHost);
				
	gettimeofday(&t2,0);
	printf("Number co-primers found by GPU = %d\n", h_numCop);
	
    float timdiff2 = (1000000.0*(t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec)) / 1000000.0;
    printf ("done: time taken for parallel version is %3.1f s threads %d\n", timdiff2, 0);
	
	//}

	//Free device memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_numCop);
	
	
	//Free Host memory
	free(h_A);
	free(h_B);
}

