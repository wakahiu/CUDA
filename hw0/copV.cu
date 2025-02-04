#include <time.h>
#include <stdio.h>
#include <stdlib.h>	//rand, atoi
#include <sys/time.h>
#include <unistd.h>

#define MAX_N 10000000

#define GPU_CHECKERROR(err) (gpuCheckError(err,__FILE__,__LINE__))

static void gpuCheckError(cudaError_t err, const char * file, int line){
	if(err != cudaSuccess ){
		printf("%s in file  %s at line %d\n",
				cudaGetErrorString(err), __FILE__, line);
		exit(EXIT_FAILURE);
	}
}

//Determines if A & B are co-prime using the Euclidean Algorithm.
unsigned int __cop(unsigned int A, unsigned int B){
	unsigned int T;
	//Swap 'em
	if(B>A){
		T=B;
		B=A;
		A=T;
		
	}
	while(B){
		T=B;
		B=A%B;
		A = T;
	}
	return A==1;
}

//Finds the number of co-prime  numbers in two vectors A and B
unsigned int copV(unsigned int N, unsigned int * A, unsigned int * B){
	unsigned int numCop = 0;
	for(int i = 0 ; i < N; i++){
		numCop += __cop(A[i], B[i]);
	}
	return numCop;
}

//Calculates if correspoding numbers in two arrays are coprime
//and stores the result in the first array.
__global__ void copV_GPU(unsigned int N, unsigned int * A, unsigned int * B){
	unsigned int n = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(n >= N) return;
		
	int a = A[n];
	int b = B[n];
		
	unsigned int t;
	//Swap 'em
	if(b>a){
		t=b;
		b=a;
		a=t;
		
	}
	while(b){
		t=b;
		b=a%b;
		a = t;
	}
	A[n] = (a==1);
	
}

//Performs sum reduction of an array A. A must be aligned to blockDim.x
__global__ void reduce_GPU(unsigned int * A){
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;
   
    for( unsigned int S = blockDim.x/2; S > 0; S/=2 ){
        if( tid < S )
            A[idx] += A[idx+S];

        //Make sure all threads have finished before we continue
        __syncthreads();
    }
    
}

//Code to display device properties.
// http://gpucoder.livejournal.com/1064.html
void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %lu\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %lu\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %lu\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %lu\n",  devProp.totalConstMem);
    printf("Texture alignment:             %lu\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
}

int main(int argc, char * argv[]){

	//Create an array on N
	unsigned int N;
	unsigned int N2;
	size_t sz;
    size_t sz2;
	unsigned int * h_A;
	unsigned int * h_B;

	unsigned int nThreads ;
	
	srand(time(NULL));
	
	//Query device properties.
	int dev_count, best_dev;
	cudaDeviceProp dev_prop_curr;
	cudaDeviceProp dev_prop_max_;
	cudaDeviceProp *dev_prop_max = &dev_prop_max_;
	dev_prop_max->maxThreadsPerBlock = 0;
	dev_prop_max->multiProcessorCount = 0;
	dev_prop_max->clockRate = 0;
	cudaGetDeviceCount( &dev_count );
	for( int i = 0; i < dev_count ; i++ ){
		cudaGetDeviceProperties( &dev_prop_curr , i);
		printf("\n-------------------Device %d----------------------\n",i);
		printDevProp( dev_prop_curr );
		//Select the best device
		if( 	(dev_prop_curr.maxThreadsPerBlock > dev_prop_max->maxThreadsPerBlock) &&
			(dev_prop_curr.multiProcessorCount > dev_prop_max->multiProcessorCount) && 
			(dev_prop_curr.clockRate > dev_prop_max->clockRate) ){
			best_dev = i;
			dev_prop_max = &dev_prop_curr;
			nThreads = dev_prop_curr.maxThreadsPerBlock;
			}
		
	}
	cudaSetDevice(best_dev);
	printf("\nBest Device found %d\n",best_dev);
	
	if(argc < 2){
		//Fill the arrays with 1000 random numbers
		N = 1000;
		N2=((N-1)/nThreads + 1)*nThreads;
        sz = sizeof( unsigned int ) * N;
		//Make sure array size is a multiple of blockDim;
        sz2 = sizeof( unsigned int ) * N2;
    
		h_A = (unsigned int *)malloc( sz2 );
		h_B = (unsigned int *)malloc( sz );
		
		printf("Argc=%d Filling array A & B with %d integers %d\n", argc-1, N,N2);
		for(unsigned int i = 0 ; i < N; i++){
			h_A[i] = rand() % MAX_N;
			h_B[i] = rand() % MAX_N;
			//printf("%d\n%d\n",h_A[i],h_B[i]);
		}	
	}else if(argc == 2){
		//Fill the arrays with random numbers
		N = (unsigned int) atoi( argv[1] );
		N2=((N-1)/nThreads + 1)*nThreads;
		sz = sizeof( unsigned int ) * N;
        //Make sure array size is a multiple of blockDim;
        sz2 = sizeof( unsigned int ) * N2;
    
		h_A = (unsigned int *)malloc( sz2 );
		h_B = (unsigned int *)malloc( sz );
	
		printf("\nArgc=%d Filling array A & B with %d integers and %d\n\n", argc-1, N,N2);
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
		N2=((N-1)/nThreads + 1)*nThreads;
		rewind(fr);
		//Now re-allocate a new array
		sz = sizeof( unsigned int ) * N;
        //Make sure array size is a multiple of blockDim;
        sz2 = sizeof( unsigned int ) * N2;
		printf("Starting %d, %d\n", N,N2);

		h_A = (unsigned int *)malloc( sz2 );
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
	
	//--------------------------------------------------------------------------
	//Serial Version
	//
	//--------------------------------------------------------------------------
	struct timeval t0, t1, t2;
	gettimeofday(&t0,0);
	unsigned int numCoPrimes = copV(N, h_A, h_B);
	gettimeofday(&t1,0);
	float timdiff1 = (1000000.0*(t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec)) / 1000000.0;
	printf ("\nDone: time taken for serial version is %3.1f s\n", timdiff1);
	printf("Number co-primes found by CPU = %d\n\n", numCoPrimes);


	//--------------------------------------------------------------------------
	//Parallel Version
	//
	//--------------------------------------------------------------------------
	
	printf("Best device %d\n",best_dev);
	//Allocate vectors and count in device memory
	gettimeofday(&t1,0);
	unsigned int * d_A;
	GPU_CHECKERROR(  cudaMalloc(&d_A,sz2));
	unsigned int * d_B;
	GPU_CHECKERROR(  cudaMalloc(&d_B,sz));
	GPU_CHECKERROR(  cudaMemset((void *) &d_A[N],0,sz2-sz) );		//Pad with 0's
	
	//Copy vectors from host to device memory
	GPU_CHECKERROR(  cudaMemcpy(d_A,h_A,sz2,cudaMemcpyHostToDevice));
	GPU_CHECKERROR(  cudaMemcpy(d_B,h_B,sz,cudaMemcpyHostToDevice));
	
	unsigned int nBlocks = (N + nThreads - 1)/nThreads;

    copV_GPU<<< nBlocks, nThreads>>>(N,d_A,d_B);

	cudaDeviceSynchronize();	
	
	//Reduce 'em
	reduce_GPU<<< nBlocks, nThreads>>>(d_A);
	cudaDeviceSynchronize();
	
	//Fetch the results;
	GPU_CHECKERROR(  cudaMemcpy( (void *) h_A,
						(void *) d_A,
						sz2,
						cudaMemcpyDeviceToHost) );
				
	int sum = 0;
	for( int i = 0; i < N2 ; i++ ){
		if(!(i%nThreads)){
			sum += h_A[i];
			//printf("\n\n");
		}
		//printf("h_A[%d]=%u\t",i,h_A[i]);
	}
	printf("Number co-primes found by GPU = %d Num reduced %d\n", sum, N2);	
	
	gettimeofday(&t2,0);
    float timdiff2 = (1000000.0*(t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec)) / 1000000.0;
    printf ("done: time taken for parallel version is %3.1f s threads %d\n", timdiff2, 0);

	//Free device memory
	GPU_CHECKERROR( cudaFree(d_A) );
	GPU_CHECKERROR( cudaFree(d_B) );
	
	
	//Free Host memory
	free(h_A);
	free(h_B);
}

