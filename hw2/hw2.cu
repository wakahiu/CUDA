#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <assert.h>

// STUDENTS: be sure to set the single define at the top of this file, 
// depending on which machines you are running on.
#include "im1.h"

using namespace std;
#define N_THREADS_X 512

// handy error macro:
#define GPU_CHECKERROR( err ) (gpuCheckError( err, __FILE__, __LINE__ ))
static void gpuCheckError( cudaError_t err,
                          const char *file,
                          int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
               file, line );
        exit( EXIT_FAILURE );
    }
}




//
// your __global__ kernel can go here, if you want:
//
__global__ void BW_GPU(float * d_imageArray, int size){
	
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	
	//Bounds check
	if(i>size)
		return;
		
	unsigned int idx = i * 3;
            
    float L = 0.2126f*d_imageArray[idx] + 
              0.7152f*d_imageArray[idx+1] + 
              0.0722f*d_imageArray[idx+2];

    d_imageArray[idx] = L;
    d_imageArray[idx+1] = L;
    d_imageArray[idx+2] = L;
	
}


int main (int argc, char *argv[])
{
 

	int numStreams = atoi(argv[2]);
    printf("Filename %s number of pieces to split image into %d\n", argv[1],numStreams);
       
    int w, h;   // the width & height of the image, used frequently!

    float *h_imageArray;
    readOpenEXRFile (argv[1], &h_imageArray, w, h);
    
    int N = w*h;
    float * d_imageArray;
    float * h_imageArray_pinned;
    float ms;
    cudaEvent_t start;
    cudaEvent_t stop;
    
   	GPU_CHECKERROR( cudaEventCreate(&start));
    GPU_CHECKERROR( cudaEventCreate(&stop));
    GPU_CHECKERROR( cudaEventRecord(start,0));
    
    cudaDeviceProp dev;
    unsigned int chunkSz = ceil((float)N/numStreams);
    cudaStream_t streams[numStreams];
        
   	dim3 numBlocks( ceil(float(N)/N_THREADS_X),1,1 );
    dim3 numThreads( N_THREADS_X, 1, 1  );
    
    cudaGetDeviceProperties(&dev, 0);
    
    assert(dev.asyncEngineCount);
    
    GPU_CHECKERROR( cudaMalloc((void **)&d_imageArray, sizeof(float)*w*h*3.0) );
    GPU_CHECKERROR( cudaMallocHost((void**)&h_imageArray_pinned, sizeof(float)*N*3.0) );
    
    
    //Create an array of streams
    for(int i = 0; i < numStreams; i++){
    	GPU_CHECKERROR( cudaStreamCreate(&streams[i]) );
    }
    
    //One copy engine. Need to make
    if( dev.asyncEngineCount==1 ){
    	cout << "1 copy engine" << endl;
    	
    	/* 
    	* copy image to page locked memory then to device.
    	*/
		for(int i =0; i < numStreams ; i++){
			int offset = i*chunkSz;
			size_t sz = (i == numStreams-1)? N-offset : chunkSz;
			
			memcpy(	&h_imageArray_pinned[offset*3], 
					&h_imageArray[offset*3], 
					sz*sizeof(float)*3 );
					
			 GPU_CHECKERROR( cudaMemcpyAsync( 	&d_imageArray[offset*3],
			 									&h_imageArray_pinned[offset*3],
			 									sz*sizeof(float)*3 ,
			 									cudaMemcpyHostToDevice,
			 									streams[i] ));
		}
		/*
    	* launch the kernel.
    	*/
    	for(int i = 0; i < numStreams; i++){
    		int offset = i*chunkSz;
			size_t sz = (i == numStreams-1)? N-offset : chunkSz;
    		BW_GPU<<< numBlocks, numThreads,0, streams[i]>>>( &d_imageArray[offset*3],sz);
   		}
    
    	/*
    	* Copy the image back into host memory
    	*/
		for(int i =0; i < numStreams ; i++){
			int offset = i*chunkSz;
			size_t sz = (i == numStreams-1)? N-offset : chunkSz;
			
			 GPU_CHECKERROR( cudaMemcpyAsync( 	&h_imageArray_pinned[offset*3],
			 									&d_imageArray[offset*3],
			 									sz*sizeof(float)*3,
			 									cudaMemcpyDeviceToHost,
			 									streams[i] ));
		}

    }else if( dev.asyncEngineCount==2 ){

    	cout << "2 copy engines" << endl;
		for(int i =0; i < numStreams ; i++){
			int offset = i*chunkSz;
			size_t sz = (i == numStreams-1)? N-offset : chunkSz;
			
			/* 
			* copy image to page locked memory then to device.
			*/
			memcpy(	&h_imageArray_pinned[offset*3], 
					&h_imageArray[offset*3], 
					sz*sizeof(float)*3 );
					
			 GPU_CHECKERROR( cudaMemcpyAsync( 	&d_imageArray[offset*3],
			 									&h_imageArray_pinned[offset*3],
			 									sz*sizeof(float)*3 ,
			 									cudaMemcpyHostToDevice,
			 									streams[i] ));
			/*
			* launch the kernel.
			*/
    		BW_GPU<<< numBlocks, numThreads,0, streams[i]>>>( &d_imageArray[offset*3],sz);

    
			/*
			* Copy the image back into host memory
			*/
			 GPU_CHECKERROR( cudaMemcpyAsync( 	&h_imageArray_pinned[offset*3],
			 									&d_imageArray[offset*3],
			 									sz*sizeof(float)*3,
			 									cudaMemcpyDeviceToHost,
			 									streams[i] ));
		}
    }
    
    //Destroy the array of streams
    for(int i = 0; i < numStreams; i++){
    	GPU_CHECKERROR( cudaStreamDestroy(streams[i]) );
    }

    // All your work is done. Here we assume that you have copied the 
    // processed image data back, from the device to the host, into the
    // original host array h_imageArray. You can do it some other way,
    // this is just a suggestion
   
  	GPU_CHECKERROR( cudaEventRecord(stop,0));
  	GPU_CHECKERROR( cudaEventSynchronize(stop));
  	GPU_CHECKERROR( cudaEventElapsedTime(&ms, start, stop) );
    
    printf("Time taken (ms): %f\n", ms);
    printf("writing output image hw2_gpu.exr %d\n", N);
    writeOpenEXRFile ("hw2_gpu.exr", h_imageArray_pinned, w, h);
    
    //Deallocate memory
    free (h_imageArray);
    GPU_CHECKERROR( cudaFree((void *)d_imageArray ));
    GPU_CHECKERROR( cudaFreeHost((void*)h_imageArray_pinned ));

    return 0;
}


