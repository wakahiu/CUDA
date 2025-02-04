#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>


// STUDENTS: be sure to set the single define at the top of this file, 
// depending on which machines you are running on.
#include "im1.h"

#define MAX_SHARED_MEM 16384

//Constant buffer for the gaussian kernel
#define D_MAX 50		//10000Bytes < 64KB
__constant__ float d_const_Gaussian[D_MAX*D_MAX];


// handy error macro:
#define GPU_CHECKERROR( err ) (gpuCheckError( err, __FILE__, __LINE__ ))
inline static void gpuCheckError( cudaError_t err,
                          const char *file,
                          int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
               file, line );
        exit( EXIT_FAILURE );
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

//
// your __global__ kernel can go here, if you want:
//
__global__ void blurr_GPU(float * d_imageArray,float * d_imageArrayResult, float *  d_dev_Gaussian, int w,int h, int r){
	
	int d = 2*r + 1;
	extern __shared__ float picBlock[];
	
	int x = blockDim.x*blockIdx.x + threadIdx.x; 
	int y = blockDim.y*blockIdx.y + threadIdx.y;

	if(x >= w || y >= h)
		return;
		
	int idx;
	unsigned int idxN, idxS, idxW, idxE;
	float tempR = 0.0;
    float tempG = 0.0;
    float tempB = 0.0;
	float Gaus_val = 0.0;
	int	shDim_x = (blockDim.x + 2*r);
	int	shDim_y = (blockDim.y + 2*r);
	int offset = shDim_x * shDim_y;
	int i, j;
	int iSh, jSh;
	
	//Copy the gaussian kernel into shared memory
	//Blocks that do not require boundry check
	if( (blockIdx.x*blockDim.x >=  r )		&& (blockIdx.y*blockDim.y >=  r) &&
		(((blockIdx.x+1)*blockDim.x +r) < w)	&& (((blockIdx.y+1)*blockDim.y +r) < h)){	
		
		//Collaborative loading into shared memory
		for( i = y-r, iSh = threadIdx.y ; i< (blockDim.y*(blockIdx.y + 1) + r)  ; i+=blockDim.y , iSh+=blockDim.y ){
			for( j = x-r, jSh = threadIdx.x ; j < (blockDim.x*(blockIdx.x + 1) + r)  ; j+=blockDim.x , jSh+=blockDim.x){
				picBlock[(iSh*shDim_x+jSh)] = d_imageArray[(i*w+j)*3];
				picBlock[(iSh*shDim_x+jSh)+offset] = d_imageArray[(i*w+j)*3+1];
				picBlock[(iSh*shDim_x+jSh)+offset*2] = d_imageArray[(i*w+j)*3+2];
			}
		}
	}
	//These blocks may access picture elements that are out of bounds
	else{
		int xLim = blockDim.x*(blockIdx.x + 1) > w ? w : (blockDim.x*(blockIdx.x + 1) + r) ;
		int yLim = blockDim.y*(blockIdx.y + 1) > h ? h : (blockDim.y*(blockIdx.y + 1) + r);
		int xStep = blockDim.x*(blockIdx.x + 1) > w ? w%blockDim.x : blockDim.x ;
		int yStep = blockDim.y*(blockIdx.y + 1) > h ? h%blockDim.y : blockDim.y ;
		
		//Collaborative loading into shared memory
		for( i = y-r, iSh = threadIdx.y ; i< yLim ; iSh+=yStep , i +=yStep ){
			for( j = x-r, jSh = threadIdx.x ; j < xLim ; jSh+=xStep , j +=xStep){
				
				idxN = i<0? 0 : (i>=h ? h-1 : i );
				idxS = j<0? 0 : (j>=w ? w-1 : j );
	
				picBlock[(iSh*shDim_x+jSh)] = d_imageArray[(idxN*w+idxS)*3];
				picBlock[(iSh*shDim_x+jSh)+offset] = d_imageArray[(idxN*w+idxS)*3+1];
				picBlock[(iSh*shDim_x+jSh)+offset*2] = d_imageArray[(idxN*w+idxS)*3+2];			
			}
		}
		
	}
	__syncthreads();		//Make sure every thread has loaded all its portions.
	
	/*
	* All the subblocks are now in shared memory. Now we blurr the image.
	*/

	
	for( i = 0; i <= r; i++){
		//Kernel is symetrix along x and y axis.
		idxN = idxS = ((threadIdx.y+r+i)*shDim_x + (threadIdx.x+r));
		idxW = idxE = ((threadIdx.y+r-i)*shDim_x + (threadIdx.x+r));
		iSh = (i+r)*d+r;
		
		//Loop Unrolling 2 times.
		for( j = 0; j <= r-1 ; j+=2){
				
			Gaus_val = d_const_Gaussian[ iSh ];
			tempR += (picBlock[idxN]+picBlock[idxS] + picBlock[idxE]+picBlock[idxW])*Gaus_val;
			tempG += (picBlock[idxN+offset]+picBlock[idxS+offset]+picBlock[idxE+offset]+picBlock[idxW+offset])*Gaus_val;
			tempB += (picBlock[idxN+offset*2]+picBlock[idxS+offset*2]+picBlock[idxE+offset*2]+picBlock[idxW+offset*2])*Gaus_val;
			
			idxS++;	idxN--;	idxE++;	idxW--; iSh++;
			
			Gaus_val = d_const_Gaussian[ iSh ];
			tempR += (picBlock[idxN]+picBlock[idxS] + picBlock[idxE]+picBlock[idxW])*Gaus_val;
			tempG += (picBlock[idxN+offset]+picBlock[idxS+offset]+picBlock[idxE+offset]+picBlock[idxW+offset])*Gaus_val;
			tempB += (picBlock[idxN+offset*2]+picBlock[idxS+offset*2]+picBlock[idxE+offset*2]+picBlock[idxW+offset*2])*Gaus_val;
			
			idxS++;	idxN--;	idxE++;	idxW--; iSh++;

		}
		//Complete the unrolled portion
		for(  ; j <= r ; j++){
			Gaus_val = d_const_Gaussian[ iSh ];
			tempR += (picBlock[idxN]+picBlock[idxS] + picBlock[idxE]+picBlock[idxW])*Gaus_val;
			tempG += (picBlock[idxN+offset]+picBlock[idxS+offset]+picBlock[idxE+offset]+picBlock[idxW+offset])*Gaus_val;
			tempB += (picBlock[idxN+offset*2]+picBlock[idxS+offset*2]+picBlock[idxE+offset*2]+picBlock[idxW+offset*2])*Gaus_val;
			
			idxS++;	idxN--;	idxE++;	idxW--; iSh++;

		}
	}
	
	//store the blurrred image.
	idx = ((y * w) + x)*3;
	d_imageArrayResult[idx++] = tempR;
	d_imageArrayResult[idx++] = tempG;
	d_imageArrayResult[idx++] = tempB;
	
}

void blurr_CPU( float *h_imageArray, float *h_imageArrayResult, int w, int h, int r, float * h_Gaussian  ){
	// for every pixel in p, get it's Rgba structure, and convert the
    // red/green/blue values there to luminance L, effectively converting
    // it to greyscale:
	int d = 2*r+1;
    for (int y = (0+r); y < (h-r); ++y) {
        for (int x = (0+r); x < (w-r); ++x) {
		
			float tempR = 0.0f;
			float tempG = 0.0f;
			float tempB = 0.0f;
			
			for(int i = -r; i <= r; i++){
				for(int j = -r; j <= r ; j++){
				
					unsigned int idx = (((y+i) * w) + (x+j)) * 3;
					float Gaus_val = h_Gaussian[(i+r)*d+r];
					tempR += h_imageArray[idx]*Gaus_val;
					tempG += h_imageArray[idx+1]*Gaus_val;
					tempB += h_imageArray[idx+2]*Gaus_val;
					
				}
			}
            unsigned int idx = (((y) * w) + (x)) * 3;
            h_imageArrayResult[idx] = tempR;
            h_imageArrayResult[idx+1] = tempG;
            h_imageArrayResult[idx+2] = tempB;

       }
    }
}

int main (int argc, char *argv[])
{
    struct timeval t0, t1;
    int w, h;   // the width & height of the image, used frequently
    int r = atoi( argv[2] );		//Pray that the input radius is a single digit


	cudaDeviceProp dev_prop_curr;
	cudaGetDeviceProperties( &dev_prop_curr , 0);
	//printDevProp( dev_prop_curr );
        
   	/*
 	*Start by calculating the Gaussian Kernel
 	*/
 	int d = 2*r+1;
 	float h_Gaussian[d][d];
	float sigma = ((float)r/3.0);
	printf("sigma %10.4f\n",sigma);
	float preFactor = 1/(2*M_PI*sigma*sigma);
	float normalization = 0.0f;
	
	/* 
	* Generate a Gaussian matrix
	*/
	/*
	* non-normalized Gausian
	*/
	for(int i = -r; i <= r; i++){
 		for(int j = -r; j <= r ; j++){
			float tempGauss = preFactor* exp( -(i*i+j*j)/(2*sigma*sigma) );
			h_Gaussian[i+r][j+r]= tempGauss;
			normalization += tempGauss;
 		}
 	}
	/*
	* normalized Gaussian
	*/
	for(int i = -r; i <= r; i++){
 		for(int j = -r; j <= r ; j++){
			h_Gaussian[i+r][j+r] /= normalization;
			if(i==0){
				h_Gaussian[i+r][j+r]/=2.0;
			}
			if(j==0){
				h_Gaussian[i+r][j+r]/=2.0;
			}
			//printf("%e\t",h_Gaussian[i+r][j+r]);
 		}
		//printf("\n");
 	}
 	

    //
    // GPU version: it will save whatever is in h_imageArrayResult
    // to the file "hw1b.exr"
    //
    float *h_imageArray;
    readOpenEXRFile (argv[1], &h_imageArray, w, h);
    float * h_imageArrayResult = (float*)malloc(sizeof(float)*3*w*h);
    
    gettimeofday(&t0,0);
    
    //
    // process it on the GPU: 1) copy it to device memory, 2) process
    // it with a 2d grid of 2d blocks, with each thread assigned to a 
    // pixel. then 3) copy it back.
    // 
    float * d_test;
    float BLOCK_X = 16.0;
	float BLOCK_Y = 16.0;
	
	//For Nvidia cluster.
	if( dev_prop_curr.maxThreadsPerBlock >= 1024 ){ 
		BLOCK_X = 32.0;	
		BLOCK_Y = 32.0;
	}
	size_t sharedBlockSZ = 3*(BLOCK_X+2*r) * (BLOCK_Y+2*r) * sizeof(float);	//Picture blocks
	if(sharedBlockSZ > dev_prop_curr.sharedMemPerBlock){
		printf("Shared Memory exceeded allocated size per block: %lu Max %lu\n",sharedBlockSZ, dev_prop_curr.sharedMemPerBlock);
		return -1;
	}
	dim3 numThreads( BLOCK_X, BLOCK_Y,1);
	dim3 numBlocks( ceil(w/BLOCK_X), ceil(h/BLOCK_Y),1);
    float * d_imageArray;
    float * d_imageArrayResult;

    
    GPU_CHECKERROR( cudaMalloc((void **)&d_imageArray, sizeof(float)*w*h*3) );
    GPU_CHECKERROR( cudaMalloc((void **)&d_imageArrayResult, sizeof(float)*w*h*3) );
    GPU_CHECKERROR( cudaMalloc((void **)&d_test, sizeof(float)*D_MAX*D_MAX) );
    
    GPU_CHECKERROR( cudaMemcpyToSymbol( 
    							d_const_Gaussian, 
    							&h_Gaussian[0][0], 
    							sizeof(float)*d*d,
    							0,
    							cudaMemcpyHostToDevice));
															
	GPU_CHECKERROR( cudaMemcpy(	d_imageArray, 
								h_imageArray, 
								sizeof(float)*w*h*3, 
								cudaMemcpyHostToDevice ) );						
	//
	// Your memory copy, & kernel launch code goes here:
	//
	printf("Launching one kernel\n");
	blurr_GPU<<< numBlocks, numThreads , sharedBlockSZ>>>( d_imageArray,d_imageArrayResult,d_test,w,h,r);
	GPU_CHECKERROR( cudaGetLastError() );
	
    //}
    GPU_CHECKERROR( cudaDeviceSynchronize() );		
	//
	//Fetch the results
	//
    GPU_CHECKERROR( cudaMemcpy( h_imageArrayResult, 
								d_imageArrayResult, 
								sizeof(float)*w*h*3, 
								cudaMemcpyDeviceToHost ) );
					
    gettimeofday(&t1,0);
    float timdiff = (1000000.0*(t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec)) / 1000.0;
    printf ("\ndone: time taken for parallel version is %10.1f ms\n", timdiff);
    printf("writing output image hw1b.exr\n");
    writeOpenEXRFile ("hw1b.exr", h_imageArrayResult, w, h);
    
    
    free (h_imageArray);
	free (h_imageArrayResult);
	GPU_CHECKERROR( cudaFree(d_imageArray) );
    GPU_CHECKERROR( cudaFree(d_imageArrayResult));
	GPU_CHECKERROR( cudaFree(d_test) );
	
    printf("done.\n");

    return 0;
}


