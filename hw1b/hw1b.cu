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


//
// your __global__ kernel can go here, if you want:
//
__global__ void blurr_GPU(float * d_imageArray,float * d_imageArrayResult, float * Gaussian, int w,int h, int r,float * test){
	
	int d = 2*r + 1;
	extern __shared__ float picBlock[];
	int x = blockDim.x*blockIdx.x + threadIdx.x; 
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	
	//Bounds check
	if( x>=w || y>=h )
		return;
	
	unsigned int idx;
	float tempColor = 0.0;
	float Gaus_val = 0.0;
	int	shDim_x = blockDim.x + 2*r;
	int i, j;
	int iSh, jSh;
	//Blocks that do not require boundry check
	if( (((blockIdx.x+1)*blockDim.x)<(w-r)) && 
		((blockIdx.x > blockDim.x*(r/blockDim.x)))  && 
		(((blockIdx.y+1)*blockDim.y)<(h-r)) && 
		((blockIdx.y > blockDim.y*(r/blockDim.y)))){	
		
		//Collaborative loading into shared memory
		for( i = y-r, iSh = threadIdx.y ; i< (blockDim.y*(blockIdx.y + 1) + r)  ; i+=blockDim.y , iSh+=blockDim.y ){
			for( j = x-r, jSh = threadIdx.x ; j < (blockDim.x*(blockIdx.x + 1) + r)  ; j+=blockDim.x , jSh+=blockDim.x){
				picBlock[iSh*shDim_x+jSh] = d_imageArray[i*w+j];
			}
		}/*
		__syncthreads();		//Make sure every thread has loaded all its portions.
		
		for( i = -r; i <= r; i++){
				for( j = -r; j <= r ; j++){
					idx = (threadIdx.y+r+i)*shDim_x + (threadIdx.x+r+j);
					//Gaus_val = Gaussian[(i+r)*d+(j+r)];
					Gaus_val = test[(i+r)*d+(j+r)];
					tempColor += picBlock[idx]*Gaus_val ;
			}
		}
		
		idx = ((y * w) + x) ;
		d_imageArrayResult[idx] = tempColor;
  */  	
	}
	//These blocks may access picture elements that are out of bounds
	else{
		//Collaborative loading into shared memory
		for( i = y-r, iSh = threadIdx.y ; i< (blockDim.y*(blockIdx.y + 1) + r)  ; i+=blockDim.y , iSh+=blockDim.y ){
			for( j = x-r, jSh = threadIdx.x ; j < (blockDim.x*(blockIdx.x + 1) + r)  ; j+=blockDim.x , jSh+=blockDim.x){
				int iImg = i;
				int jImg = j;
				if(i<0){
					iImg = 0;
				}
				if( j < 0){
					jImg = 0;
				}
				if(i>=h){
					iImg = h-1;
				}
				if(j>=w){
					jImg = w-1;
				}	
				picBlock[iSh*shDim_x+jSh] = d_imageArray[iImg*w+jImg];
				
			}
		}
/*		
		__syncthreads();		//Make sure every thread has loaded all its portions.
		
		for( i = -r; i <= r; i++){
				for( j = -r; j <= r ; j++){
					idx = (threadIdx.y+r+i)*shDim_x + (threadIdx.x+r+j);
					//Gaus_val = Gaussian[(i+r)*d+(j+r)];
					Gaus_val = test[(i+r)*d+(j+r)];
					tempColor += picBlock[idx]*Gaus_val ;
			}
		}
		
		idx = ((y * w) + x) ;
		d_imageArrayResult[idx] = tempColor;
*/		
	}
		__syncthreads();		//Make sure every thread has loaded all its portions.
		
		for( i = -r; i <= r; i++){
				for( j = -r; j <= r ; j++){
					idx = (threadIdx.y+r+i)*shDim_x + (threadIdx.x+r+j);
					//Gaus_val = Gaussian[(i+r)*d+(j+r)];
					Gaus_val = test[(i+r)*d+(j+r)];
					tempColor += picBlock[idx]*Gaus_val ;
			}
		}
		
		idx = ((y * w) + x) ;
		d_imageArrayResult[idx] = tempColor;

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



    // First, convert the openEXR file into a form we can use on the CPU
    // and the GPU: a flat array of floats:
    // This makes an array h*w*sizeof(float)*3, with sequential r/g/b indices
    // don't forget to free it at the end


    float *h_imageArray;
    float *h_imageArrayResult;
    readOpenEXRFile (argv[1], &h_imageArray, w, h);
    printf("reading openEXR file %s Size WxH = %dx%d Blur radius%d\n", "stillife.exr",w,h,r);
    h_imageArrayResult = (float *)malloc(w*h*3*sizeof(float));
        
   	/*
 	*Start by calculating the Gaussian Kernel
 	*/
 	int d = 2*r+1;
 	float h_Gaussian[d][d];
	float sigma = ((float)r/3.0);
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
			float tempGauss = preFactor*exp( -(i*i+j*j)/(2*sigma) );
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
			printf("%e\t",h_Gaussian[i+r][j+r]);
 		}
		printf("\n");
 	}
 	
 	gettimeofday(&t0,0);
 	
    // 
    // serial code: saves the image in "hw1_serial.exr"
    //
    blurr_CPU( h_imageArray, h_imageArrayResult, w, h, r, &h_Gaussian[0][0]  );
    gettimeofday(&t1,0);
    float timdiff2 = (1000000.0*(t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec)) / 1000000.0;
    printf ("done: time taken for serial version is %3.1f s threads %d\n", timdiff2, 0);
    
    printf("writing output image hw1_serial.exr\n");
    writeOpenEXRFile ("hw1_serial.exr", h_imageArrayResult, w, h);
    free(h_imageArray); // make sure you free it: if you use this variable
                        // again, readOpenEXRFile will allocate more memory
  	free( h_imageArrayResult );
	
    //
    // Now the GPU version: it will save whatever is in h_imageArray
    // to the file "hw1_gpu.exr"
    //
    
    // read the file again - the file read allocates memory for h_imageArray:
    readOpenEXRFile (argv[1], &h_imageArray, w, h);
	gettimeofday(&t0,0);


    // at this point, h_imageArray has sequenial floats for red, green , and
    // blue for each pixel: r,g,b,r,g,b,r,g,b,r,g,b. You need to copy
    // this array to GPU global memory, and have one thread per pixel compute
    // the luminance value, with which you will overwrite each r,g,b, triple.

    //
    // process it on the GPU: 1) copy it to device memory, 2) process
    // it with a 2d grid of 2d blocks, with each thread assigned to a 
    // pixel. then 3) copy it back.
    // 
    float * d_test;
    float BLOCK_X = 16.0;
	float BLOCK_Y = 16.0;
	dim3 numThreads( BLOCK_X, BLOCK_Y,1);
	dim3 numBlocks( ceil(w/BLOCK_X), ceil(h/BLOCK_Y),1);
    float * d_imageArray;
    float * d_imageArrayResult;
    float h_RGB_planes[3][h][w];
    size_t sharedBlockSZ = 3*(BLOCK_X+2*r) * (BLOCK_Y+2*r) * sizeof(float);
	if(sharedBlockSZ > MAX_SHARED_MEM){
		printf("Shared Memory exceeded allocated size per block");
		return -1;
	}
	
    for(int i = 0; i < h ; i++ ){
    	for(int j=0; j < w ; j++ ){
    		unsigned int idx = ((i * w) + j) * 3;
    		h_RGB_planes[0][i][j] = h_imageArray[idx];
    		h_RGB_planes[1][i][j] = h_imageArray[idx+1];
    		h_RGB_planes[2][i][j] = h_imageArray[idx+2];
    	}
    }
    
    GPU_CHECKERROR( cudaMalloc((void **)&d_imageArray, sizeof(float)*w*h*3) );
    GPU_CHECKERROR( cudaMalloc((void **)&d_imageArrayResult, sizeof(float)*w*h*3) );
    GPU_CHECKERROR( cudaMalloc((void **)&d_test, sizeof(float)*D_MAX*D_MAX) );
    
    GPU_CHECKERROR( cudaMemcpy(	d_const_Gaussian, 
    							&h_Gaussian[0][0], 
    							sizeof(float)*d*d,
    							cudaMemcpyHostToDevice));
	
	GPU_CHECKERROR( cudaMemcpy(	d_test, 
    							&h_Gaussian[0][0], 
    							sizeof(float)*d*d,
    							cudaMemcpyHostToDevice));
    							
    for(int m = 0 ; m < 3; m++){
		GPU_CHECKERROR( cudaMemcpy(	&d_imageArray[m*w*h], 
									&h_RGB_planes[m][0][0], 
									sizeof(float)*w*h, 
									cudaMemcpyHostToDevice ) );						
		//
		// Your memory copy, & kernel launch code goes here:
		//
		printf("Launching one kernel\n");
		blurr_GPU<<< numBlocks, numThreads , sharedBlockSZ>>>( &d_imageArray[m*w*h],&d_imageArrayResult[m*w*h],d_const_Gaussian,w,h,r, d_test);
		GPU_CHECKERROR( cudaGetLastError() );
	
    }
    GPU_CHECKERROR( cudaDeviceSynchronize() );		
	/*
	*
	*
	float h_test[D_MAX*D_MAX];
	GPU_CHECKERROR( cudaMemcpy(	h_test, 
								d_test, 
								sizeof(float)*D_MAX*D_MAX, 
								cudaMemcpyDeviceToHost ) );
								
	for(int i =0; i < D_MAX * D_MAX && i < d*d ; i++){
		if(!(i%d))
			printf("\n");
		printf(  "%e ",h_test[i]);
	}*/
	//
	//Fetch the results
	//
	
    GPU_CHECKERROR( cudaMemcpy( &h_RGB_planes[0][0][0], 
								d_imageArrayResult, 
								sizeof(float)*w*h*3, 
								cudaMemcpyDeviceToHost ) );
 	
 	for(int i = 0; i < h ; i++ ){
    	for(int j=0; j < w; j++ ){
    		unsigned int idx = ((i * w) + j) * 3;
    		h_imageArray[idx] = h_RGB_planes[0][i][j];
    		h_imageArray[idx+1] = h_RGB_planes[1][i][j];
    		h_imageArray[idx+2] = h_RGB_planes[2][i][j];
    	}
    }
    					
    gettimeofday(&t1,0);
    timdiff2 = (1000000.0*(t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec)) / 1000000.0;
    printf ("\ndone: time taken for parallel version is %3.1f s threads %d\n", timdiff2, 0);
    printf("writing output image hw1_gpu.exr\n");
    writeOpenEXRFile ("hw1_gpu.exr", h_imageArray, w, h);
    
    free (h_imageArray);
	GPU_CHECKERROR( cudaFree(d_imageArray) );
	GPU_CHECKERROR( cudaFree(d_test) );
	
    printf("done.\n");

    return 0;
}


