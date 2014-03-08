#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <sys/time.h>


// STUDENTS: be sure to set the single define at the top of this file, 
// depending on which machines you are running on.
#include "im1.h"

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
__global__ void blurr_GPU(float * d_imageArray, float * Gaussian, int w,int h, int r,float * test){
	
	
	int d = 2*r + 1;
	int x = blockDim.x*blockIdx.x + threadIdx.x; 
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	
	
	//Bounds check
	if( x>=w || y>=h )
		return;
	/*
	if( x!=0|| y!=0 )
		return;
		
	int k = 0;
	for(int i = -r; i <= r; i++){
			for(int j = -r; j <= r ; j++, k++){
				test[k] = Gaussian[(i+r)*d+(j+r)];
		}
	}

	return;
	*/
	
	unsigned int idx;
	float tempR = 0.0;
	//float tempG = 0.0;
	//float tempB = 0.0;
	float Gaus_val = 0.0;
	
	//Pixels that do not require bounds checks
	if(((x+r)<w) && ((x-r)>=0)  && ((y+r)<h) && ((y-r)>=0) ){
		for(int i = -r; i <= r; i++){
			for(int j = -r; j <= r ; j++){
			
				idx = (((y+i) * w) + (x+j));
				//Gaus_val = Gaussian[(i+r)*d+(j+r)];
				Gaus_val = test[(i+r)*d+(j+r)];
				tempR += (d_imageArray[idx]*Gaus_val) ;
				//tempG += (d_imageArray[idx+1]*Gaus_val) ;
				//tempB += (d_imageArray[idx+2]*Gaus_val) ;
			}
		}
	}
	
	idx = ((y * w) + x) ;
    d_imageArray[idx] = tempR;
    //d_imageArray[idx+1] = tempG;
    //d_imageArray[idx+2] = tempB;

	
}


int main (int argc, char *argv[])
{
 
    struct timeval t0, t1;
    int w, h;   // the width & height of the image, used frequently
    int r = argv[2][0]-'0';		//Pray that the input radius is a single digit
    //int r = 5;


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
	float sigma = (float)(2*r);
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

    // for every pixel in p, get it's Rgba structure, and convert the
    // red/green/blue values there to luminance L, effectively converting
    // it to greyscale:

    for (int y = (0+r); y < (h-r); ++y) {
        for (int x = (0+r); x < (w-r); ++x) {
		
			float tempR = 0.0f;
			float tempG = 0.0f;
			float tempB = 0.0f;
			
			for(int i = -r; i <= r; i++){
				for(int j = -r; j <= r ; j++){
				
					unsigned int idx = (((y+i) * w) + (x+j)) * 3;
					float Gaus_val = h_Gaussian[i+r][j+r];
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
    
    float BLOCK_X = 16.0;
	float BLOCK_Y = 16.0;
	dim3 numThreads( BLOCK_X, BLOCK_Y,1);
	dim3 numBlocks( ceil(w/BLOCK_X), ceil(h/BLOCK_Y),1);
    float * d_imageArray;
    
    float h_RGB_planes[3][h][w];
    for(int i = 0; i < h ; i++ ){
    	for(int j=0; j < w ; j++ ){
    		unsigned int idx = ((i * w) + j) * 3;
    		h_RGB_planes[0][i][j] = h_imageArray[idx];
    		h_RGB_planes[1][i][j] = h_imageArray[idx+1];
    		h_RGB_planes[2][i][j] = h_imageArray[idx+2];
    	}
    }
    
    GPU_CHECKERROR( cudaMalloc((void **)&d_imageArray, sizeof(float)*w*h*3) );
    
    float * d_test;
    GPU_CHECKERROR( cudaMalloc((void **)&d_test, sizeof(float)*D_MAX*D_MAX) );
    GPU_CHECKERROR( cudaMemset( d_test, 0, sizeof(float)*D_MAX*D_MAX) );
    
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
    				
	blurr_GPU<<< numBlocks, numThreads >>>( &d_imageArray[m*w*h],d_const_Gaussian,w,h,r, d_test);
	GPU_CHECKERROR( cudaGetLastError() );
	GPU_CHECKERROR( cudaDeviceSynchronize() );
	
    }			
	/*
	*
	*/
	float h_test[D_MAX*D_MAX];
	GPU_CHECKERROR( cudaMemcpy(	h_test, 
								d_test, 
								sizeof(float)*D_MAX*D_MAX, 
								cudaMemcpyDeviceToHost ) );
								
	for(int i =0; i < D_MAX * D_MAX && i < d*d ; i++){
		if(!(i%d))
			printf("\n");
		printf(  "%e ",h_test[i]);
	}
	//
	//Fetch the results
	//
	
    GPU_CHECKERROR( cudaMemcpy( &h_RGB_planes[0][0][0], 
								d_imageArray, 
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
    					
    // All your work is done. Here we assume that you have copied the 
    // processed image data back, from the device to the host, into the
    // original host array h_imageArray. You can do it some other way,
    // this is just a suggestion
    gettimeofday(&t1,0);
    timdiff2 = (1000000.0*(t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec)) / 1000000.0;
    printf ("\ndone: time taken for parallel version is %3.1f s threads %d\n", timdiff2, 0);
    printf("writing output image hw1_gpu.exr\n");
    writeOpenEXRFile ("hw1_gpu.exr", h_imageArray, w, h);
    
    free (h_imageArray);
	GPU_CHECKERROR( cudaFree(d_imageArray) );
	
    printf("done.\n");

    return 0;
}


