#include <stdio.h>
#include <cuda.h>

// STUDENTS: be sure to set the single define at the top of this file, 
// depending on which machines you are running on.
#include "im1.h"



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
__global__ void BW_GPU(float * d_imageArray, int w,int h){
	int i = blockDim.x*blockIdx.x + threadIdx.x; 
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	
	//Bounds check
	if((i>w) || (j>h))
		return;
		
	unsigned int idx = ((j * w) + i) * 3;
            
    float L = 0.2126f*d_imageArray[idx] + 
              0.7152f*d_imageArray[idx+1] + 
              0.0722f*d_imageArray[idx+2];

    d_imageArray[idx] = L;
    d_imageArray[idx+1] = L;
    d_imageArray[idx+2] = L;
	
}


int main (int argc, char *argv[])
{
 

    printf("reading openEXR file %s\n", argv[1]);
        
    int w, h;   // the width & height of the image, used frequently!


    // First, convert the openEXR file into a form we can use on the CPU
    // and the GPU: a flat array of floats:
    // This makes an array h*w*sizeof(float)*3, with sequential r/g/b indices
    // don't forget to free it at the end


    float *h_imageArray;
    readOpenEXRFile (argv[1], &h_imageArray, w, h);
    printf("reading openEXR file %s Size WxH = %dx%d\n", argv[1],w,h);
        

    // 
    // serial code: saves the image in "hw1_serial.exr"
    //

    // for every pixel in p, get it's Rgba structure, and convert the
    // red/green/blue values there to luminance L, effectively converting
    // it to greyscale:

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            
            unsigned int idx = ((y * w) + x) * 3;
            
            float L = 0.2126f*h_imageArray[idx] + 
                      0.7152f*h_imageArray[idx+1] + 
                      0.0722f*h_imageArray[idx+2];

            h_imageArray[idx] = L;
            h_imageArray[idx+1] = L;
            h_imageArray[idx+2] = L;

       }
    }
    
    printf("writing output image hw1_serial.exr\n");
    writeOpenEXRFile ("hw1_serial.exr", h_imageArray, w, h);
    free(h_imageArray); // make sure you free it: if you use this variable
                        // again, readOpenEXRFile will allocate more memory


    //
    // Now the GPU version: it will save whatever is in h_imageArray
    // to the file "hw1_gpu.exr"
    //
    
    // read the file again - the file read allocates memory for h_imageArray:
    readOpenEXRFile (argv[1], &h_imageArray, w, h);



    // at this point, h_imageArray has sequenial floats for red, green , and
    // blue for each pixel: r,g,b,r,g,b,r,g,b,r,g,b. You need to copy
    // this array to GPU global memory, and have one thread per pixel compute
    // the luminance value, with which you will overwrite each r,g,b, triple.

    //
    // process it on the GPU: 1) copy it to device memory, 2) process
    // it with a 2d grid of 2d blocks, with each thread assigned to a 
    // pixel. then 3) copy it back.
    //
    
    float * d_imageArray;
    GPU_CHECKERROR( cudaMalloc((void **)&d_imageArray, sizeof(float)*w*h*3.0) );
    GPU_CHECKERROR( cudaMemcpy(	d_imageArray, 
    							h_imageArray, 
    							sizeof(float)*w*h*3, 
    							cudaMemcpyHostToDevice ) );

	float BLOCK_X = 32.0;
	float BLOCK_Y = 16.0;
	dim3 numThreads( BLOCK_X, BLOCK_Y,1);
	dim3 numBlocks( ceil(w/BLOCK_X), ceil(h/BLOCK_Y),1);
	
    //
    // Your memory copy, & kernel launch code goes here:
    //

	BW_GPU<<< numBlocks, numThreads >>>( d_imageArray,w,h);
	GPU_CHECKERROR( cudaDeviceSynchronize() );
	
	//
	//Fetch the results
	//
    GPU_CHECKERROR( cudaMemcpy(	h_imageArray, 
								d_imageArray, 
								sizeof(float)*w*h*3, 
								cudaMemcpyDeviceToHost ) );


    // All your work is done. Here we assume that you have copied the 
    // processed image data back, from the device to the host, into the
    // original host array h_imageArray. You can do it some other way,
    // this is just a suggestion
    
    printf("writing output image hw1_gpu.exr\n");
    writeOpenEXRFile ("hw1_gpu.exr", h_imageArray, w, h);
    free (h_imageArray);

    printf("done.\n");

    return 0;
}


