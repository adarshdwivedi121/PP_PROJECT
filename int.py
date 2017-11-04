import numpy

import pycuda.autoinit
import copy
import os
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import cv2

mod = SourceModule("""

#include <stdint.h>

//Create Mask for Gaussian Blur
__global__ void createKernel(float *kernel, uint8_t size, int sig){
	int x = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ float k[];

    int i=0;
    for(i = 0; i<size; i++)
        k[x*size + i] = powf(2.71, ((-powf(x-size/2,2)-powf(i-size/2,2)) / (2*powf(sig, 2)))) / (2*3.14*sig*sig);

    __syncthreads();
    if(x == 0){
        int i, j;
        float s = 0;
        for(int i=0; i<size*size; i++)
            s += k[i];

		for(i=0; i<size; i++)
			for(j=0; j<size; j++)
				kernel[i*size + j] = k[i*size + j] / s;
    }
}

//Apply Gaussian Blur to given Grayscale Image
__global__ void gaussianBlur(float *k, uint8_t size, uint8_t *image, uint8_t *res, int width, int height){
	extern __shared__ float kernel[];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if(threadIdx.x < size && threadIdx.y < size)
	    kernel[threadIdx.x*size + threadIdx.y] = k[threadIdx.x*size + threadIdx.y];
	__syncthreads();

	if(row >= height && col >= width)
	    return;
    float val = 0;

    int st_row = row - size/2;
    int st_col = col - size/2;

    for(int i=0; i<size; i++){
        for(int j=0; j<size; j++){
            int cr_row = st_row + i;
            int cr_col = st_col + j;
            if(cr_row >= 0 && cr_row < height && cr_col >= 0 && cr_col < width)
                val += image[cr_row * width + cr_col] * kernel[i * size + j];
        }
    }
    res[row * width + col] = (uint8_t) val;
}

//Apply Sobel Edge Detection to given Image
__global__ void sobelFilter(uint8_t *image, uint8_t *sobVal, float *sobAngle, int width, int height){
    __shared__ uint8_t img[32][32];

    int mask[3][3] =  { { 3,  10,  3},
                        { 0,   0,  0},
                        {-3, -10, -3}
                      };
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;


    if(row >= height && col >= width)
        return;

    if(threadIdx.x < blockDim.x && threadIdx.y < blockDim.y)
        img[threadIdx.y][threadIdx.x] = image[row * width + col];
    __syncthreads();

    int r, c, Rx=0, Ry=0;
    for(int i=-1; i<=1; i++){
        r = threadIdx.y + i;

        for(int j=-1; j<=1; j++){
            c = threadIdx.x + j;

            if(r>=0 && r<blockDim.y && c>=0 && c<blockDim.x){
                Rx += img[r][c] * mask[i+1][j+1];
                Ry += img[r][c] * mask[j+1][i+1];
            }
            else{
                int cr_row = row + i;
                int cr_col = col + j;
                if(cr_row >= 0 && cr_row < height && cr_col >= 0 && cr_col < width){
                    int iVal = image[cr_row * width + cr_col];
                    Rx += iVal * mask[i+1][j+1];
                    Ry += iVal * mask[j+1][i+1];
                }
            }
        }
    }

    float val = sqrt((float) Rx*Rx + Ry*Ry);

    sobVal[row*width+col] =(uint8_t) val;
    sobAngle[row*width+col] = (Rx == 0)?90.0f:atan2((double)Rx, (double)Ry);
}

//Suppress Non-Max Values in Sobel Edge Image. For Canny Edge Detection
__global__ void nonMaxSuppress(uint8_t *sobVal, float *sobAngle, uint8_t *res, int width, int height){
	__shared__ uint8_t img[32][32];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadIdx.x < blockDim.x && threadIdx.y < blockDim.y && row < height && col < width)
        img[threadIdx.y][threadIdx.x] = sobVal[row * width + col];
    __syncthreads();

    if((row<=0 && row>=height-1) && (col<=0 && col>=width-1))
        return;


    float angle = sobAngle[row*width+col];
    uint8_t val = sobVal[row*width+col];

    int r, c, arr[3][3];
    for(int i=-1; i<=1; i++){
        r = threadIdx.y + i;

        for(int j=-1; j<=1; j++){
            c = threadIdx.x + j;

            if(r>=0 && r<blockDim.y && c>=0 && c<blockDim.x)
                arr[j+1][i+1] = img[r][c];
        }
    }


    //Horizontal Edge
    if (((-22.5 < angle) && (angle <= 22.5)) || ((157.5 < angle) && (angle <= -157.5)))
        if ((arr[1][1] < arr[1][2]) || (arr[1][1] < arr[1][0]))
            val = 0;

    //Vertical Edge
    if (((-112.5 < angle) && (angle <= -67.5)) || ((67.5 < angle) && (angle <= 112.5)))
        if ((arr[1][1] < arr[2][1]) || (arr[1][1] < arr[0][1]))
            val = 0;

    //-45 Degree Edge
    if (((-67.5 < angle) && (angle <= -22.5)) || ((112.5 < angle) && (angle <= 157.5)))
        if ((arr[1][1] < arr[0][2]) || (arr[1][1] < arr[2][0]))
            val = 0;


    //45 Degree Edge
    if (((-157.5 < angle) && (angle <= -112.5)) || ((22.5 < angle) && (angle <= 67.5)))
        if ((arr[1][1] < arr[2][2]) || (arr[1][1] < arr[0][0]))
            val = 0;

    res[row* width + col] = val;
}

//Apply thresholds to segregate Noise and Non-Essential Pixels. For Canny Edge Detection
__global__ void threshold(uint8_t *image, uint8_t *edgeImage, int width, int height, int low, int high){
	__shared__ int img[32][32];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadIdx.x < blockDim.x && threadIdx.y < blockDim.y && row < height && col < width)
        img[threadIdx.y][threadIdx.x] = image[row * width + col];
    __syncthreads();

    if(row<=0 && row>=height-1 && col<=0 && col>=width-1)
        return;

    uint8_t val = image[row*width+col];

    if(val >= high)
        edgeImage[row*width+col] = 255;
    else if(val <= low)	edgeImage[row*width+col] = 0;
    else if (low<val && val<high){
        bool anyHigh = false;
        bool anyBetween = false;

        int r, c;
        for(int i=-1; i<=1; i++){
            r = threadIdx.y + i;

            for(int j=-1; j<=1; j++){
                c = threadIdx.x + j;

                uint8_t temp = 0;
                if(r>=0 && r<blockDim.y && c>=0 && c<blockDim.x)
                    temp = img[r][c];

                if(temp > val){
                    edgeImage[row*width+col] = 255;
                    anyHigh = true;
                    break;
                }
                else if(low<=temp && temp<=high)
                    anyBetween = true;
            }
            if(anyHigh)
                break;
        }
        if(!anyHigh && anyBetween)
            for(int i=-2; i<=2; i++){
                r = threadIdx.y + i;

                for(int j=-2; j<=2; j++){
                    c = threadIdx.x + j;

                    if((row+i<0 || height<=row+i) || (col+j<0 || width<=col+j))
                        continue;

                    uint8_t temp = 0;
                    if(0<=r && r<blockDim.y && 0<=c && c<blockDim.x)
                        temp = img[r][c];
                    else
                        temp = image[(row+i) * width + (col+j)];
                    if(temp > val){
                        edgeImage[row*width+col] = 255;
                        anyHigh = true;
                        break;
                    }
                    else if(low<=temp && temp<=high)
                        anyBetween = true;
                }
                if(anyHigh)
                    break;
            }

        if(!anyHigh)
            edgeImage[row*width+col] = 0;
    }
}

""")

# Kernel Function Declaration
createKernel = mod.get_function("createKernel")
gaussianBlur = mod.get_function("gaussianBlur")
sobelFilter = mod.get_function("sobelFilter")
nonMaxSuppress = mod.get_function("nonMaxSuppress")
threshold = mod.get_function("threshold")

# Gaussian Filter Create
size = numpy.uint8(5)
sig = numpy.int32(2)
s = numpy.int32(0)
k = numpy.zeros(size * size, dtype=numpy.float32)
kernel = cuda.mem_alloc_like(k)
createKernel(kernel, size, sig, grid=(1, 1), block=(int(size), 1, 1), shared=int(size * size))

# Gaussian Blur
colorImg = cv2.imread('Original.jpg')
img = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY)
blur = numpy.zeros(img.shape, dtype=numpy.uint8)
width, height = img.shape
i = copy.deepcopy(img)
d_i = cuda.mem_alloc_like(i)
d_res = cuda.mem_alloc_like(i)
w = numpy.int32(width)
h = numpy.int32(height)
cuda.memcpy_htod(d_i, i)
gaussianBlur(kernel, size, d_i, d_res, h, w, grid=(int(numpy.ceil(height / 32.0)), int(numpy.ceil(width / 32.0))),
             block=(32, 32, 1), shared=int(size * size))
cuda.memcpy_dtoh(blur, d_res)

# Sobel Filter
h_sobVal = numpy.zeros(img.shape, dtype=numpy.uint8)
h_sobAng = numpy.zeros(img.shape, dtype=numpy.float32)
d_sobVal = cuda.mem_alloc_like(h_sobVal)
d_sobAng = cuda.mem_alloc_like(h_sobAng)
sobelFilter(d_res, d_sobVal, d_sobAng, h, w, grid=(int(numpy.ceil(height / 32.0)), int(numpy.ceil(width / 32.0))),
            block=(32, 32, 1))
cuda.memcpy_dtoh(h_sobVal, d_sobVal)

# Canny Edge Detector
low = numpy.int32(150)
high = numpy.int32(255)
h_canny = numpy.zeros(img.shape, dtype=numpy.uint8)
d_nonMax = cuda.mem_alloc_like(h_canny)
d_res = cuda.mem_alloc_like(h_canny)
nonMaxSuppress(d_sobVal, d_sobAng, d_nonMax, h, w, grid=(int(numpy.ceil(height / 32.0)), int(numpy.ceil(width / 32.0))),
               block=(32, 32, 1))
threshold(d_nonMax, d_res, h, w, low, high, grid=(int(numpy.ceil(height / 32.0)), int(numpy.ceil(width / 32.0))),
          block=(32, 32, 1))
cuda.memcpy_dtoh(h_canny, d_res)

# Hough Lines Transform
cv2.imwrite('temp.jpg', h_canny)
h_canny = cv2.imread('temp.jpg', cv2.CV_8UC1)
os.remove('temp.jpg')

minLineLength = 100
maxLineGap = 10

lines = cv2.HoughLinesP(h_canny, 5, numpy.pi / 180, 100, minLineLength=minLineLength, maxLineGap=maxLineGap)
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(colorImg, (x1, y1), (x2, y2), (0, 255, 0), 2)


cv2.imshow('Image', colorImg)
cv2.imshow('Grayscale', img)
cv2.imshow('Blur', blur)
cv2.imshow('Sobel', h_sobVal)
cv2.imshow('Canny', h_canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
