#include <iostream>
#include <cmath>
#include <png.h>
#include "utils/pngio.h"
#include "cuda_runtime.h"

/*
    Implementation based on function z = x^2 + y^2
*/

// CUDA error checking macro
#define CUDA_CHECK_RETURN(value) {                            \
    cudaError_t err = value;                                    \
    if (err != cudaSuccess) {                                  \
        fprintf(stderr, "Error %s at line %d in file %s\n",    \
                cudaGetErrorString(err), __LINE__, __FILE__);  \
        exit(1);                                                \
    } }

// Namespace for constants
namespace MatrixConstants {
    const unsigned int width = 1024;
    const unsigned int height = 1024;
    const float min = -2.0f;
    const float max = 2.0f;
    const float step = 0.01f;
}

// Device constant variables
__constant__ unsigned int d_width;
__constant__ unsigned int d_height;
__constant__ float d_min;
__constant__ float d_max;
__constant__ float d_step;

// First kernel for matrix generation
__global__ void first_kernel_matrix_generation(float *matrix) {
    __shared__ float sharedMatrix[32][32];  // assuming the 32x32 block size

    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int index = threadIdx.x + threadIdx.y * blockDim.x;

    // Check if thread is within matrix bounds
    if (x < d_width && y < d_height) {
        // Shared memory computation
        sharedMatrix[threadIdx.x][threadIdx.y] = (float)(x * x + y * y);
        __syncthreads();

        // Calculate matrix index and store result
        unsigned int matrixIndex = x + y * d_width;
        matrix[matrixIndex] = d_min + (sharedMatrix[index % 32][index / 32] / (d_width * d_width + d_height * d_height)) * (d_max - d_min);
    }
}

// Second kernel for image generation
__global__ void second_kernel_image_generation(float *matrix, unsigned char *img) {
    __shared__ float sharedMatrix[32][32];  // assuming 32x32 block size
    __shared__ unsigned char sharedImg[32][32][3];  // assuming 32x32 block size

    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int index = threadIdx.x + threadIdx.y * blockDim.x;

    // Check if thread is within matrix bounds
    if (x < d_width && y < d_height) {
        // Shared memory computation
        sharedMatrix[threadIdx.x][threadIdx.y] = matrix[x + y * d_width];
        __syncthreads();

        // Normalize values and perform color mapping
        float normalizedValue = (sharedMatrix[index % 32][index / 32] - d_min) / (d_max - d_min);
        sharedImg[threadIdx.x][threadIdx.y][0] = static_cast<unsigned char>(normalizedValue * 255.0f);
        sharedImg[threadIdx.x][threadIdx.y][1] = static_cast<unsigned char>((1.0f - normalizedValue) * 255.0f);
        sharedImg[threadIdx.x][threadIdx.y][2] = static_cast<unsigned char>(sin(normalizedValue * 3.14f) * 255.0f);
        __syncthreads();

        // Calculate image index and store result
        unsigned int imgIndex = x + y * d_width;
        img[imgIndex * 3] = sharedImg[index % 32][index / 32][0];
        img[imgIndex * 3 + 1] = sharedImg[index % 32][index / 32][1];
        img[imgIndex * 3 + 2] = sharedImg[index % 32][index / 32][2];
    }
}

// Main function
int main() {
    std::cout << "Matrix Generation and Image Conversion..." << std::endl;

    // Copy constants to device constant memory
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_width, &MatrixConstants::width, sizeof(unsigned int)));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_height, &MatrixConstants::height, sizeof(unsigned int)));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_min, &MatrixConstants::min, sizeof(float)));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_max, &MatrixConstants::max, sizeof(float)));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_step, &MatrixConstants::step, sizeof(float)));

    // Allocate device memory for matrix
    unsigned int size = MatrixConstants::width * MatrixConstants::height * sizeof(float);
    float *d_matrix = nullptr;
    CUDA_CHECK_RETURN(cudaMalloc(&d_matrix, size));

    // Set block and grid dimensions
    dim3 block_dim(32, 32);  // Increased to 32x32 block size
    dim3 grid_dim((MatrixConstants::width + block_dim.x - 1) / block_dim.x, (MatrixConstants::height + block_dim.y - 1) / block_dim.y);

    // Launch first kernel for matrix generation
    first_kernel_matrix_generation<<<grid_dim, block_dim>>>(d_matrix);
    cudaDeviceSynchronize();

    // Allocate device memory for image
    size_t imageSize = MatrixConstants::width * MatrixConstants::height * 3 * sizeof(unsigned char);
    unsigned char *d_img = nullptr;
    CUDA_CHECK_RETURN(cudaMalloc(&d_img, imageSize));

    // Launch second kernel for image generation
    second_kernel_image_generation<<<grid_dim, block_dim>>>(d_matrix, d_img);
    cudaDeviceSynchronize();

    // Copy image from device to host
    unsigned char *h_img = new unsigned char[imageSize];
    CUDA_CHECK_RETURN(cudaMemcpy(h_img, d_img, imageSize, cudaMemcpyDeviceToHost));

    // Write image to file
    png::image<png::rgb_pixel> img_out(MatrixConstants::width, MatrixConstants::height);
    for (size_t i = 0; i < MatrixConstants::width; ++i) {
        for (size_t j = 0; j < MatrixConstants::height; ++j) {
            size_t index = (i + j * MatrixConstants::width) * 3;
            img_out[j][i] = png::rgb_pixel(h_img[index], h_img[index + 1], h_img[index + 2]);
        }
    }
    img_out.write("../matrixGeneratedImage.png");

    delete[] h_img;
    cudaFree(d_matrix);
    cudaFree(d_img);
    return 0;
}
