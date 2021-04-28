#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

// Enum for faces' indices
enum bar {
    X_POS, X_NEG, Y_POS, Y_NEG, Z_POS, Z_NEG
};

__device__ struct coords_n_face {
    float x;
    float y;
    float z;
    int faceIndex;
};


__device__ float2 unit3DToUnit2D(float x, float y, float z, int faceIndex) {
    float x2D, y2D;
    
    if(faceIndex == X_POS) { // X+
	x2D = y + 0.5;
	y2D = z + 0.5;
    }
    else if(faceIndex == X_NEG) { // X-
	x2D = (y * -1) + 0.5;
	y2D = z + 0.5;
    }
    else if(faceIndex == Y_POS) { // Y+
	x2D = (x * -1) + 0.5;
	y2D = z + 0.5;
    }
    else if(faceIndex == Y_NEG) { // Y-
	x2D = x + 0.5;
	y2D = z + 0.5;
    }
    else if(faceIndex == Z_POS) { // Z+
	x2D = y + 0.5;
	y2D = (x * -1) + 0.5;
    }
    else { //Z-
	x2D = y + 0.5;
	y2D = x + 0.5;
    }
    // need to do this as image.getPixel takes pixels from the top left corner.

    y2D = 1 - y2D;

    return make_float2(x2D, y2D);
}

__device__ coords_n_face projectX(float theta, float phi, int sign) {
    coords_n_face result;

    result.x = sign * 0.5;
    result.faceIndex = sign == 1 ? X_POS : X_NEG;
    float rho = result.x / (cos(theta) * sin(phi));
    result.y = rho * sin(theta) * sin(phi);
    result.z = rho * cos(phi);
    return result;
}

__device__ coords_n_face projectY(float theta, float phi, int sign) {
    coords_n_face result;

    result.y = sign * 0.5;
    result.faceIndex = sign == 1 ? Y_POS : Y_NEG;
    float rho = result.y / (sin(theta) * sin(phi));
    result.x = rho * cos(theta) * sin(phi);
    result.z = rho * cos(phi);
    return result;
}


__device__ coords_n_face projectZ(float theta, float phi, int sign) {
    coords_n_face result;

    result.z = sign * 0.5;
    result.faceIndex = sign == 1 ? Z_POS : Z_NEG;
    float rho = result.z / cos(phi);
    result.x = rho * cos(theta) * sin(phi);
    result.y = rho * sin(theta) * sin(phi);
    return result;
}

// __device__ uchar3 getColour(x, y, index) {

//     if index == X_POS {
// 	    return posx.getpixel(y, x);
// 	}
//     else if index == X_NEG {
// 	    return negx.getpixel(y, x);
// 	}
//     else if index == Y_POS {
// 	    return posy.getpixel(y, x);
// 	}
//     else if index == Y_NEG {
// 	    return negy.getpixel(y, x);
// 	}
//     else if index == Z_POS {
// 	    return posz.getpixel(y, x);
// 	}
//     else if index == Z_NEG {
// 	    return negz.getpixel(y, x);
// 	}
// }

__global__ void process(const cv::cuda::PtrStep<uchar3> posY,
			const cv::cuda::PtrStep<uchar3> posX,
			const cv::cuda::PtrStep<uchar3> negY,
			const cv::cuda::PtrStep<uchar3> negX,
			const cv::cuda::PtrStep<uchar3> posZ,
			const cv::cuda::PtrStep<uchar3> negZ,
			cv::cuda::PtrStep<uchar3> dst, int rows, int cols,
			int square_length) {

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    const int half_square_len = square_length / 2;

    const int outputWidth = square_length * 2;
    const int outputHeight = square_length * 1;

    if (dst_y < rows && dst_x < cols) {

	float3 val = make_float3(0, 0, 0);
	dst(dst_y, dst_x).x = val.x;
	dst(dst_y, dst_x).y = val.y;
	dst(dst_y, dst_x).z = val.z;
    }
}

int divUp(int a, int b) {
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA (cv::cuda::GpuMat& posY, cv::cuda::GpuMat& posX,
		cv::cuda::GpuMat& negY, cv::cuda::GpuMat& negX,
		cv::cuda::GpuMat& posZ, cv::cuda::GpuMat& negZ,
		cv::cuda::GpuMat& dst) {
    const dim3 block(32, 8);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    process<<<grid, block>>>(posY, posX, negY, negX, posZ, negZ, dst,
			     dst.rows, dst.cols, posX.rows);
}

