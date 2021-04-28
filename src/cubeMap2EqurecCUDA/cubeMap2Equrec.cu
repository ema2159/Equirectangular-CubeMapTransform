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

__device__ struct cart3D {
    float x;
    float y;
    float z;
    int faceIndex;
};

__device__ struct cart2D {
    float x;
    float y;
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

__device__ cart3D projectX(float theta, float phi, int sign) {
    cart3D result;

    result.x = sign * 0.5;
    result.faceIndex = sign == 1 ? X_POS : X_NEG;
    float rho = result.x / (cos(theta) * sin(phi));
    result.y = rho * sin(theta) * sin(phi);
    result.z = rho * cos(phi);
    return result;
}

__device__ cart3D projectY(float theta, float phi, int sign) {
    cart3D result;

    result.y = sign * 0.5;
    result.faceIndex = sign == 1 ? Y_POS : Y_NEG;
    float rho = result.y / (sin(theta) * sin(phi));
    result.x = rho * cos(theta) * sin(phi);
    result.z = rho * cos(phi);
    return result;
}


__device__ cart3D projectZ(float theta, float phi, int sign) {
    cart3D result;

    result.z = sign * 0.5;
    result.faceIndex = sign == 1 ? Z_POS : Z_NEG;
    float rho = result.z / cos(phi);
    result.x = rho * cos(theta) * sin(phi);
    result.y = rho * sin(theta) * sin(phi);
    return result;
}

__device__ cart2D convertEquirectUVtoUnit2D(float theta, float phi, int squareLength) {
    // Calculate the unit vector
    float x = cos(theta) * sin(phi);
    float y = sin(theta) * sin(phi);
    float z = cos(phi);

    // Find the maximum value in the unit vector
    float maximum = max(abs(x), max(abs(y), abs(z)));
    float xx = x / maximum;
    float yy = y / maximum;
    float zz = z / maximum;

    // Project ray to cube surface
    cart3D equirectUV;
    if (xx == 1 or xx == -1) {
	equirectUV = projectX(theta, phi, xx);
    }
    else if (yy == 1 or yy == -1) {
	equirectUV = projectY(theta, phi, yy);
    }
    else {
	equirectUV = projectZ(theta, phi, zz);
    }

    float2 unit2D = unit3DToUnit2D(equirectUV.x, equirectUV.y, equirectUV.z,
				   equirectUV.faceIndex);

    unit2D.x *= squareLength;
    unit2D.y *= squareLength;

    cart2D result;
    result.x = int(unit2D.x);
    result.y = int(unit2D.y);
    result.faceIndex = equirectUV.faceIndex;

    return result;
}

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

    // 1. Loop through all of the pixels in the output image
    if (dst_y < rows && dst_x < cols) {
        // 2. Get the normalised u,v coordinates for the current pixel
	float U = (float)dst_x / (outputWidth - 1);  // 0..1
	float V = (float)dst_y / (outputHeight - 1);  // No need for 1-... as the image output
	                                              // needs to start from the top anyway.
	// 3. Taking the normalised cartesian coordinates calculate the polar
	// coordinate for the current pixel
        float theta = U * 2 * M_PI;
        float phi = V * M_PI;
        // 4. calculate the 3D cartesian coordinate which has been projected to a cubes face
        cart2D cart = convertEquirectUVtoUnit2D(theta, phi, square_length);

        // 5. use this pixel to extract the colour
	uchar3 val = make_uchar3(0, 0, 0);
	if (cart.faceIndex == X_POS) {
	    val = posX(cart.y, cart.x);
	}
	else if (cart.faceIndex == X_NEG) {
	    val = negX(cart.y, cart.x);
	}
	else if (cart.faceIndex == Y_POS) {
	    val = posY(cart.y, cart.x);
	}
	else if (cart.faceIndex == Y_NEG) {
	    val = negY(cart.y, cart.x);
	}
	else if (cart.faceIndex == Z_POS) {
	    val = posZ(cart.y, cart.x);
	}
	else if (cart.faceIndex == Z_NEG) {
	    val = negZ(cart.y, cart.x);
	}

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

