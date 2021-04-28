#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

__device__ float getTheta(float x, float y) {
  float rtn = 0;
  if (y < 0) {
    rtn = atan2(y, x) * -1;
  } else {
    rtn = M_PI + (M_PI - atan2(y, x));
  }
  return rtn;
}

__global__ void process(const cv::cuda::PtrStep<uchar3> src,
			cv::cuda::PtrStep<uchar3> dst, int rows, int cols,
			int inputHeight, int inputWidth) {

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (dst_y < rows && dst_x < cols) {

	float3 val = make_float3(0, 0, 0);
        float tx = 0.0;
        float ty = 0.0;
        float x = 0.0;
        float y = 0.0;
        float z = 0.0;
	const int sqr = inputWidth / 4.0;

	if (dst_y < sqr + 1) {   // top half
          if (dst_x < sqr + 1) { // top left box [Y+]
            tx = dst_x;
            ty = dst_y;
            x = tx - 0.5 * sqr;
            y = 0.5 * sqr;
            z = ty - 0.5 * sqr;
          } else if (dst_x < 2 * sqr + 1) { // top middle [X+]
            tx = dst_x - sqr;
            ty = dst_y;
            x = 0.5 * sqr;
            y = (tx - 0.5 * sqr) * -1;
            z = ty - 0.5 * sqr;
          }

          else { // top right [Y-]
            tx = dst_x - sqr * 2;
            ty = dst_y;
            x = (tx - 0.5 * sqr) * -1;
            y = -0.5 * sqr;
            z = ty - 0.5 * sqr;
          }
        } else {                 // bottom half
          if (dst_x < sqr + 1) { // bottom left box [X-]

            tx = dst_x;
            ty = dst_y - sqr;
            x = int(-0.5 * sqr);
            y = int(tx - 0.5 * sqr);
            z = int(ty - 0.5 * sqr);
          }

          else if (dst_x < 2 * sqr + 1) { // bottom middle [Z-]

            tx = dst_x - sqr;
            ty = dst_y - sqr;
            x = (ty - 0.5 * sqr) * -1;
            y = (tx - 0.5 * sqr) * -1;
            z = 0.5 * sqr; // was -0.5 might be due to phi being reversed
          }

          else { // bottom right [Z+]

            tx = dst_x - sqr * 2;
            ty = dst_y - sqr;
            x = ty - 0.5 * sqr;
            y = (tx - 0.5 * sqr) * -1;
            z = -0.5 * sqr; // was +0.5 might be due to phi being reversed
          }
        }

        // now find out the polar coordinates
        float rho = sqrt(x * x + y * y + z * z);
        float normTheta =
            getTheta(x, y) / (2 * M_PI); // /(2*M_PI) normalise theta
        float normPhi = (M_PI - acos(z / rho)) / M_PI; // /M_PI normalise phi

        // use this for coordinates
        float iX = normTheta * inputWidth;
        float iY = normPhi * inputHeight;

        // catch possible overflows
        if (iX >= inputWidth) {
          iX = iX - (inputWidth);
        }
        if (iY >= inputHeight) {
          iY = iY - (inputHeight);
        }

	dst(dst_y, dst_x).x = src(int(iY), int(iX)).x;
	dst(dst_y, dst_x).y = src(int(iY), int(iX)).y;
	dst(dst_y, dst_x).z = src(int(iY), int(iX)).z;
    }
}

int divUp(int a, int b) {
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA (cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst) {
    const dim3 block(32, 8);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    process<<<grid, block>>>(src, dst, dst.rows, dst.cols, src.rows, src.cols);
}

