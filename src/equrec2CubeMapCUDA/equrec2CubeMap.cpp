#include <chrono> // for high_resolution_clock
#include <iostream>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

void startCUDA(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst);

int main(int argc, char **argv) {
  cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

  cv::Mat h_img = cv::imread(argv[1]);
  const int sqr = h_img.cols / 4.0;
  const int output_width = sqr * 3;
  const int output_height = sqr * 2;
  cv::Mat h_result(output_height, output_width, CV_8UC3, cv::Scalar(0, 255, 0));
  cv::cuda::GpuMat d_img, d_result;

  cv::imshow("Original Image", h_img);

  auto begin = chrono::high_resolution_clock::now();
  const int iter = 1;

  d_img.upload(h_img);
  d_result.upload(h_result);

  for (int i = 0; i < iter; i++) {
    startCUDA(d_img, d_result);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - begin;

  cv::imshow("Processed Image", d_result);
  d_result.download(h_result);
  std::string output_file(argv[2]);
  cv::imwrite(output_file, h_result);

  cout << "Processing time: " << diff.count() << " s" << endl;
  cout << "Time for 1 iteration: " << diff.count() / iter << " s" << endl;
  cout << "IPS: " << iter / diff.count() << endl;

  cv::waitKey();
  return 0;
}
