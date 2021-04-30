#include <chrono> // for high_resolution_clock
#include <iostream>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;

void startCUDA(cv::cuda::GpuMat &posY, cv::cuda::GpuMat &posX,
               cv::cuda::GpuMat &negY, cv::cuda::GpuMat &negX,
               cv::cuda::GpuMat &negZ, cv::cuda::GpuMat &posZ,
               cv::cuda::GpuMat &dst);

int main(int argc, char **argv) {
  cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

  cv::cuda::GpuMat d_result, posY, posX, negY, negX, posZ, negZ;

  // Get six cube images inside a given directory
  std::string imgs_path(argv[1]);
  posY.upload(cv::imread(imgs_path + "posy.jpg"));
  posX.upload(cv::imread(imgs_path + "posx.jpg"));
  negY.upload(cv::imread(imgs_path + "negy.jpg"));
  negX.upload(cv::imread(imgs_path + "negx.jpg"));
  negZ.upload(cv::imread(imgs_path + "negz.jpg"));
  posZ.upload(cv::imread(imgs_path + "posz.jpg"));

  // Extract images from cube map from a single file with the following format:
  // 	+----+----+----+
  // 	| Y+ | X+ | Y- |
  // 	+----+----+----+
  // 	| X- | Z- | Z+ |
  // 	+----+----+----+
  // cv::Mat h_img = cv::imread(argv[1]);
  // posY.upload(h_img(cv::Rect(0, 0, h_img.cols / 3, h_img.rows / 2)));
  // posX.upload(
  //     h_img(cv::Rect(h_img.cols / 3, 0, h_img.cols / 3, h_img.rows / 2)));
  // negY.upload(
  //     h_img(cv::Rect(2 * h_img.cols / 3, 0, h_img.cols / 3, h_img.rows / 2)));
  // negX.upload(
  //     h_img(cv::Rect(0, h_img.rows / 2, h_img.cols / 3, h_img.rows / 2)));
  // negZ.upload(h_img(cv::Rect(h_img.cols / 3, h_img.rows / 2, h_img.cols / 3,
  //                            h_img.rows / 2)));
  // posZ.upload(h_img(cv::Rect(2 * h_img.cols / 3, h_img.rows / 2, h_img.cols / 3,
  //                            h_img.rows / 2)));
  // Write individual extracted images cv::imwrite(
  //     "posy.jpg", h_img(cv::Rect(0, 0, h_img.cols / 3, h_img.rows / 2)));
  // cv::imwrite("posx.jpg", h_img(cv::Rect(h_img.cols / 3, 0, h_img.cols / 3,
  //                                        h_img.rows / 2)));
  // cv::imwrite("negy.jpg", h_img(cv::Rect(2 * h_img.cols / 3, 0, h_img.cols / 3,
  //                                        h_img.rows / 2)));
  // cv::imwrite("negx.jpg", h_img(cv::Rect(0, h_img.rows / 2, h_img.cols / 3,
  //                                        h_img.rows / 2)));
  // cv::imwrite("negz.jpg", h_img(cv::Rect(h_img.cols / 3, h_img.rows / 2,
  //                                        h_img.cols / 3, h_img.rows / 2)));
  // cv::imwrite("posz.jpg", h_img(cv::Rect(2 * h_img.cols / 3, h_img.rows / 2,
  //                                        h_img.cols / 3, h_img.rows / 2)));

  const int output_width = posY.rows * 2;
  const int output_height = posY.rows;
  cv::Mat h_result(output_height, output_width, CV_8UC3, cv::Scalar(0, 255, 0));

  auto begin = chrono::high_resolution_clock::now();
  const int iter = 1;

  d_result.upload(h_result);

  for (int i = 0; i < iter; i++) {
    startCUDA(posY, posX, negY, negX, negZ, posZ, d_result);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - begin;

  cv::imshow("Processed Image", d_result);
  d_result.download(h_result);
  std::string output_file(argv[2]);
  cout << output_file << endl;
  cv::imwrite(output_file, h_result);

  cout << "Processing time: " << diff.count() << " s" << endl;
  cout << "Time for 1 iteration: " << diff.count() / iter << " s" << endl;
  cout << "IPS: " << iter / diff.count() << endl;

  cv::waitKey();
  return 0;
}
