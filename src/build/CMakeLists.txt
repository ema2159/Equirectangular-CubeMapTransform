cmake_minimum_required(VERSION 2.8)
project(CUDAOpenCVImageProcessing)

# Use lower GCC version for CUDA
set(CMAKE_C_COMPILER "/usr/bin/gcc-8")
set(CMAKE_C++_COMPILER "/usr/bin/g++-8")

# Include OpenCV
find_package(OpenCV REQUIRED PATHS "/home/ema2159/.local/opencv/opencv-4.2.0/build")
include_directories(${OpenCV_INCLUDE_DIRS})
# Include CUDA
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};)
    # Equirectangular to cube map    
    cuda_compile(equrec2Cube ../equrec2CubeMapCUDA/equrec2CubeMap.cu)
    cuda_add_library(equrec2CubeCUDA ../equrec2CubeMapCUDA/equrec2CubeMap.cu)
    # Cubemap to equirectangular
    cuda_compile(cube2Equrec ../cubeMap2EqurecCUDA/cubeMap2Equrec.cu)
    cuda_add_library(cube2EqurecCUDA ../cubeMap2EqurecCUDA/cubeMap2Equrec.cu)
    # Laplacian filter
    cuda_compile(laplace ../convolutionLaplace/laplaceFilter.cu)
    cuda_add_library(laplaceCUDA ../convolutionLaplace/laplaceFilter.cu)
    # Color segmentation
    cuda_compile(colorSeg ../colorSegmentation/colorSeg.cu)
    cuda_add_library(colorSegCUDA ../colorSegmentation/colorSeg.cu)
    add_definitions(-DGPU_OPENCV_ENABLE)
endif()
  
# Link opencv libs to all executables
link_libraries(${OpenCV_LIBS})

# Create executables
# Equirectangular to Cube Map
add_executable(equrec2CubeMap ../equrec2CubeMap/equrec2CubeMap.cpp)
# Cube Map to Equirectangular
add_executable(cubeMap2Equrec ../cubeMap2Equrec/cubeMap2Equrec.cpp)
# CUDA implementation
add_executable(equrec2CubeMapCUDA ../equrec2CubeMapCUDA/equrec2CubeMap.cpp)
target_link_libraries(equrec2CubeMapCUDA equrec2CubeCUDA)
# Cube Map to Equirectangular
# CUDA implementation
add_executable(cubeMap2EqurecCUDA ../cubeMap2EqurecCUDA/cubeMap2Equrec.cpp)
target_link_libraries(cubeMap2EqurecCUDA cube2EqurecCUDA)
# Laplacian filter
add_executable(laplaceFilter ../convolutionLaplace/laplaceFilter.cpp)
target_link_libraries(laplaceFilter laplaceCUDA)
# Gaussian filter
add_executable(gaussFilter ../convolutionGauss/gaussFilter.cpp)
# Color Segmentation
add_executable(colorSeg ../colorSegmentation/colorSeg.cpp)
target_link_libraries(colorSeg colorSegCUDA)
