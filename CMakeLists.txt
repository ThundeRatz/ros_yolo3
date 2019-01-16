cmake_minimum_required(VERSION 2.8.3)
project(yolo3)

option(CUDNN "Enable CuDNN" OFF)
option(FORTIFY "Fortify GCC compilation" OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_FLAGS_RELEASE "-Ofast -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG")
set(CUDA_NVCC_FLAGS_RELEASE --use_fast_math)
add_compile_options(-Wall -Wextra)
add_definitions(-DGPU=1)
if (${CUDNN})
  add_definitions(-DCUDNN=1)
endif()

if (FORTIFY)
  add_compile_options(-fstack-protector-all -Wstack-protector --param ssp-buffer-size=4 -D_FORTIFY_SOURCE=2 -O2 -g)
endif()

find_package(catkin REQUIRED COMPONENTS
  image_transport message_generation nodelet roscpp roslib roslint sensor_msgs std_msgs
)
find_package(CUDA REQUIRED)

roslint_cpp(include/darknet/yolo3.h src/darknet/yolo3.cpp src/yolo3_node.cpp src/yolo3_nodelet.cpp)
roslint_add_test()

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependencies might have been
##     pulled in transitively but can be declared for certainty nonetheless:
##     * add a build_depend tag for "message_generation"
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

add_message_files(
  FILES
  Detection.msg
  ImageDetections.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  CATKIN_DEPENDS message_runtime
)

include_directories(
  SYSTEM ${catkin_INCLUDE_DIRS}
  SYSTEM ${CUDA_INCLUDE_DIRS}
  include
  darknet/include
  darknet/src
)

cuda_add_library(yolo3
  darknet/src/activation_kernels.cu
  darknet/src/activation_layer.c
  darknet/src/activations.c
  darknet/src/avgpool_layer.c
  darknet/src/avgpool_layer_kernels.cu
  darknet/src/batchnorm_layer.c
  darknet/src/blas_kernels.cu
  darknet/src/box.c
  darknet/src/convolutional_kernels.cu
  darknet/src/convolutional_layer.c
  darknet/src/cuda.c
  darknet/src/gemm.c
  darknet/src/im2col.c
  darknet/src/im2col_kernels.cu
  darknet/src/image.c
  darknet/src/layer.c
  darknet/src/list.c
  darknet/src/matrix.c
  darknet/src/maxpool_layer.c
  darknet/src/maxpool_layer_kernels.cu
  darknet/src/network.c
  darknet/src/option_list.c
  darknet/src/parser.c
  darknet/src/utils.c
  darknet/src/yolo_layer.c
)
cuda_add_cublas_to_target(yolo3)

target_link_libraries(yolo3
  ${CUDA_LIBRARIES}
  ${CUDA_curand_LIBRARY}
)
if (${CUDNN})
  target_link_libraries(yolo3
    cudnn
  )
endif()

add_library(yolo3_nodelet
  src/yolo3_nodelet.cpp
  src/darknet/yolo3.cpp
)
add_dependencies(yolo3_nodelet yolo3_generate_messages_cpp)

target_link_libraries(yolo3_nodelet
  yolo3
  ${catkin_LIBRARIES}
)
