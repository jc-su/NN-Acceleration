cmake_minimum_required(VERSION 3.17)
project(conv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Debug mode")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_61,code=sm_61;-std=c++11;-g;-lineinfo;-Xcompiler;-ggdb)
else ()
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_61,code=sm_61;-std=c++11;-g;-lineinfo;-Xcompiler;-ggdb)
endif ()

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

find_package(OpenCV PATHS ${CMAKE_CURRENT_SOURCE_DIR}/thrid_party/opencv/lib)
if (${OpenCV_FOUND})
    message(STATUS "Found OpenCV")
    add_definitions(-DOPENCV_DEVELOP)

    include_directories(
            ${OpenCV_INCLUDE_DIRS}
    )

else (${OpenCV_FOUND})
    message(STATUS "Could not support OpenCV")
endif (${OpenCV_FOUND})

link_directories(/proj/ciptmp/zu60fucy/local/cudnn/lib64)

add_executable(conv_benchmark conv_1.cu)
target_link_libraries(conv_benchmark ${OpenCV_LIBS} -lcudnn)


# add_executable(conv_2 conv_2.cu)
# target_link_libraries(conv_2 ${OpenCV_LIBS} -lcudnn)

# add_executable(conv_benchmark conv_3.cu)
# target_link_libraries(conv_benchmark -lcudnn)