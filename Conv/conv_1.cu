#include "cudnn_cnn_infer.h"
#include <chrono>
#include <cudnn.h>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

// Check cudnn
#define checkCUDNN(expression)                                     \
    {                                                              \
        cudnnStatus_t status = (expression);                       \
        if (status != CUDNN_STATUS_SUCCESS) {                      \
            std::cerr << "Error on line " << __LINE__ << ": "      \
                      << cudnnGetErrorString(status) << std::endl; \
            std::exit(EXIT_FAILURE);                               \
        }                                                          \
    }


// Load Image function
cv::Mat load_image(const char *image_path) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    image.convertTo(image, CV_32FC3);
    cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
    return image;
}

// Save Image function
void save_image(const char *output_filename, float *buffer, int height,
                int width) {
    cv::Mat output_image(height, width, CV_32FC3, buffer);
    // Make negative values zero.
    cv::threshold(output_image, output_image, 0, 0, cv::THRESH_TOZERO);
    cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
    output_image.convertTo(output_image, CV_8UC3);

    cv::imwrite(output_filename, output_image);
}

int main(int argc, char *argv[]) {
    // Init cudnn
    cudaDeviceReset();

    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    char *outputfile = (char *) "cudnn_out.png";
    // Check input image name
    if (argc < 2) {
        std::cout << "No file input" << std::endl;
        return 0;
    }
    //
    // Check if the filename is valid
    char *filename = argv[1];
    // std::cout << argv[1] << " ";
    // Load Image
    cv::Mat image;
    image = load_image(filename);
    // cv::Mat image = cv::dnn::blobFromImage(t, 1.0f, cv::Size(256,256), cv::Scalar(0,0,0));

    if (image.empty()) {
        std::cout << "File not exist" << std::endl;
        return 0;
    }
    auto timeStamp0 = std::chrono::high_resolution_clock::now();

    // Input Descriptor
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor))
            checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                                  /*format=*/CUDNN_TENSOR_NHWC,
                                                  /*dataType=*/CUDNN_DATA_FLOAT,
                                                  /*batch_size=*/1,
                                                  /*channels=*/3,
                                                  /*image_height=*/image.rows,
                                                  /*image_width=*/image.cols))

                    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor))
            checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                                  /*format=*/CUDNN_TENSOR_NHWC,
                                                  /*dataType=*/CUDNN_DATA_FLOAT,
                                                  /*batch_size=*/1,
                                                  /*channels=*/3,
                                                  /*image_height=*/image.rows,
                                                  /*image_width=*/image.cols))

                    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor))
            checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                                  /*dataType=*/CUDNN_DATA_FLOAT,
                                                  /*format=*/CUDNN_TENSOR_NCHW,
                                                  /*out_channels=*/3,
                                                  /*in_channels=*/3,
                                                  /*kernel_height=*/3,
                                                  /*kernel_width=*/3))

                    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                               /*pad_height=*/1,
                                               /*pad_width=*/1,
                                               /*vertical_stride=*/1,
                                               /*horizontal_stride=*/1,
                                               /*dilation_height=*/1,
                                               /*dilation_width=*/1,
                                               /*mode=*/CUDNN_CROSS_CORRELATION,
                                               /*computeType=*/CUDNN_DATA_FLOAT));

    cudnnConvolutionFwdAlgo_t convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
    //  checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
    //      cudnn, input_descriptor, kernel_descriptor, convolution_descriptor,
    //      output_descriptor, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
    //      /*memoryLimitInBytes=*/0, &convolution_algorithm));
    // convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING;
    // std::cout << "Cudnn Forward Algorithm Index :" << convolution_algorithm << std::endl;
    size_t workspace_bytes;

    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
            cudnn, input_descriptor, kernel_descriptor, convolution_descriptor,
            output_descriptor, convolution_algorithm, &workspace_bytes))
            // std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
            //           << std::endl;

            void *d_workspace;
    cudaMalloc(&d_workspace, workspace_bytes);
    // std::cout << "allocate workspace" << std::endl;
    int batch_size;
    int channels;
    int height;
    int width;

    cudnnGetConvolution2dForwardOutputDim(
            convolution_descriptor, input_descriptor, kernel_descriptor, &batch_size,
            &channels, &height, &width);

    int image_bytes = batch_size * channels * height * width * sizeof(float);

    float *d_input;
    cudaMalloc(&d_input, image_bytes);
    cudaMemcpy(d_input, image.ptr<float>(0), image_bytes, cudaMemcpyHostToDevice);

    float *d_output;
    cudaMalloc(&d_output, image_bytes);
    cudaMemset(d_output, 0, image_bytes);

    // std::cout << "Height and width:" << height << " x " << width << std::endl;
    std::cout << height << " "
              << width << " ";
    // Mystery kernel
    const float kernel_template[3][3] = {{1, 1, 1},
                                         {1, -8, 1},
                                         {1, 1, 1}};
    //const float kernel_template[3][3] = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};

    float h_kernel[3][3][3][3];
    for (auto &kernel: h_kernel) {
        for (auto &channel: kernel) {
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    channel[row][column] = kernel_template[row][column];
                }
            }
        }
    }

    float *d_kernel;
    cudaMalloc(&d_kernel, sizeof(h_kernel));
    cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);
    const float alpha = 1, beta = 0;

    // std::cout << "Start conv" << std::endl;
    auto timeStampA = std::chrono::high_resolution_clock::now();
    checkCUDNN(cudnnConvolutionForward(
            cudnn, &alpha, input_descriptor, d_input, kernel_descriptor, d_kernel,
            convolution_descriptor, convolution_algorithm, d_workspace,
            workspace_bytes, &beta, output_descriptor, d_output))

            cudaDeviceSynchronize();
    auto timeStampB = std::chrono::high_resolution_clock::now();
    auto *h_output = new float[image_bytes];
    cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost);
    auto timeStamp1 = std::chrono::high_resolution_clock::now();

    auto total_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(timeStamp1 - timeStamp0).count();

    auto conv_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(timeStampB - timeStampA).count();

    // Print result
    // std::cout << "Total process time: " << total_duration
    //           << std::endl;
    // std::cout << "Total convolution time: " << conv_duration
    //           << std::endl;
    // std::cout << "Save Output to " << outputfile << std::endl;
    save_image(outputfile, h_output, height, width);
    std::cout << conv_duration << "\n";
    // Delete
    delete[] h_output;
    cudaFree(d_kernel);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_workspace);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

    cudnnDestroy(cudnn);
}
