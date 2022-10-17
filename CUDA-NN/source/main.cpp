#include "MeasurementSeries.hpp"
#include "argparse.hpp"
#include "common.h"
#include "dtime.hpp"
#include "helper_functions.cuh"
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <string>

#include <cublas.h>
#include <cublas_v2.h>

#include "AvgPoolLayer.hpp"
#include "BatchNormLayer.hpp"
#include "ConvLayer.hpp"
#include "Layer.hpp"
#include "LinearLayer.hpp"
#include "MaxPoolLayer.hpp"
#include "Net.hpp"
#include "ReluLayer.hpp"
#include "Tensor.hpp"

#include "imagenet_map.hpp"

// #include <likwid-marker.h>

int main(int argc, const char **argv) {

    argparse::ArgumentParser parser;
    try {
        parser.addArgument("--input", 1, false);
        parser.addArgument("--weights_dir", 1, false);
        parser.addArgument("--batch_size", 1, false);
        parser.addArgument("--iters", 1, false);
        parser.parse(argc, argv);
    } catch (const std::exception &e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }

    try {
        InitializeCUDA(0);
        cublasHandle_t cublas_handle;
        checkCublasErrors(cublasCreate(&cublas_handle));

        std::vector<float> mean = {0.485, 0.456, 0.406};
        std::vector<float> std = {0.229, 0.224, 0.225};

        int status = 0;
        PPMImage *img;
        img = readPPM(parser.retrieve<std::string>("input").c_str());
        int iters = parser.retrieve<int>("iters");
        int n = parser.retrieve<int>("batch_size"), c = 3, h = 32, w = 32;
        float *float_data_ptr = (float *) malloc(n * c * h * w * sizeof(float));
        uchar *uchar_ptr = (uchar *) img->data;
        float cur_mean;
        float cur_std;
        for (int bi = 0; bi < n; ++bi) {
            for (int i = 0; i < c; ++i) {
                cur_mean = mean[i];
                cur_std = std[i];
                for (int j = 0; j < h * w; ++j) {
                    float_data_ptr[bi * (c * h * w) + i * (h * w) + j] =
                            (((float) uchar_ptr[j * 3 + i]) / 255 - cur_mean) / cur_std;
                }
            }
        }

        std::shared_ptr<Tensor<float>> input(new Tensor<float>({n, c, h, w}));
        input->from_cpu(float_data_ptr);

        std::string w_path = parser.retrieve<std::string>("weights_dir").c_str();

        // stub

        ConvLayer *conv1 =
                new ConvLayer(cublas_handle, w_path + "conv1", 3, 2, false);

        BatchNormLayer *bn1 = new BatchNormLayer(w_path + "bn1");
        ReluLayer *relu1 = new ReluLayer();
        MaxPoolLayer *mp1 = new MaxPoolLayer(3, 2, 1);

        conv1->set_input(input);
        bn1->set_input(conv1->get_output());
        relu1->set_input(bn1->get_output());
        mp1->set_input(relu1->get_output());

        // layer1

        // block1
        ConvLayer *layer1_0_conv1 =
                new ConvLayer(cublas_handle, w_path + "layer1.0.conv1", 1, 1, false);
        BatchNormLayer *layer1_0_bn1 = new BatchNormLayer(w_path + "layer1.0.bn1");
        ReluLayer *layer1_0_relu1 = new ReluLayer();
        ConvLayer *layer1_0_conv2 =
                new ConvLayer(cublas_handle, w_path + "layer1.0.conv2", 1, 1, false);
        BatchNormLayer *layer1_0_bn2 = new BatchNormLayer(w_path + "layer1.0.bn2");
        ReluLayer *layer1_0_relu2 = new ReluLayer();

        layer1_0_conv1->set_input(mp1->get_output());
        layer1_0_bn1->set_input(layer1_0_conv1->get_output());
        layer1_0_relu1->set_input(layer1_0_bn1->get_output());
        layer1_0_conv2->set_input(layer1_0_relu1->get_output());
        layer1_0_bn2->set_input(layer1_0_conv2->get_output());
        layer1_0_relu2->set_input(layer1_0_bn2->get_output());

        // block2
        ConvLayer *layer1_1_conv1 =
                new ConvLayer(cublas_handle, w_path + "layer1.1.conv1", 1, 1, false);
        BatchNormLayer *layer1_1_bn1 = new BatchNormLayer(w_path + "layer1.1.bn1");
        ReluLayer *layer1_1_relu1 = new ReluLayer();
        ConvLayer *layer1_1_conv2 =
                new ConvLayer(cublas_handle, w_path + "layer1.1.conv2", 1, 1, false);
        BatchNormLayer *layer1_1_bn2 = new BatchNormLayer(w_path + "layer1.1.bn2");
        ReluLayer *layer1_1_relu2 = new ReluLayer();

        layer1_1_conv1->set_input(layer1_0_relu2->get_output());
        layer1_1_bn1->set_input(layer1_1_conv1->get_output());
        layer1_1_relu1->set_input(layer1_1_bn1->get_output());
        layer1_1_conv2->set_input(layer1_1_relu1->get_output());
        layer1_1_bn2->set_input(layer1_1_conv2->get_output());
        layer1_1_relu2->set_input(layer1_1_bn2->get_output());

        // layer2

        // block1
        ConvLayer *layer2_0_conv1 =
                new ConvLayer(cublas_handle, w_path + "layer2.0.conv1", 1, 2, false);
        BatchNormLayer *layer2_0_bn1 = new BatchNormLayer(w_path + "layer2.0.bn1");
        ReluLayer *layer2_0_relu1 = new ReluLayer();
        ConvLayer *layer2_0_conv2 =
                new ConvLayer(cublas_handle, w_path + "layer2.0.conv2", 1, 1, false);
        BatchNormLayer *layer2_0_bn2 = new BatchNormLayer(w_path + "layer2.0.bn2");
        ReluLayer *layer2_0_relu2 = new ReluLayer();

        // downsample
        ConvLayer *layer2_0_downsample_0_conv = new ConvLayer(
                cublas_handle, w_path + "layer2.0.downsample.0", 0, 2, false);
        BatchNormLayer *layer2_0_downsample_1_bn =
                new BatchNormLayer(w_path + "layer2.0.downsample.1");

        layer2_0_conv1->set_input(layer1_1_relu2->get_output());
        layer2_0_bn1->set_input(layer2_0_conv1->get_output());
        layer2_0_relu1->set_input(layer2_0_bn1->get_output());
        layer2_0_conv2->set_input(layer2_0_relu1->get_output());
        layer2_0_bn2->set_input(layer2_0_conv2->get_output());
        layer2_0_relu2->set_input(layer2_0_bn2->get_output());

        // downsample
        layer2_0_downsample_0_conv->set_input(layer1_1_relu2->get_output());
        layer2_0_downsample_1_bn->set_input(
                layer2_0_downsample_0_conv->get_output());

        // block2
        ConvLayer *layer2_1_conv1 =
                new ConvLayer(cublas_handle, w_path + "layer2.1.conv1", 1, 1, false);
        BatchNormLayer *layer2_1_bn1 = new BatchNormLayer(w_path + "layer2.1.bn1");
        ReluLayer *layer2_1_relu1 = new ReluLayer();
        ConvLayer *layer2_1_conv2 =
                new ConvLayer(cublas_handle, w_path + "layer2.1.conv2", 1, 1, false);
        BatchNormLayer *layer2_1_bn2 = new BatchNormLayer(w_path + "layer2.1.bn2");
        ReluLayer *layer2_1_relu2 = new ReluLayer();

        layer2_1_conv1->set_input(layer2_0_relu2->get_output());
        layer2_1_bn1->set_input(layer2_1_conv1->get_output());
        layer2_1_relu1->set_input(layer2_1_bn1->get_output());
        layer2_1_conv2->set_input(layer2_1_relu1->get_output());
        layer2_1_bn2->set_input(layer2_1_conv2->get_output());
        layer2_1_relu2->set_input(layer2_1_bn2->get_output());

        // layer3

        // block1
        ConvLayer *layer3_0_conv1 =
                new ConvLayer(cublas_handle, w_path + "layer3.0.conv1", 1, 2, false);
        BatchNormLayer *layer3_0_bn1 = new BatchNormLayer(w_path + "layer3.0.bn1");
        ReluLayer *layer3_0_relu1 = new ReluLayer();
        ConvLayer *layer3_0_conv2 =
                new ConvLayer(cublas_handle, w_path + "layer3.0.conv2", 1, 1, false);
        BatchNormLayer *layer3_0_bn2 = new BatchNormLayer(w_path + "layer3.0.bn2");
        ReluLayer *layer3_0_relu2 = new ReluLayer();

        // downsample
        ConvLayer *layer3_0_downsample_0_conv = new ConvLayer(
                cublas_handle, w_path + "layer3.0.downsample.0", 0, 2, false);
        BatchNormLayer *layer3_0_downsample_1_bn =
                new BatchNormLayer(w_path + "layer3.0.downsample.1");

        layer3_0_conv1->set_input(layer2_1_relu2->get_output());
        layer3_0_bn1->set_input(layer3_0_conv1->get_output());
        layer3_0_relu1->set_input(layer3_0_bn1->get_output());
        layer3_0_conv2->set_input(layer3_0_relu1->get_output());
        layer3_0_bn2->set_input(layer3_0_conv2->get_output());
        layer3_0_relu2->set_input(layer3_0_bn2->get_output());

        layer3_0_downsample_0_conv->set_input(layer2_1_relu2->get_output());
        layer3_0_downsample_1_bn->set_input(
                layer3_0_downsample_0_conv->get_output());

        // block2
        ConvLayer *layer3_1_conv1 =
                new ConvLayer(cublas_handle, w_path + "layer3.1.conv1", 1, 1, false);
        BatchNormLayer *layer3_1_bn1 = new BatchNormLayer(w_path + "layer3.1.bn1");
        ReluLayer *layer3_1_relu1 = new ReluLayer();
        ConvLayer *layer3_1_conv2 =
                new ConvLayer(cublas_handle, w_path + "layer3.1.conv2", 1, 1, false);
        BatchNormLayer *layer3_1_bn2 = new BatchNormLayer(w_path + "layer3.1.bn2");
        ReluLayer *layer3_1_relu2 = new ReluLayer();

        layer3_1_conv1->set_input(layer3_0_relu2->get_output());
        layer3_1_bn1->set_input(layer3_1_conv1->get_output());
        layer3_1_relu1->set_input(layer3_1_bn1->get_output());
        layer3_1_conv2->set_input(layer3_1_relu1->get_output());
        layer3_1_bn2->set_input(layer3_1_conv2->get_output());
        layer3_1_relu2->set_input(layer3_1_bn2->get_output());

        // layer3

        // block1
        ConvLayer *layer4_0_conv1 =
                new ConvLayer(cublas_handle, w_path + "layer4.0.conv1", 1, 2, false);
        BatchNormLayer *layer4_0_bn1 = new BatchNormLayer(w_path + "layer4.0.bn1");
        ReluLayer *layer4_0_relu1 = new ReluLayer();
        ConvLayer *layer4_0_conv2 =
                new ConvLayer(cublas_handle, w_path + "layer4.0.conv2", 1, 1, false);
        BatchNormLayer *layer4_0_bn2 = new BatchNormLayer(w_path + "layer4.0.bn2");
        ReluLayer *layer4_0_relu2 = new ReluLayer();

        // downsample
        ConvLayer *layer4_0_downsample_0_conv = new ConvLayer(
                cublas_handle, w_path + "layer4.0.downsample.0", 0, 2, false);
        BatchNormLayer *layer4_0_downsample_1_bn =
                new BatchNormLayer(w_path + "layer4.0.downsample.1");

        layer4_0_conv1->set_input(layer3_1_relu2->get_output());
        layer4_0_bn1->set_input(layer4_0_conv1->get_output());
        layer4_0_relu1->set_input(layer4_0_bn1->get_output());
        layer4_0_conv2->set_input(layer4_0_relu1->get_output());
        layer4_0_bn2->set_input(layer4_0_conv2->get_output());
        layer4_0_relu2->set_input(layer4_0_bn2->get_output());

        layer4_0_downsample_0_conv->set_input(layer3_1_relu2->get_output());
        layer4_0_downsample_1_bn->set_input(
                layer4_0_downsample_0_conv->get_output());

        // block2
        ConvLayer *layer4_1_conv1 =
                new ConvLayer(cublas_handle, w_path + "layer4.1.conv1", 1, 1, false);
        BatchNormLayer *layer4_1_bn1 = new BatchNormLayer(w_path + "layer4.1.bn1");
        ReluLayer *layer4_1_relu1 = new ReluLayer();
        ConvLayer *layer4_1_conv2 =
                new ConvLayer(cublas_handle, w_path + "layer4.1.conv2", 1, 1, false);
        BatchNormLayer *layer4_1_bn2 = new BatchNormLayer(w_path + "layer4.1.bn2");
        ReluLayer *layer4_1_relu2 = new ReluLayer();

        layer4_1_conv1->set_input(layer4_0_relu2->get_output());
        layer4_1_bn1->set_input(layer4_1_conv1->get_output());
        layer4_1_relu1->set_input(layer4_1_bn1->get_output());
        layer4_1_conv2->set_input(layer4_1_relu1->get_output());
        layer4_1_bn2->set_input(layer4_1_conv2->get_output());
        layer4_1_relu2->set_input(layer4_1_bn2->get_output());

        // head
        AvgPoolLayer *avg_pool = new AvgPoolLayer(3);

        LinearLayer *linear = new LinearLayer(cublas_handle, w_path + "fc");

        avg_pool->set_input(layer4_1_relu2->get_output());
        linear->set_input(avg_pool->get_output());

        std::shared_ptr<Tensor<float>> output = linear->get_output();
        float *cpu_result = (float *) malloc(output->count() * sizeof(float));

        // FORWARD

        auto start = std::chrono::high_resolution_clock::now();

        // LIKWID_NVMARKER_INIT;

        // LIKWID_NVMARKER_REGISTER("forward");
        MeasurementSeries time;
        MeasurementSeries energy;
        for (int it = 0; it < iters; ++it) {

            // LIKWID_NVMARKER_START("forward");
            input->from_cpu(float_data_ptr);
            // stub
            double t1 = dtime();
            conv1->forward();
            bn1->forward();
            relu1->forward();
            mp1->forward();

            // layer1
            layer1_0_conv1->forward();
            layer1_0_bn1->forward();
            layer1_0_relu1->forward();
            layer1_0_conv2->forward();
            layer1_0_bn2->forward();
            *(layer1_0_bn2->get_output()) += *(mp1->get_output());
            layer1_0_relu2->forward();

            layer1_1_conv1->forward();
            layer1_1_bn1->forward();
            layer1_1_relu1->forward();
            layer1_1_conv2->forward();
            layer1_1_bn2->forward();
            *(layer1_1_bn2->get_output()) += *(layer1_0_relu2->get_output());
            layer1_1_relu2->forward();

            layer2_0_conv1->forward();
            layer2_0_bn1->forward();
            layer2_0_relu1->forward();
            layer2_0_conv2->forward();
            layer2_0_bn2->forward();
            layer2_0_downsample_0_conv->forward();
            layer2_0_downsample_1_bn->forward();
            *(layer2_0_bn2->get_output()) +=
                    *(layer2_0_downsample_1_bn->get_output());
            layer2_0_relu2->forward();

            layer2_1_conv1->forward();
            layer2_1_bn1->forward();
            layer2_1_relu1->forward();
            layer2_1_conv2->forward();
            layer2_1_bn2->forward();
            *(layer2_1_bn2->get_output()) += *(layer2_0_relu2->get_output());
            layer2_1_relu2->forward();

            layer3_0_conv1->forward();
            layer3_0_bn1->forward();
            layer3_0_relu1->forward();
            layer3_0_conv2->forward();
            layer3_0_bn2->forward();
            layer3_0_downsample_0_conv->forward();
            layer3_0_downsample_1_bn->forward();
            *(layer3_0_bn2->get_output()) +=
                    *(layer3_0_downsample_1_bn->get_output());
            layer3_0_relu2->forward();

            layer3_1_conv1->forward();
            layer3_1_bn1->forward();
            layer3_1_relu1->forward();
            layer3_1_conv2->forward();
            layer3_1_bn2->forward();
            *(layer3_1_bn2->get_output()) += *(layer3_0_relu2->get_output());
            layer3_1_relu2->forward();

            // layer4
            layer4_0_conv1->forward();
            layer4_0_bn1->forward();
            layer4_0_relu1->forward();
            layer4_0_conv2->forward();
            layer4_0_bn2->forward();
            layer4_0_downsample_0_conv->forward();
            layer4_0_downsample_1_bn->forward();
            *(layer4_0_bn2->get_output()) +=
                    *(layer4_0_downsample_1_bn->get_output());
            layer4_0_relu2->forward();

            layer4_1_conv1->forward();
            layer4_1_bn1->forward();
            layer4_1_relu1->forward();
            layer4_1_conv2->forward();
            layer4_1_bn2->forward();
            *(layer4_1_bn2->get_output()) += *(layer4_0_relu2->get_output());
            layer4_1_relu2->forward();

            // head
            avg_pool->forward();
            avg_pool->get_output()->reshape({n, avg_pool->get_output()->count() / n});
            linear->forward();

            
            // LIKWID_NVMARKER_STOP("forward");
            cudaDeviceSynchronize();
            double t2 = dtime();

            // please comment below 3 lines if you are not measuring power energy
            auto energy_tmp = execCommand("nvidia-smi -q -d POWER | grep \"Power Draw                        : \" | cut -c 45- | sed \'s/.$// \'", status);
            energy_tmp.erase(std::remove(energy_tmp.begin(), energy_tmp.end(), ' '), energy_tmp.end());
            energy.add(std::stod(energy_tmp));

            time.add(t2 - t1);
            output->to_cpu(cpu_result);
        }
        

        // LIKWID_NVMARKER_CLOSE;
        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                        .count();

        // debug_array(linear->get_output()->_ptr, linear->get_output()->count());

        float *tmp_ptr = cpu_result;
        for (int i = 0; i < output->size()[0]; ++i) {
            float max_flt = tmp_ptr[0];
            int maxj = 0;
            for (int j = 1; j < output->size()[1]; ++j) {
                if (tmp_ptr[j] > max_flt) {
                    maxj = j;
                    max_flt = tmp_ptr[j];
                }
            }
            tmp_ptr += output->size()[1];
            //   std::cout << i << " : " << label_map[maxj] << std::endl;
        }
        float bw = 4.5 * double(n) * sizeof(float) / time.value() / 1.0e3;
        float gf = 0.04 * double(n) / time.value();

        // std::cout << n << " " << time.value() << " " << n / time.value() << " " << bw << " " << gf;
        
        // std::cout << "\n";
        // please comment below line if you are not measuring power energy
        std::cout << n << " " << energy.minValue() << " " << energy.maxValue() << " "<< energy.value()<< std::endl;


        // TODO: delete all!!

        free(img->data);
        free(img);
    } catch (const std::exception &e) {
        std::cout << "Exception: " << e.what() << std::endl;
    } catch (const std::string &e) {
        std::cout << "Exception: " << e << std::endl;
    }
}
