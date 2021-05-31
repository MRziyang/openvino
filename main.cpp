#include <iostream>
#include <string>
#include <vector>
#include <ie_core.hpp>
#include <cpp/ie_cnn_network.h>
//#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
//#include <opencv2/dnn/dnn.hpp>

//using namespace cv;
using namespace std;
//using namespace dnn;

int main()
{
    //speedTest();
    cout << "Testing................" << endl;
    system("pause");
    string binPath = "D:/E-75-newPrepro-ssd.bin";
    string xmlPath = "D:/E-75-newPrepro-ssd.xml";
    cout << binPath << '\n' << xmlPath << endl;
    system("pause");
    InferenceEngine::Core ie;
    cout << "Reading Network" << endl;
    system("pause");
    auto network = ie.ReadNetwork(xmlPath, binPath);

    cout << "Setting Input and Output" << endl;
    system("pause");
    InferenceEngine::InputsDataMap input_info(network.getInputsInfo());
    InferenceEngine::OutputsDataMap output_info(network.getOutputsInfo());

    cout << "Setting Input" << endl;
    system("pause");
    for (auto& item : input_info)
    {
        auto input_data = item.second;
        input_data->setPrecision(InferenceEngine::Precision::FP32);
        input_data->setLayout(InferenceEngine::Layout::NCHW);
        //input_data->setPrecision(InferenceEngine::Precision::FP32);
        //input_data->getPreProcess().setResizeAlgorithm(InferenceEngine::RESIZE_BILINEAR);
    }
    printf("get it");
    for (auto& item : output_info)
    {
        auto output_data = item.second;
        output_data->setPrecision(InferenceEngine::Precision::FP32);
    }

    // 创建可执行网络对象
    auto executable_network = ie.LoadNetwork(network, "CPU");
    // 请求推断图
    auto infer_request = executable_network.CreateInferRequest();
    cout << "Inferencing" << endl;
    system("pause");
    for (auto& item : input_info) {
        auto input_name = item.first;
        /** Getting input blob **/

        auto input = infer_request.GetBlob(input_name);
        size_t num_channels = input->getTensorDesc().getDims()[1];
        size_t h = input->getTensorDesc().getDims()[2];
        size_t w = input->getTensorDesc().getDims()[3];
        std::cout << h << w << std::endl;
        size_t image_size = h * w;
        //Mat blob_image;
        // NCHW
        float* data = static_cast<float*>(input->buffer());

        for (size_t row = 0; row < h; row++) {
            for (size_t col = 0; col < w; col++) {
                for (size_t ch = 0; ch < num_channels; ch++) {
                    data[image_size * ch + row * w + col] = 1.0/*blob_image.at<Vec3b>(row, col)[ch]*/;
                }
            }
        }

        clock_t startTime = clock();
        {
            // 执行预测
            infer_request.Infer();

        } clock_t startTime1 = clock(); cout << double(startTime1 - startTime) * 1000 / CLOCKS_PER_SEC << "ms\n"; {
                // 执行预测
                infer_request.Infer();

                // 处理输出结果
                //for (auto& item : output_info) {

                // 获取输出数据

        }
        clock_t endTime = clock();
        cout << double(endTime - startTime1) * 1000 / CLOCKS_PER_SEC << "ms\n";
        //// 解析输出结果
        //for (int curProposal = 0; curProposal < maxProposalCount; curProposal++) {
        //    float label = detection[curProposal * objectSize + 1];
        //    float confidence = detection[curProposal * objectSize + 2];
        //    float xmin = detection[curProposal * objectSize + 3] * 224;
        //    float ymin = detection[curProposal * objectSize + 4] * 224;
        //    float xmax = detection[curProposal * objectSize + 5] * 224;
        //    float ymax = detection[curProposal * objectSize + 6] * 224;
        //    if (confidence > 0.5) {
        //        printf("label id : %d ", static_cast<float>(label));
        //        Rect rect;
        //        rect.x = static_cast<float>(xmin);
        //        rect.y = static_cast<float>(ymin);
        //        rect.width = static_cast<float>(xmax - xmin);
        //        rect.height = static_cast<float>(ymax - ymin);
        //        putText(src, "OpenVINO-2021R02", Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2, 8);
        //        rectangle(src, rect, Scalar(0, 255, 255), 2, 8, 0);
        //    }
        //    std::cout << std::endl;
        //}
        ////}
        //imshow("OpenVINO+SSD人脸检测", src);
    }
    //waitKey();

    system("pause");

    return 0;
}