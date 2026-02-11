#include <iostream>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>
#include "yolov8Predictor.h"
#include "utils.h"

using namespace std;
using namespace sl;

// roi variables
vector<cv::Point> roiPoints; 
bool roiFinalized = false;

void mouseHandler(int event, int x, int y, int flags, void* userdata)
{
    if (roiFinalized) return;

    if (event == cv::EVENT_LBUTTONDOWN) // add point
    {
        roiPoints.push_back(cv::Point(x, y));
        if (roiPoints.size() == 5)
            roiFinalized = true;
    }
    else if (event == cv::EVENT_RBUTTONDOWN && !roiPoints.empty()) // remove last point
    {
        roiPoints.pop_back();
        roiFinalized = false;
    }
}

void drawROI(cv::Mat &frame)
{
    if (roiPoints.empty())
        return;

    cv::Mat overlay;
    frame.copyTo(overlay);

    // roi fill
    if (roiFinalized && roiPoints.size() == 5)
    {
        vector<vector<cv::Point>> contour{roiPoints};
        cv::fillPoly(overlay, contour, cv::Scalar(0, 0, 255));
        cv::addWeighted(overlay, 0.4, frame, 0.6, 0, frame);
    }

    for (size_t i = 0; i + 1 < roiPoints.size(); ++i)
        cv::line(frame, roiPoints[i], roiPoints[i + 1], cv::Scalar(0, 255, 0), 2);

    if (roiFinalized && roiPoints.size() == 5)
        cv::line(frame, roiPoints.back(), roiPoints.front(), cv::Scalar(0, 255, 0), 2);
}

int main()
{
    float confThreshold = 0.5f;
    float iouThreshold = 0.5f;
    float maskThreshold = 0.6f;
    bool useGPU = true;

    string modelPath = "../..model.onnx";
    string classNamesPath = "../..coco.names";

    const vector<string> classNames = utils::loadNames(classNamesPath);
    if (classNames.empty())
    {
        cerr << "[ERROR] class names empty\n";
        return -1;
    }

    YOLOPredictor predictor(modelPath, useGPU, confThreshold, iouThreshold, maskThreshold);

    sl::Camera zed;
    InitParameters initParams;
    initParams.camera_resolution = RESOLUTION::HD720;
    initParams.camera_fps = 30;
    initParams.depth_mode = DEPTH_MODE::PERFORMANCE;
    initParams.enable_image_enhancement = true;

    if (zed.open(initParams) != ERROR_CODE::SUCCESS)
    {
        cerr << "[ERROR] ZED failed to open\n";
        return -1;
    }

    // cout << "Left Click  : add ROI point\n";
    // cout << "Right Click : remove last point\n";
    // cout << "R           : reset ROI\n";
    // cout << "Q           : quit program\n";

    cv::namedWindow("Pov");
    cv::setMouseCallback("Pov", mouseHandler, nullptr);

    while (true)
    {
        if (zed.grab() != ERROR_CODE::SUCCESS)
            continue;

        sl::Mat image;
        zed.retrieveImage(image, VIEW::LEFT);
        cv::Mat frame(image.getHeight(), image.getWidth(), CV_8UC4, image.getPtr<sl::uchar1>(MEM::CPU));
        cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);

        // night mode enhancement
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        double avgBrightness = cv::mean(gray)[0];

        if (avgBrightness < 60.0)
        {
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8,8));
            clahe->apply(gray, gray);

            cv::Mat lut(1, 256, CV_8U);
            uchar* p = lut.ptr();
            double gamma = 0.6;
            for (int i=0; i<256; i++)
                p[i] = cv::saturate_cast<uchar>(pow(i/255.0, gamma)*255);

            cv::Mat nightFrame;
            cv::LUT(gray, lut, nightFrame);
            cv::cvtColor(nightFrame, frame, cv::COLOR_GRAY2BGR);

            // cv::putText(frame, "Night Mode ON", cv::Point(30,70), cv::FONT_HERSHEY_SIMPLEX, 1.0,
            //             cv::Scalar(255,255,0), 2);
        }

        vector<Yolov8Result> results = predictor.predict(frame);
        bool personDetected = false;

        if (roiFinalized && roiPoints.size() >= 3)
        {
            cv::Rect roiBox = cv::boundingRect(roiPoints);
            for (auto &r : results)
            {
                if (classNames[r.classId] == "person" && r.conf > confThreshold)
                {
                    cv::Rect overlap = roiBox & r.box;
                    double ratio = (double)overlap.area() / r.box.area();
                    if (ratio > 0.02)
                    {
                        personDetected = true;
                        break;
                    }
                }
            }
        }

        // display status
        string status = personDetected ? "PERSON DETECTED" : "NO PERSON";
        cv::putText(frame, status, cv::Point(30,40), cv::FONT_HERSHEY_SIMPLEX, 1,
                    personDetected ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255), 3);

        utils::visualizeDetection(frame, results, classNames);
        drawROI(frame);
        cv::imshow("Camera", frame);

        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q') break;

        if (key == 'r' || key == 'R')
        {
            roiPoints.clear();
            roiFinalized = false;
            cout << "[RESET] ROI cleared\n";
        }
    }

    zed.close();
    cv::destroyAllWindows();
    return 0;
}
