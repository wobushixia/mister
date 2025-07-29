#pragma once

// created by Yifan Xia 7/26 - 2025
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <eigen3/Eigen/Eigen>

class Texture{
private:
  cv::Mat img;
public:
  int width;
  int height;
  Texture(std::string name);
  Eigen::Vector3f get_color(float u, float v);
};
