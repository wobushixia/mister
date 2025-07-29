// created by Yifan Xia 7/26 - 2025
#include "Texture.hpp"
#include <eigen3/Eigen/Eigen>
#include <opencv2/opencv.hpp>

Texture::Texture(const std::string name)
{
  img = cv::imread(name);
  cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
  width = img.rows;
  height = img.cols;
}

Eigen::Vector3f Texture::get_color(float u, float v){
  auto u_img = u * width;
  auto v_img = (1 - v) * height;
  auto color = img.at<cv::Vec3b>(v_img, u_img);
  return Eigen::Vector3f(color[0], color[1], color[2]);
}