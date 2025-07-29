// created by Yifan Xia 7/26 - 2025

#pragma once
#include <eigen3/Eigen/Eigen>
#include "Texture.hpp"

class fragment_shader_payload
{
private:
  /* data */
public:
  
  fragment_shader_payload(Eigen::Vector3f c, Eigen::Vector3f n, Eigen::Vector2f tc, Texture* tex): color(c), normal(n), tex_coords(tc), texture(tex) {};

  Eigen::Vector3f color;
  Eigen::Vector3f normal;
  Eigen::Vector2f tex_coords;
  Eigen::Vector3f view_pos;
  Texture* texture = nullptr;

};
