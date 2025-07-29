#pragma once
#include <eigen3/Eigen/Eigen>
#include "Shader.hpp"
#include "Triangle.hpp"
#include "Texture.hpp"
#include <optional>

// created by Yifan Xia 7/26 - 2025

namespace rst{
  class rasterizer
  {
  private:
    Eigen::Matrix4f model, view, projection;
    std::function<Eigen::Vector3f(fragment_shader_payload)> fragment_shader;
    std::optional<Texture> texture;
  public:
    int width, height;
    std::vector<Eigen::Vector3f> frame_buf;
    std::vector<float> depth_buf;
    void set_model(Eigen::Matrix4f m), set_view(Eigen::Matrix4f v), set_projection(Eigen::Matrix4f p);
    void clear();
    void resize(int, int);
    void rasterize_triangle(const Triangle, const std::array<Eigen::Vector3f, 3>&);
    void set_texture(Texture);
    void set_pixel(std::tuple<int, int>, Eigen::Vector3f);
    void set_fragment_shader(std::function<Eigen::Vector3f(fragment_shader_payload)>);
    void draw(std::vector<Triangle>);
    int get_index(int, int);
    rasterizer(int, int);
  };
}