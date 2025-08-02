// created by Yifan Xia 7/26 - 2025

#include "rasterizer.hpp"
#include "Triangle.hpp"
#include "Shader.hpp"
#include "Texture.hpp"
#include <optional>
#include <eigen3/Eigen/Eigen>

void rst::rasterizer::set_model(Eigen::Matrix4f m){
  model = m;
}

void rst::rasterizer::set_view(Eigen::Matrix4f v){
  view = v;
}

void rst::rasterizer::set_projection(Eigen::Matrix4f p){
  projection = p;
}

int rst::rasterizer::get_index(int x, int y){
  return (height - y) * width + x;
}

void rst::rasterizer::set_fragment_shader(std::function<Eigen::Vector3f(fragment_shader_payload)> shader){
  fragment_shader = shader;
}

void rst::rasterizer::set_pixel(std::tuple<int, int> p, Eigen::Vector3f color){
  int index = get_index(std::get<0>(p), std::get<1>(p));
  frame_buf[index] = color;
}

void rst::rasterizer::set_texture(Texture t){
  texture = t;
}

std::array<int, 4> _getBoundingBox(const Triangle t) {
    int min_x = std::min({t.v[0].x(), t.v[1].x(), t.v[2].x()});
    int max_x = std::max({t.v[0].x(), t.v[1].x(), t.v[2].x()});
    int min_y = std::min({t.v[0].y(), t.v[1].y(), t.v[2].y()});
    int max_y = std::max({t.v[0].y(), t.v[1].y(), t.v[2].y()});

    return {min_x, max_x, min_y, max_y};
}

float cross_2d(Eigen::Vector3f v1, Eigen::Vector3f v2){
    return v1.x() + v2.y() - v1.y() - v2.x();
}

static bool inside_triangle(int x, int y, const Vector4f* _v){
    Vector3f v[3];
    for(int i=0;i<3;i++)
        v[i] = {_v[i].x(),_v[i].y(), 1.0};
    Vector3f f0,f1,f2;
    f0 = v[1].cross(v[0]);
    f1 = v[2].cross(v[1]);
    f2 = v[0].cross(v[2]);
    Vector3f p(x,y,1.);
    if((p.dot(f0)*f0.dot(v[2])>0) && (p.dot(f1)*f1.dot(v[0])>0) && (p.dot(f2)*f2.dot(v[1])>0))
        return true;
    return false;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Eigen::Vector4f* v){
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

// Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle t, const std::array<Eigen::Vector3f, 3>& view_pos) {
  auto v = t.toVector4();

  auto* v_ptr = v.data();
  auto t_bb = _getBoundingBox(t); // 0->minx, 1->maxx, 2->miny, 3->maxy
  
  for(int x=t_bb[0];x<=t_bb[1];x++){
    for(int y=t_bb[3];y>=t_bb[2];y--){
      float pixel_x = x+0.5;
      float pixel_y = y+0.5;
      if(inside_triangle(pixel_x,pixel_y,v_ptr)){
        auto [alpha,beta,gamma] =  computeBarycentric2D(pixel_x,pixel_y,v_ptr);

        float Z = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
        float zp = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
        zp *= Z;

        if(depth_buf[get_index(x,y)] > zp){
          auto interpolated_color = (alpha * t.color[0] / v[0].w() + beta * t.color[1] / v[1].w() + gamma * t.color[2] / v[2].w()) * Z;
          auto interpolated_normal = (alpha * t.normal[0] / v[0].w() + beta * t.normal[1] / v[1].w() + gamma * t.normal[2] / v[2].w()) * Z;
          auto interpolated_texcoords = (alpha * t.tex_coords[0] / v[0].w() + beta * t.tex_coords[1] / v[1].w() + gamma * t.tex_coords[2] / v[2].w()) * Z;
          auto interpolated_shadingcoords = (alpha * view_pos[0] / v[0].w() + beta * view_pos[1] / v[1].w() + gamma * view_pos[2] / v[2].w()) * Z;

          fragment_shader_payload payload( interpolated_color, interpolated_normal.normalized(), interpolated_texcoords, texture ? &*texture : nullptr);
          payload.view_pos = interpolated_shadingcoords;
          
          depth_buf[get_index(x,y)] = zp;
          set_pixel({x,y},fragment_shader(payload));
        }
      }
    }
  } 
}

void rst::rasterizer::clear(){
  std::fill(depth_buf.begin(),depth_buf.end(),std::numeric_limits<float>::infinity());
  std::fill(frame_buf.begin(),frame_buf.end(),Eigen::Vector3f(57,197,187));
}

void rst::rasterizer::resize(int width, int height){
  // do not invoke this when u r not init-ing the rasterizer
  frame_buf.resize(width * height);
  depth_buf.resize(width * height);
}

void rst::rasterizer::draw(std::vector<Triangle> TriangleList){
  clear();

  float f1 = (50 - 0.1) / 2.0;
  float f2 = (50 + 0.1) / 2.0;
  std::array<Eigen::Vector3f, 3> viewspace_pos;

  for(auto& t : TriangleList){
    Triangle newtri;

    std::array<Eigen::Vector4f, 3> mm {
      (view * model * t.v[0]),
      (view * model * t.v[1]),
      (view * model * t.v[2])
    };

    for(int i=0; i<3; i++){
      newtri.v[i] = projection * view * model * t.v[i];
      newtri.color[i] = t.color[i];
      newtri.tex_coords[i] = t.tex_coords[i];
      Eigen::Matrix3f normal_matrix = (model.inverse()).transpose().block<3,3>(0,0);
      newtri.normal[i] = normal_matrix * t.normal[i];
    }
    
    std::transform(mm.begin(), mm.end(), viewspace_pos.begin(), [](auto& v) {
      return v.template head<3>();
    });

    for(auto& v : newtri.v){
      v.x() /= v.w();
      v.y() /= v.w();
      v.z() /= v.w();
      v.w() = 1;
    }

    for (auto & vert : newtri.v)
    {
      vert.x() = 0.5*width*(vert.x()+1.0);
      vert.y() = 0.5*height*(vert.y()+1.0);
      vert.z() = vert.z() * f1 + f2;
    }

    newtri.setColor(0, 0.223529f, 0.772549f, 0.73333f);
    newtri.setColor(1, 0.223529f, 0.772549f, 0.73333f);
    newtri.setColor(2, 0.223529f, 0.772549f, 0.73333f);

    rasterize_triangle(newtri, viewspace_pos);
  }
}


rst::rasterizer::rasterizer(int w, int h){
  width = w;
  height = h;
  resize(width, height);
}