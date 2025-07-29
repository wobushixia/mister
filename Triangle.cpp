// created by Yifan Xia 7/25 - 2025
#include "Triangle.hpp"
#include <eigen3/Eigen/Eigen>

using namespace Eigen;

void Triangle::setVertex(int ind, const Vector4f& ver){
  assert(ind >= 0 && ind < 3);
  v[ind] = ver;
}

void Triangle::setNormal(int ind, const Vector3f& n){
  assert(ind >= 0 && ind < 3);
  normal[ind] = n;
}

void Triangle::setColor(int ind, float r, float g, float b){
  assert(ind >= 0 && ind < 3);
  color[ind] = Vector3f(r,g,b);
}

void Triangle::setTexCoord(int ind, Vector2f uv){
  assert(ind >= 0 && ind < 3);
  tex_coords[ind] = uv;
}

void Triangle::setNormals(const std::array<Vector3f, 3>& normals){
  normal[0] = normals[0];
  normal[1] = normals[1];
  normal[2] = normals[2];
}

void Triangle::setColors(const std::array<Vector3f, 3>& colors){
  color[0] = colors[0];
  color[1] = colors[1];
  color[2] = colors[2];
}

std::array<Vector4f, 3> Triangle::toVector4() const
{
    std::array<Vector4f, 3> res;
    std::transform(std::begin(v), std::end(v), res.begin(), [](auto& vec) { return Vector4f(vec.x(), vec.y(), vec.z(), 1.f); });
    return res;
}

Triangle::Triangle(){

}