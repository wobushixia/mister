#pragma once

// created by Yifan Xia 7/24 - 2025
#include <eigen3/Eigen/Eigen>
#include "Texture.hpp"

using namespace Eigen;
class Triangle{

public:
    Vector4f v[3];
    Vector3f color[3];
    Vector2f tex_coords[3];
    Vector3f normal[3];

    Texture *tex= nullptr;
    Triangle();

    Eigen::Vector4f a() const { return v[0]; }
    Eigen::Vector4f b() const { return v[1]; }
    Eigen::Vector4f c() const { return v[2]; }

    void setVertex(int ind, const Vector4f& ver);
    void setNormal(int ind, const Vector3f& n);
    void setColor(int ind, float r, float g, float b);
    void setTexCoord(int ind,Vector2f uv );

    void setNormals(const std::array<Vector3f, 3>& normals);
    void setColors(const std::array<Vector3f, 3>& colors);
    std::array<Vector4f, 3> toVector4() const;
};
