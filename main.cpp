// created by Yifan Xia 7/24 - 2025

#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Eigen>
#include "rasterizer.hpp"
#include "OBJ_Loader.h"
#include "global.hpp"
#include "Triangle.hpp"

Eigen::Vector3f bisector_generate(Eigen::Vector3f vec1, Eigen::Vector3f vec2){
    return (vec1 + vec2).normalized();
}

Eigen::Vector3f normal_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f return_color = (payload.normal.head<3>().normalized() + Eigen::Vector3f(1.0f, 1.0f, 1.0f)) / 2.f;
    Eigen::Vector3f result;
    result << return_color.x() * 255, return_color.y() * 255, return_color.z() * 255;
    return result;
}

struct light
{
    Eigen::Vector3f position;
    Eigen::Vector3f intensity;
};

Eigen::Vector3f texture_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f return_color = {0, 0, 0};
    if (payload.texture)
    {
        // TODO: Get the texture value at the texture coordinates of the current fragment
        float u = payload.tex_coords[0];
        float v = payload.tex_coords[1];
        if(u<0 || v<0 || u>1 || v>1) return {57,197,187};
        return_color = payload.texture->get_color(u,v);
    }
    Eigen::Vector3f texture_color;
    texture_color << return_color.x(), return_color.y(), return_color.z();

    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = texture_color / 255.f;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = texture_color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};

    for (auto& light : lights)
    {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.

        Eigen::Vector3f light_unit = (light.position - point).normalized();
        Eigen::Vector3f normal_unit = normal.normalized();
        Eigen::Vector3f view_unit = (eye_pos - point).normalized();
        Eigen::Vector3f I = light.intensity / pow((light.position - point).norm(),2);

        Eigen::Vector3f ambient_color = ka.cwiseProduct(amb_light_intensity);
        Eigen::Vector3f diffuse_color = std::max(normal_unit.dot(light_unit), 0.f) * kd.cwiseProduct(I);
        Eigen::Vector3f specular_color = ks.cwiseProduct(I) * pow(normal_unit.dot(bisector_generate(light_unit,view_unit)),p);

        result_color += ambient_color + diffuse_color + specular_color;
    }

    return result_color * 255.f;
}

Eigen::Vector3f phong_fragment_shader(const fragment_shader_payload& payload){
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};
    for (auto& light : lights)
    {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.
        Eigen::Vector3f light_unit = (light.position - point).normalized();
        Eigen::Vector3f normal_unit = normal.normalized();
        Eigen::Vector3f view_unit = (eye_pos - point).normalized();
        Eigen::Vector3f I = light.intensity / pow((light.position - point).norm(),2);

        Eigen::Vector3f ambient_color = ka.cwiseProduct(amb_light_intensity);
        Eigen::Vector3f diffuse_color = std::max(normal_unit.dot(light_unit), 0.f) * kd.cwiseProduct(I);
        Eigen::Vector3f specular_color = ks.cwiseProduct(I) * pow(normal_unit.dot(bisector_generate(light_unit,view_unit)),p);

        result_color += ambient_color + diffuse_color + specular_color;
    }

    return result_color * 255.f;
}

Eigen::Vector3f displacement_fragment_shader(const fragment_shader_payload& payload)
{
    
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color; 
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    float kh = 0.2, kn = 0.1;
    
    // TODO: Implement displacement mapping here
    // Let n = normal = (x, y, z)
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    // Vector b = n cross product t
    // Matrix TBN = [t b n]
    // dU = kh * kn * (h(u+1/w,v)-h(u,v))
    // dV = kh * kn * (h(u,v+1/h)-h(u,v))
    // Vector ln = (-dU, -dV, 1)
    // Position p = p + kn * n * h(u,v)
    // Normal n = normalize(TBN * ln)

    float x = normal.x();
    float y = normal.y();
    float z = normal.z();
    float sqrt_xx_zz = sqrt(x*x + z*z);
    Eigen::Vector3f t(x*y/sqrt_xx_zz, sqrt_xx_zz, z*y/sqrt_xx_zz);
    Eigen::Vector3f b = normal.cross(t);
    Eigen::Matrix3f TBN;
    TBN << t , b , normal;

    if(payload.tex_coords.x()<0 || payload.tex_coords.x()>1 || payload.tex_coords.y() >1 || payload.tex_coords.y()<0) return {0,0,0};
    
    float w = payload.texture->width;
    float h = payload.texture->height;
    float u = std::clamp(payload.tex_coords.x(), 0.0f, 1.0f);
    float v = std::clamp(payload.tex_coords.y(), 0.0f, 1.0f);
    float dU = kh * kn * (payload.texture->get_color(u + 1.0f/w, v).norm() - payload.texture->get_color(u, v).norm());
    float dV = kh * kn * (payload.texture->get_color(u, v + 1.0f/h).norm() - payload.texture->get_color(u, v).norm());
    Eigen::Vector3f ln(-dU, -dV, 1.0f);

    Eigen::Vector3f pos = point + kn * normal * (payload.texture->get_color(u,v)).norm();
    normal = (TBN * ln).normalized();

    Eigen::Vector3f result_color = {0,0,0};

    for (auto& light : lights)
    {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.

        Eigen::Vector3f light_unit = (light.position - pos).normalized();
        Eigen::Vector3f normal_unit = normal;
        Eigen::Vector3f view_unit = (eye_pos - pos).normalized();
        Eigen::Vector3f I = light.intensity / (light.position - pos).squaredNorm();

        Eigen::Vector3f ambient_color = ka.cwiseProduct(amb_light_intensity);
        Eigen::Vector3f diffuse_color = std::max(normal_unit.dot(light_unit), 0.0f) * kd.cwiseProduct(I);
        Eigen::Vector3f specular_color = ks.cwiseProduct(I) * pow(std::max(normal_unit.dot(bisector_generate(light_unit,view_unit)),0.0f),p);

        result_color += ambient_color + diffuse_color + specular_color;
    }

    return result_color * 255.f;
}

Eigen::Vector3f bump_fragment_shader(const fragment_shader_payload& payload)
{
    
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color; 
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;


    float kh = 0.2, kn = 0.1;

    // TODO: Implement bump mapping here
    // Let n = normal = (x, y, z)
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    Eigen::Vector3f t(normal.x()*normal.y()/sqrt(normal.x()*normal.x()+normal.z()*normal.z()),sqrt(normal.x()*normal.x()+normal.z()*normal.z()),normal.z()*normal.y()/sqrt(normal.x()*normal.x()+normal.z()*normal.z()));
    Eigen::Vector3f b = normal.cross(t);
    // Matrix TBN = [t b n]
    Eigen::Matrix3f TBN;
    TBN << t, b, normal;
    // dU = kh * kn * (h(u+1/w,v)-h(u,v))
    // dV = kh * kn * (h(u,v+1/h)-h(u,v))

    if(payload.tex_coords.x()<0 || payload.tex_coords.x()>1 || payload.tex_coords.y() >1 || payload.tex_coords.y()<0) return {0,0,0};
    
    float dU = kh * kn * (payload.texture->get_color(payload.tex_coords.x() + 1.0f /payload.texture->width,payload.tex_coords.y()).norm() - payload.texture->get_color(payload.tex_coords.x(),payload.tex_coords.y()).norm());
    float dV = kh * kn * (payload.texture->get_color(payload.tex_coords.x(),payload.tex_coords.y() + 1.0f /payload.texture->height).norm() - payload.texture->get_color(payload.tex_coords.x(),payload.tex_coords.y()).norm());

    Eigen::Vector3f ln(-dU, -dV, 1);

    // Normal n = normalize(TBN * ln)
    normal = (TBN * ln).normalized();


    Eigen::Vector3f result_color = {0, 0, 0};
    result_color = normal;

    return result_color * 255.f;
}


Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1,0,0,-eye_pos[0],
                 0,1,0,-eye_pos[1],
                 0,0,1,-eye_pos[2],
                 0,0,0,1;

    view = translate*view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float angle)
{
    Eigen::Matrix4f rotation;
    angle = angle * MY_PI / 180.f;
    rotation << cos(angle), 0, sin(angle), 0,
                0, 1, 0, 0,
                -sin(angle), 0, cos(angle), 0,
                0, 0, 0, 1;

    Eigen::Matrix4f scale;
    scale << 2.5, 0, 0, 0,
              0, 2.5, 0, 0,
              0, 0, 2.5, 0,
              0, 0, 0, 1;

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;

    return translate * rotation * scale;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar)
{
    float radian = eye_fov * MY_PI / 180;
    float t = tan(radian/2) * zNear;
    float b = -t;
    float r = aspect_ratio * t;
    float l = -r;

    Eigen::Matrix4f persp_matrix;
    persp_matrix << zNear, 0, 0, 0,
                    0, zNear, 0, 0,
                    0, 0, zFar + zNear, -zFar * zNear,
                    0, 0, 1, 0;

    Eigen::Matrix4f ortho_scale;
    ortho_scale << -2/(r-l), 0, 0, 0,
                   0, -2/(t-b), 0, 0,
                   0, 0, 2/(zFar-zNear), 0,
                   0, 0, 0, 1;
    
    Eigen::Matrix4f ortho_translate;
    ortho_translate << 1, 0, 0, -(r+l)/2,
                       0, 1, 0, -(t+b)/2,
                       0, 0, -1, -(zNear+zFar)/2,
                       0, 0, 0, 1;
    
    Eigen::Matrix4f ortho_matrix = ortho_scale * ortho_translate;
    Eigen::Matrix4f projection = ortho_matrix * persp_matrix;

    return projection;
}

void log(std::string a){
    std::cout<<a<<std::endl;
}

int main(){
    system("cls");
    log("###########################################");
    log("#     Mister Renderer v1.0 by 5xian39     #");
    log("###########################################");
    log("mister is running...");
    std::vector<Triangle> TriangleList;

    objl::Loader loader;
    log("load OBJ_Loader successfully");

    std::string file_name = "output.png";
    log("please type the file name...");
    std::cin>>file_name;

    std::string obj_path = "../models/spot/spot_triangulated_good.obj";
    log("please type the obj path...");
    std::cin>>obj_path;

    bool loadResult = loader.LoadFile(obj_path);
    if(!loadResult) {log("ERROR!the path is invalid."); exit(-1);}
    log("load OBJ successfully.\nimporting Triangles...");
    for(auto mesh : loader.LoadedMeshes){
        for(int i=0;i<mesh.Vertices.size();i+=3){
            Triangle t;
            for(int j=0;j<3;j++){
                std::cout<<"["<<i<<","<<j<<"]"<<"setting triangle..."<<std::endl;
                t.setVertex(j, Eigen::Vector4f(loader.LoadedVertices[i+j].Position.X, loader.LoadedVertices[i+j].Position.Y, loader.LoadedVertices[i+j].Position.Z, 1.0f));
                t.setNormal(j, Eigen::Vector3f(loader.LoadedVertices[i+j].Normal.X, loader.LoadedVertices[i+j].Normal.Y, loader.LoadedVertices[i+j].Normal.Z));
                t.setTexCoord(j, Eigen::Vector2f(loader.LoadedVertices[i+j].TextureCoordinate.X, loader.LoadedVertices[i+j].TextureCoordinate.Y));
            }
            TriangleList.emplace_back(t);
        }
    }

    log("load TriangleList successfully.");
    int w = 700, h = 700;

    Eigen::Vector3f eye_pos = {0,0,10};
    float angle = 140.0;

    rst::rasterizer r(w, h);

    r.clear();
    r.set_model(get_model_matrix(angle));
    r.set_view(get_view_matrix(eye_pos));
    r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));
    
    log("please type the shader type... (normal / texture / bump / displacement / phong)");
    std::string shader_type = "normal";
    std::cin >> shader_type;

    std::string texture_path = "invalid";
    std::function<Eigen::Vector3f(fragment_shader_payload)> active_shader = phong_fragment_shader;

    if (shader_type == "texture"){
        std::cout << "Rasterizing using the texture shader\n";
        active_shader = texture_fragment_shader;
        log("please type the texture path...");
        texture_path = "../models/spot/hmap.jpg";
        std::cin >> texture_path;
    } else if (shader_type == "normal"){
        std::cout << "Rasterizing using the normal shader\n";
        active_shader = normal_fragment_shader;
    } else if (shader_type == "phong"){
        std::cout << "Rasterizing using the phong shader\n";
        active_shader = phong_fragment_shader;
    } else if (shader_type == "bump"){
        std::cout << "Rasterizing using the bump shader\n";
        active_shader = bump_fragment_shader;
        log("please type the texture path...");
        texture_path = "../models/spot/hmap.jpg";
        std::cin >> texture_path;
    } else if (shader_type == "displacement"){
        std::cout << "Rasterizing using the displacement shader\n";
        active_shader = displacement_fragment_shader;
        log("please type the texture path...");
        texture_path = "../models/spot/hmap.jpg";
        std::cin >> texture_path;
    }

    if(texture_path != "invalid")
        r.set_texture(Texture(texture_path));
    r.set_fragment_shader(active_shader);
    r.draw(TriangleList);

    cv::Mat image(w, h, CV_32FC3, r.frame_buf.data());
    image.convertTo(image, CV_8UC3, 1.0f);
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

    cv::imwrite(file_name, image);
}
