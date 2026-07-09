#include "sim/coord/coordinate_transform.hpp"
#include <cmath>

namespace AeroSim {

__host__ __device__ Eigen::Vector3d CoordinateTransform::lla_to_ecef(const LLA& lla) {
    double sin_lat = sin(lla.lat);
    double cos_lat = cos(lla.lat);
    double sin_lon = sin(lla.lon);
    double cos_lon = cos(lla.lon);

    double N = Earth::A() / sqrt(1.0 - Earth::E2() * sin_lat * sin_lat);
    
    double x = (N + lla.alt) * cos_lat * cos_lon;
    double y = (N + lla.alt) * cos_lat * sin_lon;
    double z = (N * (1.0 - Earth::E2()) + lla.alt) * sin_lat;

    return Eigen::Vector3d(x, y, z);
}

__host__ __device__ LLA CoordinateTransform::ecef_to_lla(const Eigen::Vector3d& ecef) {
    double x = ecef.x();
    double y = ecef.y();
    double z = ecef.z();

    double p = sqrt(x * x + y * y);
    double lon = atan2(y, x);
    
    // Initial guess
    double lat = atan2(z, p * (1.0 - Earth::E2()));
    double alt = 0.0;
    double N = 0.0;

    // Iterative approach (usually converges in 5-6 iterations)
    for (int i = 0; i < 5; ++i) {
        double sin_lat = sin(lat);
        N = Earth::A() / sqrt(1.0 - Earth::E2() * sin_lat * sin_lat);
        alt = p / cos(lat) - N;
        lat = atan2(z, p * (1.0 - Earth::E2() * (N / (N + alt))));
    }

    return LLA{lat, lon, alt};
}

__host__ __device__ Eigen::Vector3d CoordinateTransform::ecef_to_ned(const Eigen::Vector3d& ecef, const LLA& ref_lla) {
    Eigen::Vector3d ref_ecef = lla_to_ecef(ref_lla);
    Eigen::Vector3d delta_ecef = ecef - ref_ecef;
    return ecef_to_ned_vector(delta_ecef, ref_lla);
}

__host__ __device__ Eigen::Vector3d CoordinateTransform::ecef_to_ned_vector(const Eigen::Vector3d& ecef_vec, const LLA& ref_lla) {
    double sin_lat = sin(ref_lla.lat);
    double cos_lat = cos(ref_lla.lat);
    double sin_lon = sin(ref_lla.lon);
    double cos_lon = cos(ref_lla.lon);

    // Rotation matrix from ECEF to NED
    Eigen::Matrix3d R;
    R(0, 0) = -sin_lat * cos_lon;
    R(0, 1) = -sin_lat * sin_lon;
    R(0, 2) = cos_lat;
    R(1, 0) = -sin_lon;
    R(1, 1) = cos_lon;
    R(1, 2) = 0.0;
    R(2, 0) = -cos_lat * cos_lon;
    R(2, 1) = -cos_lat * sin_lon;
    R(2, 2) = -sin_lat;

    return R * ecef_vec;
}

__host__ __device__ Eigen::Vector3d CoordinateTransform::ned_to_ecef(const Eigen::Vector3d& ned, const LLA& ref_lla) {
    double sin_lat = sin(ref_lla.lat);
    double cos_lat = cos(ref_lla.lat);
    double sin_lon = sin(ref_lla.lon);
    double cos_lon = cos(ref_lla.lon);

    // Rotation matrix from NED to ECEF (transpose of ECEF to NED)
    Eigen::Matrix3d R;
    R(0, 0) = -sin_lat * cos_lon;
    R(1, 0) = -sin_lat * sin_lon;
    R(2, 0) = cos_lat;
    R(0, 1) = -sin_lon;
    R(1, 1) = cos_lon;
    R(2, 1) = 0.0;
    R(0, 2) = -cos_lat * cos_lon;
    R(1, 2) = -cos_lat * sin_lon;
    R(2, 2) = -sin_lat;

    Eigen::Vector3d ref_ecef = lla_to_ecef(ref_lla);
    return R * ned + ref_ecef;
}

__host__ __device__ Eigen::Matrix3d CoordinateTransform::dcm_ned_to_body(double roll, double pitch, double yaw) {
    double s_r = sin(roll);
    double c_r = cos(roll);
    double s_p = sin(pitch);
    double c_p = cos(pitch);
    double s_y = sin(yaw);
    double c_y = cos(yaw);

    // NED to Body (Standard Z-Y-X rotation sequence: Yaw -> Pitch -> Roll)
    Eigen::Matrix3d dcm;
    dcm(0, 0) = c_p * c_y;
    dcm(0, 1) = c_p * s_y;
    dcm(0, 2) = -s_p;
    dcm(1, 0) = s_r * s_p * c_y - c_r * s_y;
    dcm(1, 1) = s_r * s_p * s_y + c_r * c_y;
    dcm(1, 2) = s_r * c_p;
    dcm(2, 0) = c_r * s_p * c_y + s_r * s_y;
    dcm(2, 1) = c_r * s_p * s_y - s_r * c_y;
    dcm(2, 2) = c_r * c_p;

    return dcm;
}

__host__ __device__ Eigen::Matrix3d CoordinateTransform::dcm_body_to_ned(double roll, double pitch, double yaw) {
    return dcm_ned_to_body(roll, pitch, yaw).transpose();
}

__host__ __device__ Eigen::Vector3d CoordinateTransform::rotate_vector(const Eigen::Matrix3d& dcm, const Eigen::Vector3d& vec) {
    return dcm * vec;
}

__host__ __device__ Eigen::Vector3d CoordinateTransform::ecef_to_eci(const Eigen::Vector3d& ecef, double gmst) {
    double s = sin(gmst);
    double c = cos(gmst);
    
    Eigen::Matrix3d R;
    R << c, -s, 0,
         s,  c, 0,
         0,  0, 1;
    
    return R * ecef;
}

__host__ __device__ Eigen::Vector3d CoordinateTransform::eci_to_ecef(const Eigen::Vector3d& eci, double gmst) {
    double s = sin(gmst);
    double c = cos(gmst);
    
    Eigen::Matrix3d R;
    R <<  c, s, 0,
         -s, c, 0,
          0, 0, 1;
    
    return R * eci;
}

} // namespace AeroSim
