#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <cmath>

namespace AeroSim {
    /**
     * WGS-84 地球模型常数 (WGS-84 G1762 Standard)
     * 所有的计算必须基于此高精度常数集，以确保在长距离飞行下的经纬度闭合。
     */
    namespace Earth {
        // 几何参数 (Geometric Parameters)
        constexpr double A = 6378137.0;                      // 地球赤道半径 (m)
        constexpr double F = 1.0 / 298.257223563;            // 地球扁率 (Flattening)
        constexpr double B = A * (1.0 - F);                  // 地球极半径 (m)
        constexpr double E2 = F * (2.0 - F);                 // 第一偏心率平方 (Eccentricity squared)
        constexpr double E_PRIME2 = E2 / (1.0 - E2);         // 第二偏心率平方 (Second eccentricity squared)
        
        // 物理参数 (Physical Parameters)
        constexpr double MU = 3.986004418e14;                // 地心引力常数 (m^3/s^2)
        constexpr double OMEGA_E = 7.2921151467e-5;          // 地球自转角速度 (rad/s)
        constexpr double C = 2.99792458e8;                   // 真空光速 (m/s)
        
        // 摄动参数 (Perturbation Parameters)
        constexpr double J2 = 1.08262668355e-3;              // J2 项 (二级带谐系数)
        constexpr double J3 = -2.53265648533e-6;             // J3 项
        constexpr double J4 = -1.61962159137e-6;             // J4 项
    }

    /**
     * 基础物理与数学常数
     */
    namespace Phys {
        constexpr double G0 = 9.80665;                       // 标准重力加速度 (m/s^2)
        constexpr double R_GAS = 287.05287;                  // 干空气气体常数 (J/kg·K)
        constexpr double GAMMA = 1.4;                        // 空气绝热指数
        constexpr double STEFAN_BOLTZMANN = 5.670373e-8;     // 斯特藩-玻尔兹曼常数 (W/m^2·K^4)
    }

    namespace Math {
        constexpr double PI = 3.14159265358979323846;
        constexpr double DEG_TO_RAD = PI / 180.0;
        constexpr double RAD_TO_DEG = 180.0 / PI;
    }
}

#endif // CONSTANTS_HPP
