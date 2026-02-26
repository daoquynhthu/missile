#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace AeroSim {
    namespace Utils {
        
        /**
         * @brief Linear Interpolation for 1D data
         * Assumes x is sorted. Extrapolates using end values (clamps).
         */
        inline double interpolate_1d(const std::vector<double>& x, const std::vector<double>& y, double val) {
            if (x.empty()) return 0.0;
            if (x.size() != y.size()) return 0.0; // Error handling
            
            if (val <= x.front()) return y.front();
            if (val >= x.back()) return y.back();

            auto it = std::lower_bound(x.begin(), x.end(), val);
            size_t i = std::distance(x.begin(), it);
            if (i == 0) return y.front();
            
            double x1 = x[i-1];
            double x2 = x[i];
            double y1 = y[i-1];
            double y2 = y[i];
            
            // Avoid division by zero
            if (std::abs(x2 - x1) < 1e-9) return y1;
            
            return y1 + (val - x1) * (y2 - y1) / (x2 - x1);
        }

        /**
         * @brief Bilinear Interpolation for 2D data (Regular Grid)
         * @param x_grid Row coordinates (Mach) - must be sorted
         * @param y_grid Column coordinates (Alpha) - must be sorted
         * @param z_data Data table (flattened row-major: z[row][col] -> z[row * n_cols + col])
         * @param x_val Query x (Mach)
         * @param y_val Query y (Alpha)
         */
        inline double interpolate_2d(
            const std::vector<double>& x_grid, 
            const std::vector<double>& y_grid, 
            const std::vector<double>& z_data, 
            double x_val, 
            double y_val
        ) {
            if (x_grid.empty() || y_grid.empty() || z_data.empty()) return 0.0;
            size_t nx = x_grid.size();
            size_t ny = y_grid.size();
            if (z_data.size() != nx * ny) return 0.0; // Error

            // Clamp inputs
            if (x_val <= x_grid.front()) x_val = x_grid.front();
            if (x_val >= x_grid.back()) x_val = x_grid.back();
            if (y_val <= y_grid.front()) y_val = y_grid.front();
            if (y_val >= y_grid.back()) y_val = y_grid.back();

            // Find indices
            auto it_x = std::lower_bound(x_grid.begin(), x_grid.end(), x_val);
            size_t i = std::distance(x_grid.begin(), it_x);
            if (i > 0) i--; // Ensure i is the lower index
            if (i >= nx - 1) i = nx - 2;

            auto it_y = std::lower_bound(y_grid.begin(), y_grid.end(), y_val);
            size_t j = std::distance(y_grid.begin(), it_y);
            if (j > 0) j--; // Ensure j is the lower index
            if (j >= ny - 1) j = ny - 2;

            double x1 = x_grid[i];
            double x2 = x_grid[i+1];
            double y1 = y_grid[j];
            double y2 = y_grid[j+1];

            // Values at corners
            double q11 = z_data[i * ny + j];     // (x1, y1)
            double q12 = z_data[i * ny + j + 1]; // (x1, y2)
            double q21 = z_data[(i + 1) * ny + j]; // (x2, y1)
            double q22 = z_data[(i + 1) * ny + j + 1]; // (x2, y2)

            // Interpolate in X direction
            double r1 = ((x2 - x_val) / (x2 - x1)) * q11 + ((x_val - x1) / (x2 - x1)) * q21;
            double r2 = ((x2 - x_val) / (x2 - x1)) * q12 + ((x_val - x1) / (x2 - x1)) * q22;

            // Interpolate in Y direction
            double p = ((y2 - y_val) / (y2 - y1)) * r1 + ((y_val - y1) / (y2 - y1)) * r2;

            return p;
        }

    }
}
