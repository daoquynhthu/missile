#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <cmath>

namespace aerosp {

class ProgressBar {
public:
    ProgressBar(size_t total_steps, size_t bar_width = 50, std::string prefix = "Progress")
        : m_total_steps(total_steps), m_bar_width(bar_width), m_prefix(prefix) {
        m_start_time = std::chrono::steady_clock::now();
        m_last_update_time = m_start_time;
        m_current_step = 0;
    }

    void update(size_t current_step) {
        m_current_step = current_step;
        
        // Throttle updates to avoid console flickering (e.g., every 100ms)
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - m_last_update_time).count() < 100 && current_step < m_total_steps) {
            return;
        }
        m_last_update_time = now;
        
        float progress = (float)m_current_step / m_total_steps;
        if (progress > 1.0f) progress = 1.0f;
        
        size_t filled_width = (size_t)(progress * m_bar_width);
        
        // Calculate ETA
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - m_start_time).count();
        long long eta = 0;
        // Print
        std::cout << "\r" << m_prefix << " [";
        for (size_t i = 0; i < m_bar_width; ++i) {
            if (i < filled_width) std::cout << "=";
            else if (i == filled_width) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << "%";

        if (progress > 0.001f) {
            eta = (long long)(elapsed / progress) - elapsed;
            
            // Format ETA
            int eta_h = eta / 3600;
            int eta_m = (eta % 3600) / 60;
            int eta_s = eta % 60;
            
            std::cout << " | ETA: ";
            if (eta_h > 0) std::cout << eta_h << "h ";
            if (eta_m > 0) std::cout << eta_m << "m ";
            std::cout << eta_s << "s";
        } else {
            std::cout << " | ETA: --";
        }
        
        // Clear remaining characters on the line
        std::cout << "      "; 

        
        std::cout << std::flush;
    }

    void finish() {
        update(m_total_steps);
        std::cout << std::endl;
    }

private:
    size_t m_total_steps;
    size_t m_bar_width;
    size_t m_current_step;
    std::string m_prefix;
    std::chrono::steady_clock::time_point m_start_time;
    std::chrono::steady_clock::time_point m_last_update_time;
};

} // namespace aerosp
