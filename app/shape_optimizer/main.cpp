#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>
#include <string>
#include <iomanip>
#include <chrono>
#include <utility>

#include "aero/panel/aero_solver.hpp"
// Removed cuda_runtime.h to avoid conflict/overhead
// #include <cuda_runtime.h>

// --- Math Constants & Helpers ---
const float PI = 3.14159265359f;
const float GAMMA = 1.4f;

float to_rad(float deg) { return deg * PI / 180.0f; }
float to_deg(float rad) { return rad * 180.0f / PI; }

// --- Taylor-Maccoll Solver ---
class TaylorMaccoll {
public:
    // Returns cone half-angle in degrees, and populates theta_r_map (theta_rad -> r_ratio)
    // r_ratio = r_local / r_shock
    static bool solve_flow(float mach, float beta_deg, float& cone_angle_deg, std::vector<std::pair<float, float>>& theta_r_map) {
        float beta = to_rad(beta_deg);
        float Mn1 = mach * std::sin(beta);
        if (Mn1 <= 1.0f) return false; // Shock detached/weak

        // V/V_max where V_max is limit speed into vacuum
        float V_inf_bar = mach / std::sqrt(2.0f/(GAMMA-1.0f) + mach*mach);
        
        float rho_ratio = ((GAMMA + 1) * Mn1 * Mn1) / ((GAMMA - 1) * Mn1 * Mn1 + 2);
        
        float Vn1 = V_inf_bar * std::sin(beta);
        float Vt1 = V_inf_bar * std::cos(beta);
        
        float Vn2 = Vn1 / rho_ratio;
        float Vt2 = Vt1;
        
        float V2 = std::sqrt(Vn2*Vn2 + Vt2*Vt2);
        
        // Deflection angle delta
        // tan(delta) = 2 cot(beta) (M^2 sin^2b - 1) / (M^2 (gamma + cos 2b) + 2)
        float num = 2.0f * (Mn1*Mn1 - 1.0f);
        float den = std::tan(beta) * (mach*mach * (GAMMA + std::cos(2*beta)) + 2.0f);
        float delta = std::atan(num/den);
        
        // Post-shock velocity at shock (theta = beta)
        // Vr = V2 cos(beta - delta)
        // Vtheta = -V2 sin(beta - delta)
        float Vr = V2 * std::cos(beta - delta);
        float Vtheta = -V2 * std::sin(beta - delta);
        
        theta_r_map.clear();
        theta_r_map.reserve(1000);
        theta_r_map.push_back({beta, 1.0f}); // At shock, r = r_shock
        
        float theta = beta;
        float step = -0.0005f; // Small step for accuracy (radians)
        
        float log_r_ratio = 0.0f;

        // Integration loop
        int max_steps = 50000;
        for(int i=0; i<max_steps; ++i) {
            auto derivatives = [&](float t, float vr, float vt) -> std::pair<float, float> {
                float v2 = vr*vr + vt*vt;
                float a2 = (GAMMA - 1.0f) / 2.0f * (1.0f - v2);
                float denom = vt*vt - a2;
                if (std::abs(denom) < 1e-8f) denom = 1e-8f * (denom < 0 ? -1 : 1);
                
                float dvr = vt;
                float dvt = -vr + (a2 * (vr + vt / std::tan(t))) / denom;
                return {dvr, dvt};
            };
            
            // RK4
            float k1_vr, k1_vt, k2_vr, k2_vt, k3_vr, k3_vt, k4_vr, k4_vt;
            auto d1 = derivatives(theta, Vr, Vtheta);
            k1_vr = d1.first; k1_vt = d1.second;
            
            auto d2 = derivatives(theta + step/2, Vr + k1_vr*step/2, Vtheta + k1_vt*step/2);
            k2_vr = d2.first; k2_vt = d2.second;
            
            auto d3 = derivatives(theta + step/2, Vr + k2_vr*step/2, Vtheta + k2_vt*step/2);
            k3_vr = d3.first; k3_vt = d3.second;
            
            auto d4 = derivatives(theta + step, Vr + k3_vr*step, Vtheta + k3_vt*step);
            k4_vr = d4.first; k4_vt = d4.second;
            
            float next_Vr = Vr + (step/6.0f)*(k1_vr + 2*k2_vr + 2*k3_vr + k4_vr);
            float next_Vtheta = Vtheta + (step/6.0f)*(k1_vt + 2*k2_vt + 2*k3_vt + k4_vt);
            float next_theta = theta + step;
            
            // Streamline integration: d(ln r)/dtheta = Vr/Vtheta
            // Be careful with Vtheta near 0
            float vt_safe = (std::abs(Vtheta) < 1e-6f) ? -1e-6f : Vtheta;
            float next_vt_safe = (std::abs(next_Vtheta) < 1e-6f) ? -1e-6f : next_Vtheta;
            
            float term1 = Vr / vt_safe;
            float term2 = next_Vr / next_vt_safe;
            
            log_r_ratio += 0.5f * (term1 + term2) * step;
            
            theta = next_theta;
            Vr = next_Vr;
            Vtheta = next_Vtheta;
            
            theta_r_map.push_back({theta, std::exp(log_r_ratio)});
            
            if (Vtheta >= 0.0f) {
                // Surface reached
                cone_angle_deg = to_deg(theta);
                return true;
            }
        }
        
        return false;
    }
};

// --- Geometry Generator ---
struct DesignParams {
    float nose_length;
    float body_length;
    float body_diameter;
    float fin_root_chord;
    float fin_tip_chord;
    float fin_span;
    float fin_sweep_angle; // degrees
};

class MissileGenerator {
public:
    static std::vector<AeroSim::Solver::Triangle> generate(const DesignParams& params) {
        std::vector<AeroSim::Solver::Triangle> mesh;
        
        float Ln = params.nose_length;
        float Lb = params.body_length;
        float D = params.body_diameter;
        float R = D / 2.0f;
        
        // --- 1. Nose Cone (Tangent Ogive) ---
        int segments = 32;
        int rings = 20;
        
        // Ogive radius of curvature (rho)
        // Tangent Ogive Logic
        float rho = (R*R + Ln*Ln) / (2.0f * R);
        
        std::vector<float3> ring_prev;
        // Tip
        float3 tip = {0, 0, 0};
        
        // Generate nose rings
        for (int i = 1; i <= rings; ++i) {
            float t = (float)i / rings;
            float x = t * Ln;
            
            // Tangent Ogive Equation: r(x) = sqrt(rho^2 - (Ln - x)^2) + R - rho
            // Where x is from 0 to Ln.
            float term = Ln - x;
            float r = std::sqrt(rho*rho - term*term) + R - rho;
            
            std::vector<float3> ring_curr;
            for (int j = 0; j < segments; ++j) {
                float theta = (float)j / segments * 2.0f * 3.14159f;
                float y = r * std::cos(theta);
                float z = r * std::sin(theta);
                ring_curr.push_back({x, y, z});
            }
            
            if (i == 1) {
                // Connect tip to first ring
                for (int j = 0; j < segments; ++j) {
                    int next_j = (j + 1) % segments;
                    add_tri(mesh, tip, ring_curr[j], ring_curr[next_j]);
                }
            } else {
                // Connect prev ring to curr ring
                for (int j = 0; j < segments; ++j) {
                    int next_j = (j + 1) % segments;
                    // Two triangles (quad)
                    add_tri(mesh, ring_prev[j], ring_curr[j], ring_curr[next_j]);
                    add_tri(mesh, ring_prev[j], ring_curr[next_j], ring_prev[next_j]);
                }
            }
            ring_prev = ring_curr;
        }
        
        // --- 2. Cylindrical Body ---
        int body_rings = 10;
        for (int i = 1; i <= body_rings; ++i) {
            float t = (float)i / body_rings;
            float x = Ln + t * Lb;
            
            std::vector<float3> ring_curr;
            for (int j = 0; j < segments; ++j) {
                float theta = (float)j / segments * 2.0f * 3.14159f;
                float y = R * std::cos(theta);
                float z = R * std::sin(theta);
                ring_curr.push_back({x, y, z});
            }
            
            // Connect prev ring to curr ring
            for (int j = 0; j < segments; ++j) {
                int next_j = (j + 1) % segments;
                add_tri(mesh, ring_prev[j], ring_curr[j], ring_curr[next_j]);
                add_tri(mesh, ring_prev[j], ring_curr[next_j], ring_prev[next_j]);
            }
            ring_prev = ring_curr;
        }
        
        // --- 3. Base Closure ---
        float3 center_base = {Ln + Lb, 0, 0};
        for (int j = 0; j < segments; ++j) {
            int next_j = (j + 1) % segments;
            // Base normal points +X, so vertices order: center, next, curr (or check winding)
            // Actually normal should point +X.
            // Current winding is CCW looking from outside?
            // Let's stick to CCW.
            add_tri(mesh, center_base, ring_prev[next_j], ring_prev[j]);
        }
        
        // --- 4. Fins (Cruciform) ---
        // 4 fins at 0, 90, 180, 270 degrees
        float fin_x_start = Ln + Lb - params.fin_root_chord;
        
        for (int k = 0; k < 4; ++k) {
            float angle = k * 3.14159f / 2.0f;
            float ca = std::cos(angle);
            float sa = std::sin(angle);
            
            float rc = params.fin_root_chord;
            float tc = params.fin_tip_chord;
            float span = params.fin_span;
            float sweep = params.fin_sweep_angle * 3.14159f / 180.0f;
            float tip_x_offset = span * std::tan(sweep);
            
            // Fin defined in local (x, y) plane then rotated
            // Root LE: (fin_x_start, R)
            // Root TE: (fin_x_start + rc, R)
            // Tip LE: (fin_x_start + tip_x_offset, R + span)
            // Tip TE: (fin_x_start + tip_x_offset + tc, R + span)
            
            float3 r_le = {fin_x_start, R, 0};
            float3 r_te = {fin_x_start + rc, R, 0};
            float3 t_le = {fin_x_start + tip_x_offset, R + span, 0};
            float3 t_te = {fin_x_start + tip_x_offset + tc, R + span, 0};
            
            auto rotate = [&](float3 p) -> float3 {
                return {p.x, p.y * ca - p.z * sa, p.y * sa + p.z * ca};
            };
            
            // Thickness (simplified as thin plate for now, or small thickness)
            // Let's make it a thin surface (two triangles) for now, or a thin wedge
            // To be robust, let's make it a thin wedge (diamond airfoil)
            float thickness = 0.1f; 
            
            // Upper surface
            add_tri(mesh, rotate(r_le), rotate(t_le), rotate(t_te));
            add_tri(mesh, rotate(r_le), rotate(t_te), rotate(r_te));
            
            // Lower surface (reversed winding) - assuming zero thickness for simple aero
            // But for volume, we need thickness.
            // Let's just add thickness later or ignore fin volume.
            // For now, let's treat fins as zero-thickness surfaces for aero code (if it supports it)
            // OR make a thin diamond.
            
            // Let's do double-sided triangles for "thin fin" visual
             add_tri(mesh, rotate(r_le), rotate(r_te), rotate(t_te)); // Reverse
             add_tri(mesh, rotate(r_le), rotate(t_te), rotate(t_le)); // Reverse
        }
        
        return mesh;
    }

private:
    static void add_tri(std::vector<AeroSim::Solver::Triangle>& mesh, float3 v0, float3 v1, float3 v2) {
        AeroSim::Solver::Triangle tri;
        tri.v0 = v0; tri.v1 = v1; tri.v2 = v2;
        
        // Compute Center
        tri.center = make_float3((v0.x+v1.x+v2.x)/3, (v0.y+v1.y+v2.y)/3, (v0.z+v1.z+v2.z)/3);
        
        // Compute Normal
        float3 e1 = make_float3(v1.x-v0.x, v1.y-v0.y, v1.z-v0.z);
        float3 e2 = make_float3(v2.x-v0.x, v2.y-v0.y, v2.z-v0.z);
        float3 n = make_float3(
            e1.y*e2.z - e1.z*e2.y,
            e1.z*e2.x - e1.x*e2.z,
            e1.x*e2.y - e1.y*e2.x
        );
        float len = std::sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
        if (len > 1e-6f) {
            tri.normal = make_float3(n.x/len, n.y/len, n.z/len);
            tri.area = 0.5f * len;
        } else {
            tri.normal = make_float3(0,0,1);
            tri.area = 0;
        }
        
        mesh.push_back(tri);
    }
};

// --- Optimizer ---
class ShapeOptimizer {
    AeroSim::Solver::AeroSolver solver;
    std::mt19937 rng;
    
public:
    ShapeOptimizer() {
        rng.seed(std::random_device{}());
    }
    
    float calculate_volume(const std::vector<AeroSim::Solver::Triangle>& mesh) {
        float vol = 0.0f;
        for(const auto& tri : mesh) {
             // Signed volume of tetrahedron from origin to triangle
             // v = 1/6 * det(p1, p2, p3)
             // = 1/6 * dot(p1, cross(p2, p3))
             float3 p1 = tri.v0;
             float3 p2 = tri.v1;
             float3 p3 = tri.v2;
             
             // Cross p2, p3
             float cx = p2.y*p3.z - p2.z*p3.y;
             float cy = p2.z*p3.x - p2.x*p3.z;
             float cz = p2.x*p3.y - p2.y*p3.x;
             
             float dot = p1.x*cx + p1.y*cy + p1.z*cz;
             vol += dot;
        }
        return std::abs(vol) / 6.0f; 
    }
    
    float evaluate(const DesignParams& p) {
        auto mesh = MissileGenerator::generate(p);
        
        if (mesh.empty()) return 1000.0f;
        
        // Volume Check (Cylinder + Nose approx)
        // V_cyl = pi * R^2 * Lb
        // V_nose approx 0.5 * pi * R^2 * Ln (for Ogive)
        // Calculate exact from mesh to be sure
        float vol = calculate_volume(mesh);
        
        // Target Volume: at least 12.0 m^3 for an IRBM (approx 15 tons density ~1.2)
        if (vol < 12.0f) {
            return 500.0f + (12.0f - vol) * 100.0f; // Penalty
        }
        
        // Geometric constraints
        // Fineness Ratio (L/D) should be reasonable (e.g. 8-15 for hypersonic)
        float total_length = p.nose_length + p.body_length;
        float fineness = total_length / p.body_diameter;
        if (fineness < 8.0f || fineness > 18.0f) {
            return 200.0f + std::abs(fineness - 12.0f) * 10.0f;
        }
        
        // Nose Fineness Ratio Check (Ln/D) - prevent too sharp nose
        // Tactical missiles typically have Ln/D between 2 and 4. 
        // Hypersonic/IRBM can be sharper (3-5).
        float nose_fineness = p.nose_length / p.body_diameter;
        if (nose_fineness > 5.0f) {
            return 300.0f + (nose_fineness - 5.0f) * 50.0f;
        }

        // Fin span constraint relative to body diameter
        // Span shouldn't be more than 1.5x diameter for typical missiles
        if (p.fin_span > p.body_diameter * 1.5f) {
            return 300.0f + (p.fin_span - p.body_diameter * 1.5f) * 100.0f;
        }
        
        // Ref Area (Cross section area or Planform?)
        // Missile Aero usually uses Cross Section Area (pi * R^2) as reference.
        // But the solver might expect Planform.
        // Let's use Planform for L/D calculation consistency if solver uses it.
        // Solver uses whatever we pass.
        // Let's pass Planform Area = D * L_total + Fin_Area
        float planform_area = p.body_diameter * total_length + 2.0f * 0.5f * (p.fin_root_chord + p.fin_tip_chord) * p.fin_span;
        float ref_len = total_length;
        float ref_span = p.fin_span * 2.0f + p.body_diameter;
        
        if (!solver.load_mesh(mesh, planform_area, ref_len, ref_span)) return 1000.0f;
        
        // Evaluate at design point (Mach 15, Alpha 5)
        auto coeffs = solver.compute_coefficients(15.0f, 5.0f, 0.0f);
        
        if (std::isnan(coeffs.L_D)) return 1000.0f;
        
        // Optimization Objective:
        // Maximize L/D
        // Note: Standard missiles have low L/D. 
        // If we want a "Glide" missile, we want higher L/D.
        return -coeffs.L_D;
    }
    
    DesignParams optimize() {
        // Differential Evolution
        int pop_size = 50;
        int generations = 30;
        float CR = 0.9f;
        float F = 0.6f;
        
        std::vector<DesignParams> pop(pop_size);
        std::vector<float> scores(pop_size);
        
        // Init Bounds - Tighter for IRBM
        std::uniform_real_distribution<float> dist_Ln(2.0, 4.0); // Nose Length
        std::uniform_real_distribution<float> dist_Lb(8.0, 12.0); // Body Length
        std::uniform_real_distribution<float> dist_D(1.0, 1.5);  // Diameter
        std::uniform_real_distribution<float> dist_Frc(1.0, 3.0); // Fin Root Chord
        std::uniform_real_distribution<float> dist_Ftc(0.2, 0.8); // Fin Tip Chord
        std::uniform_real_distribution<float> dist_Fs(0.5, 1.2);  // Fin Span
        std::uniform_real_distribution<float> dist_Fsw(30.0, 60.0); // Fin Sweep
        
        for(int i=0; i<pop_size; ++i) {
            pop[i] = {dist_Ln(rng), dist_Lb(rng), dist_D(rng), dist_Frc(rng), dist_Ftc(rng), dist_Fs(rng), dist_Fsw(rng)};
            scores[i] = evaluate(pop[i]);
        }
        
        std::uniform_real_distribution<float> rand01(0.0f, 1.0f);
        std::uniform_int_distribution<int> rand_idx(0, pop_size-1);
        
        std::cout << "Starting Missile Shape Optimization..." << std::endl;
        
        for(int gen=0; gen<generations; ++gen) {
            float best_score = 1e9f;
            int best_idx = 0;
            for(int i=0; i<pop_size; ++i) {
                if(scores[i] < best_score) {
                    best_score = scores[i];
                    best_idx = i;
                }
            }
            
            std::cout << "Gen " << gen+1 << ": Best L/D = " << -best_score 
                      << " [Ln=" << pop[best_idx].nose_length 
                      << " Lb=" << pop[best_idx].body_length 
                      << " D=" << pop[best_idx].body_diameter << "]" << std::endl;
                      
            for(int i=0; i<pop_size; ++i) {
                int a = rand_idx(rng);
                int b = rand_idx(rng);
                int c = rand_idx(rng);
                while(a==i) a = rand_idx(rng);
                while(b==i || b==a) b = rand_idx(rng);
                while(c==i || c==a || c==b) c = rand_idx(rng);
                
                DesignParams mutant = pop[i];
                int R = std::uniform_int_distribution<int>(0, 6)(rng);
                
                auto mutate = [&](float x, float y, float z) { return x + F * (y - z); };
                
                if(rand01(rng)<CR || R==0) mutant.nose_length = mutate(pop[a].nose_length, pop[b].nose_length, pop[c].nose_length);
                if(rand01(rng)<CR || R==1) mutant.body_length = mutate(pop[a].body_length, pop[b].body_length, pop[c].body_length);
                if(rand01(rng)<CR || R==2) mutant.body_diameter = mutate(pop[a].body_diameter, pop[b].body_diameter, pop[c].body_diameter);
                if(rand01(rng)<CR || R==3) mutant.fin_root_chord = mutate(pop[a].fin_root_chord, pop[b].fin_root_chord, pop[c].fin_root_chord);
                if(rand01(rng)<CR || R==4) mutant.fin_tip_chord = mutate(pop[a].fin_tip_chord, pop[b].fin_tip_chord, pop[c].fin_tip_chord);
                if(rand01(rng)<CR || R==5) mutant.fin_span = mutate(pop[a].fin_span, pop[b].fin_span, pop[c].fin_span);
                if(rand01(rng)<CR || R==6) mutant.fin_sweep_angle = mutate(pop[a].fin_sweep_angle, pop[b].fin_sweep_angle, pop[c].fin_sweep_angle);
                
                // Clamp
                mutant.nose_length = std::max(1.0f, std::min(5.0f, mutant.nose_length));
                mutant.body_length = std::max(6.0f, std::min(15.0f, mutant.body_length));
                mutant.body_diameter = std::max(0.8f, std::min(1.8f, mutant.body_diameter));
                mutant.fin_root_chord = std::max(0.5f, std::min(3.0f, mutant.fin_root_chord));
                mutant.fin_tip_chord = std::max(0.1f, std::min(1.5f, mutant.fin_tip_chord));
                mutant.fin_span = std::max(0.3f, std::min(2.0f, mutant.fin_span));
                mutant.fin_sweep_angle = std::max(10.0f, std::min(70.0f, mutant.fin_sweep_angle));
                
                float new_score = evaluate(mutant);
                if(new_score < scores[i]) {
                    scores[i] = new_score;
                    pop[i] = mutant;
                }
            }
        }
        
        int best_idx = 0;
        for(int i=1; i<pop_size; ++i) if(scores[i] < scores[best_idx]) best_idx = i;
        return pop[best_idx];
    }
    
    void save_stl(const DesignParams& p, const std::string& filename) {
        auto mesh = MissileGenerator::generate(p);
        std::ofstream file(filename, std::ios::binary);
        
        // Header
        char header[80] = "Missile Model";
        file.write(header, 80);
        uint32_t num_tris = mesh.size();
        file.write((char*)&num_tris, 4);
        
        for(const auto& tri : mesh) {
            float n[3] = {tri.normal.x, tri.normal.y, tri.normal.z};
            float v1[3] = {tri.v0.x, tri.v0.y, tri.v0.z};
            float v2[3] = {tri.v1.x, tri.v1.y, tri.v1.z};
            float v3[3] = {tri.v2.x, tri.v2.y, tri.v2.z};
            uint16_t attr = 0;
            
            file.write((char*)n, 12);
            file.write((char*)v1, 12);
            file.write((char*)v2, 12);
            file.write((char*)v3, 12);
            file.write((char*)&attr, 2);
        }
    }
    
    void save_config(const DesignParams& p, const std::string& filename) {
        std::ofstream file(filename);
        file << "{\n";
        file << "    \"nose_length\": " << p.nose_length << ",\n";
        file << "    \"body_length\": " << p.body_length << ",\n";
        file << "    \"body_diameter\": " << p.body_diameter << ",\n";
        file << "    \"fin_root_chord\": " << p.fin_root_chord << ",\n";
        file << "    \"fin_tip_chord\": " << p.fin_tip_chord << ",\n";
        file << "    \"fin_span\": " << p.fin_span << ",\n";
        file << "    \"fin_sweep_angle\": " << p.fin_sweep_angle << "\n";
        file << "}\n";
    }
};

int main() {
    ShapeOptimizer optimizer;
    auto start = std::chrono::high_resolution_clock::now();
    
    DesignParams best = optimizer.optimize();
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    
    std::cout << "Optimization Finished in " << duration.count() << "s" << std::endl;
    std::cout << "Best Parameters:" << std::endl;
    std::cout << "  Nose Length: " << best.nose_length << std::endl;
    std::cout << "  Body Length: " << best.body_length << std::endl;
    std::cout << "  Diameter: " << best.body_diameter << std::endl;
    std::cout << "  Fin Span: " << best.fin_span << std::endl;
    
    optimizer.save_stl(best, "hgv_model_optimized.stl");
    optimizer.save_config(best, "hgv_config_optimized.json");
    
    return 0;
}
