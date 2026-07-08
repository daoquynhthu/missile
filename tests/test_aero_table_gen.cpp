#include "aero_solver/aero_solver.hpp"
#include "aero_cfd/cfd_solver.hpp"
#include "aero_cfd/cfd_mesh.hpp"
#include "aero_cfd/cfd_config.hpp"
#include "aero_cfd/cfd_result.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdio>
#include <vector>
#include <string>

using namespace AeroSim::Solver;
namespace Cfd = AeroSim::Cfd;

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { \
    ++test_count; \
    std::cout << "[" << (name) << "]\n"; \
} while(0)

#define FAIL(...) do { \
    char buf[512]; snprintf(buf, sizeof(buf), __VA_ARGS__); \
    std::cerr << "FAIL: " << buf << "\n"; \
    return 1; \
} while(0)

#define PASS do { ++pass_count; } while(0)

// RAII temp file guard
struct TempFile {
    std::string path;
    explicit TempFile(const std::string& p) : path(p) { std::remove(path.c_str()); }
    ~TempFile() { std::remove(path.c_str()); }
};

// ─── Helper: read CSV into rows ─────────────────────────────────────────
struct CsvRow {
    double mach, alpha, beta;
    double CX, CY, CZ, CL, CD, L_D, Cl, Cm, Cn;
    std::string fidelity;
};

static bool read_csv(const std::string& path, std::vector<CsvRow>& rows,
                     bool has_fidelity, std::string* error) {
    std::ifstream f(path);
    if (!f.is_open()) {
        if (error) *error = "cannot open " + path;
        return false;
    }
    std::string line;
    if (!std::getline(f, line)) {
        if (error) *error = "empty file";
        return false;
    }
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string tok;
        CsvRow r;
        auto next = [&](double& v) {
            if (!std::getline(ss, tok, ',')) return false;
            v = std::stod(tok);
            return true;
        };
        // CSV format: Mach,Alpha,Beta,CX,CY,CZ,CL,CD,L_D,Cl,Cm,Cn[,Fidelity]
        if (!next(r.mach) || !next(r.alpha) || !next(r.beta) ||
            !next(r.CX) || !next(r.CY) || !next(r.CZ) ||
            !next(r.CL) || !next(r.CD) || !next(r.L_D) ||
            !next(r.Cl) || !next(r.Cm) || !next(r.Cn))
            return false;
        if (has_fidelity) {
            std::getline(ss, r.fidelity, ',');
            // trim whitespace
            if (!r.fidelity.empty() && r.fidelity.back() == '\r') r.fidelity.pop_back();
        } else {
            r.fidelity = "newtonian";
        }
        rows.push_back(r);
    }
    return true;
}

// ─── Test: CFD table in-range ────────────────────────────────────────────
static int test_cfd_gpu_table_in_range() {
    TEST("TABLE-CFD-1 CFD table generation with 3x3 in-range grid");
    TempFile csv("test_table_cfd_gpu.csv");

    AeroTableConfig cfg;
    cfg.ref_area   = 1.131f;
    cfg.ref_length = 12.0f;
    cfg.ref_span   = 3.0f;
    cfg.com_x      = 6.0f;
    cfg.use_fvm    = true;
    cfg.mesh_subdivisions = 125;  // rapid: n=5, ~320 tets

    std::vector<double> mach   = {2.0, 4.0, 6.0};
    std::vector<double> alpha  = {0.0, 5.0, 10.0};
    std::vector<double> beta   = {0.0};

    if (!generate_aero_table(
            "data/missile/hgv_model_optimized.stl",
            csv.path, mach, alpha, beta, cfg))
        FAIL("generate_aero_table returned false for 3x3 in-range grid");

    std::vector<CsvRow> rows;
    std::string err;
    if (!read_csv(csv.path, rows, true, &err))
        FAIL("read_csv failed: %s", err.c_str());
    if (rows.size() != 9)
        FAIL("expected 9 rows, got %zu", rows.size());

    for (size_t i = 0; i < rows.size(); ++i) {
        auto& r = rows[i];
        if (r.fidelity != "cfd-gpu")
            FAIL("row %zu: fidelity='%s' expected 'cfd-gpu'", i, r.fidelity.c_str());
        if (!std::isfinite(r.CX) || !std::isfinite(r.CY) || !std::isfinite(r.CZ) ||
            !std::isfinite(r.CL) || !std::isfinite(r.CD) ||
            !std::isfinite(r.Cl) || !std::isfinite(r.Cm) || !std::isfinite(r.Cn))
            FAIL("row %zu: non-finite force (CX=%g CY=%g CZ=%g)", i, r.CX, r.CY, r.CZ);
        // beta=0 symmetry: expect CY, Cl, Cn near zero
        if (std::abs(r.CY) > 1e-2 || std::abs(r.Cl) > 1e-2 || std::abs(r.Cn) > 1e-2)
            FAIL("row %zu: symmetry violation beta=0: CY=%g Cl=%g Cn=%g (tol=1e-2)", i, r.CY, r.Cl, r.Cn);
    }

    std::cout << "PASS: " << rows.size() << " rows, all finite, symmetry holds\n";
    PASS;
    return 0;
}

// ─── Test: out-of-range → rejection ──────────────────────────────────────
static int test_cfd_gpu_out_of_range() {
    TEST("TABLE-CFD-2 CFD table rejects out-of-range conditions");
    AeroTableConfig cfg;
    cfg.ref_area   = 1.131f;
    cfg.ref_length = 12.0f;
    cfg.ref_span   = 3.0f;
    cfg.com_x      = 6.0f;
    cfg.use_fvm    = true;

    // Test each boundary: Mach < 1.2, |alpha| > 30, |beta| > 10, Mach > 30

    // (a) Mach=0.5 below minimum
    {
        TempFile csv("test_cfd_out_mach_low.csv");
        bool ok = generate_aero_table("data/missile/hgv_model_optimized.stl",
            csv.path, {0.5}, {0.0}, {0.0}, cfg);
        if (ok) FAIL("Mach=0.5 should be rejected (below 1.2)");
    }

    // (b) alpha=31 above maximum
    {
        TempFile csv("test_cfd_out_alpha_high.csv");
        bool ok = generate_aero_table("data/missile/hgv_model_optimized.stl",
            csv.path, {4.0}, {31.0}, {0.0}, cfg);
        if (ok) FAIL("alpha=31 should be rejected (above 30)");
    }

    // (c) alpha=-31 below minimum
    {
        TempFile csv("test_cfd_out_alpha_low.csv");
        bool ok = generate_aero_table("data/missile/hgv_model_optimized.stl",
            csv.path, {4.0}, {-31.0}, {0.0}, cfg);
        if (ok) FAIL("alpha=-31 should be rejected (below -30)");
    }

    // (d) beta=11 above maximum
    {
        TempFile csv("test_cfd_out_beta_high.csv");
        bool ok = generate_aero_table("data/missile/hgv_model_optimized.stl",
            csv.path, {4.0}, {0.0}, {11.0}, cfg);
        if (ok) FAIL("beta=11 should be rejected (above 10)");
    }

    // (e) beta=-11 below minimum
    {
        TempFile csv("test_cfd_out_beta_low.csv");
        bool ok = generate_aero_table("data/missile/hgv_model_optimized.stl",
            csv.path, {4.0}, {0.0}, {-11.0}, cfg);
        if (ok) FAIL("beta=-11 should be rejected (below -10)");
    }

    // (f) Mach=31 above maximum
    {
        TempFile csv("test_cfd_out_mach_high.csv");
        bool ok = generate_aero_table("data/missile/hgv_model_optimized.stl",
            csv.path, {31.0}, {0.0}, {0.0}, cfg);
        if (ok) FAIL("Mach=31 should be rejected (above 30)");
    }

    std::cout << "PASS: all 6 out-of-range conditions correctly rejected\n";
    PASS;
    return 0;
}

// ─── Test: Newtonian baseline unchanged ──────────────────────────────────
static int test_newtonian_baseline() {
    TEST("TABLE-CFD-3 Newtonian baseline still works (use_fvm=false)");
    TempFile csv("test_table_newtonian.csv");

    AeroTableConfig cfg;
    cfg.ref_area   = 1.131f;
    cfg.ref_length = 12.0f;
    cfg.ref_span   = 3.0f;
    cfg.com_x      = 6.0f;
    cfg.use_fvm    = false;

    std::vector<double> mach   = {2.0, 4.0};
    std::vector<double> alpha  = {0.0, 5.0};
    std::vector<double> beta   = {0.0};

    if (!generate_aero_table(
            "data/missile/hgv_model_optimized.stl",
            csv.path, mach, alpha, beta, cfg))
        FAIL("Newtonian: generate_aero_table returned false");

    std::vector<CsvRow> rows;
    std::string err;
    if (!read_csv(csv.path, rows, false, &err))
        FAIL("Newtonian: read_csv failed: %s", err.c_str());
    if (rows.size() != 4)
        FAIL("Newtonian: expected 4 rows, got %zu", rows.size());

    for (size_t i = 0; i < rows.size(); ++i) {
        if (!std::isfinite(rows[i].CX))
            FAIL("Newtonian row %zu: CX=%g not finite", i, rows[i].CX);
    }

    std::cout << "PASS: " << rows.size() << " Newtonian rows\n";
    PASS;
    return 0;
}

// ─── Test: CFD results differ from Newtonian ─────────────────────────────
static int test_cfd_differs_from_newtonian() {
    TEST("TABLE-CFD-4 CFD forces differ from Newtonian at non-zero alpha");
    AeroTableConfig cfg;
    cfg.ref_area   = 1.131f;
    cfg.ref_length = 12.0f;
    cfg.ref_span   = 3.0f;
    cfg.com_x      = 6.0f;
    cfg.mesh_subdivisions = 125;

    std::vector<double> mach   = {4.0};
    std::vector<double> alpha  = {10.0};  // non-zero alpha for larger difference
    std::vector<double> beta   = {0.0};

    // Newtonian baseline
    TempFile nt_csv("test_nt.csv");
    cfg.use_fvm = false;
    if (!generate_aero_table(
            "data/missile/hgv_model_optimized.stl",
            nt_csv.path, mach, alpha, beta, cfg))
        FAIL("Newtonian table failed");

    // CFD GPU
    TempFile cfd_csv("test_cfd.csv");
    cfg.use_fvm = true;
    if (!generate_aero_table(
            "data/missile/hgv_model_optimized.stl",
            cfd_csv.path, mach, alpha, beta, cfg))
        FAIL("CFD table failed");

    std::vector<CsvRow> nrows, crows;
    std::string err;
    if (!read_csv(nt_csv.path,  nrows, false, &err)) FAIL("read Newtonian: %s", err.c_str());
    if (!read_csv(cfd_csv.path, crows, true,  &err)) FAIL("read CFD: %s", err.c_str());

    // Compare L/D ratio - more robust discriminator than single force component
    double newtonian_ld = std::abs(nrows[0].CL) / (std::abs(nrows[0].CD) + 1e-12);
    double cfd_ld       = std::abs(crows[0].CL) / (std::abs(crows[0].CD) + 1e-12);
    double diff = std::abs(cfd_ld - newtonian_ld) / (newtonian_ld + 1e-12);

    if (diff < 0.02)
        FAIL("L/D too similar: Newtonian L/D=%g CFD L/D=%g rel_diff=%g (need >=0.02)",
             newtonian_ld, cfd_ld, diff);

    std::cout << "PASS: Newtonian L/D=" << newtonian_ld << " CFD L/D=" << cfd_ld
              << " rel_diff=" << diff << "\n";
    PASS;
    return 0;
}

// ─── Test: single beta=0 ─────────────────────────────────────────────────
static int test_cfd_gpu_single_beta() {
    TEST("TABLE-CFD-5 CFD table with single beta=0 works");
    TempFile csv("test_cfd_beta0.csv");

    AeroTableConfig cfg;
    cfg.ref_area   = 1.131f;
    cfg.ref_length = 12.0f;
    cfg.ref_span   = 3.0f;
    cfg.com_x      = 6.0f;
    cfg.use_fvm    = true;
    cfg.mesh_subdivisions = 125;

    std::vector<double> mach   = {3.0, 5.0};
    std::vector<double> alpha  = {2.0};
    std::vector<double> beta   = {0.0};

    if (!generate_aero_table(
            "data/missile/hgv_model_optimized.stl",
            csv.path, mach, alpha, beta, cfg))
        FAIL("table gen failed");

    std::vector<CsvRow> rows;
    std::string err;
    if (!read_csv(csv.path, rows, true, &err))
        FAIL("read_csv: %s", err.c_str());
    if (rows.size() != 2)
        FAIL("expected 2 rows, got %zu", rows.size());

    for (size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].fidelity != "cfd-gpu")
            FAIL("row %zu: fidelity='%s' expected 'cfd-gpu'", i, rows[i].fidelity.c_str());
        if (std::abs(rows[i].CY) > 1e-2)
            FAIL("row %zu: beta=0 symmetry CY=%g > 1e-2", i, rows[i].CY);
    }

    std::cout << "PASS: " << rows.size() << " rows, fidelity=cfd-gpu\n";
    PASS;
    return 0;
}

// ─── Test: non-zero beta grid ────────────────────────────────────────────
static int test_cfd_gpu_nonzero_beta() {
    TEST("TABLE-CFD-6 CFD table with multidimensional beta grid");
    TempFile csv("test_cfd_beta_multi.csv");

    AeroTableConfig cfg;
    cfg.ref_area   = 1.131f;
    cfg.ref_length = 12.0f;
    cfg.ref_span   = 3.0f;
    cfg.com_x      = 6.0f;
    cfg.use_fvm    = true;
    cfg.mesh_subdivisions = 125;

    std::vector<double> mach   = {3.0};
    std::vector<double> alpha  = {0.0, 5.0};
    std::vector<double> beta   = {-5.0, 0.0, 5.0};

    if (!generate_aero_table(
            "data/missile/hgv_model_optimized.stl",
            csv.path, mach, alpha, beta, cfg))
        FAIL("table gen failed for multi-beta grid");

    std::vector<CsvRow> rows;
    std::string err;
    if (!read_csv(csv.path, rows, true, &err))
        FAIL("read_csv: %s", err.c_str());
    // 1 mach x 2 alpha x 3 beta = 6 rows
    if (rows.size() != 6)
        FAIL("expected 6 rows (1x2x3), got %zu", rows.size());

    for (size_t i = 0; i < rows.size(); ++i) {
        auto& r = rows[i];
        if (r.fidelity != "cfd-gpu")
            FAIL("row %zu: fidelity='%s' expected 'cfd-gpu'", i, r.fidelity.c_str());
        if (!std::isfinite(r.CX))
            FAIL("row %zu: CX=%g not finite", i, r.CX);
        // beta symmetry: row pairs (beta=-5 vs beta=+5) at same alpha
        // For now, just verify finite forces for all rows
    }

    // Check beta symmetry: rows i (beta=-5) and i+2 (beta=+5) should have near-opposite CY
    // Row order: alpha=0 beta=-5, alpha=0 beta=0, alpha=0 beta=+5,
    //            alpha=5 beta=-5, alpha=5 beta=0, alpha=5 beta=+5
    double cy_neg = rows[0].CY;  // alpha=0, beta=-5
    double cy_pos = rows[2].CY;  // alpha=0, beta=+5
    // CY should be antisymmetric with beta
    double cy_sum = cy_neg + cy_pos;
    double cy_avg = 0.5 * (std::abs(cy_neg) + std::abs(cy_pos));
    if (cy_avg > 1e-3 && std::abs(cy_sum) / cy_avg > 0.5)
        FAIL("beta symmetry violation: alpha=0 CY(beta=-5)=%g CY(beta=+5)=%g sum/avg=%g",
             cy_neg, cy_pos, std::abs(cy_sum)/cy_avg);

    std::cout << "PASS: " << rows.size() << " rows, beta symmetry holds\n";
    PASS;
    return 0;
}

// ─── Test: empty input vectors ───────────────────────────────────────────
static int test_cfd_gpu_empty_input() {
    TEST("TABLE-CFD-7 Empty input vectors rejected");
    AeroTableConfig cfg;
    cfg.ref_area   = 1.131f;
    cfg.ref_length = 12.0f;
    cfg.ref_span   = 3.0f;
    cfg.com_x      = 6.0f;
    cfg.use_fvm    = true;

    // All empty
    {
        TempFile csv("test_cfd_empty.csv");
        bool ok = generate_aero_table("data/missile/hgv_model_optimized.stl",
            csv.path, {}, {}, {}, cfg);
        if (ok) FAIL("empty input should return false");
    }

    // Partial empty
    {
        TempFile csv("test_cfd_empty_part.csv");
        bool ok = generate_aero_table("data/missile/hgv_model_optimized.stl",
            csv.path, {3.0}, {}, {0.0}, cfg);
        if (ok) FAIL("partially empty input (alpha empty) should return false");
    }

    std::cout << "PASS: empty input correctly rejected\n";
    PASS;
    return 0;
}

int main() {
    int failures = 0;
    failures += test_cfd_gpu_table_in_range();
    failures += test_cfd_gpu_out_of_range();
    failures += test_newtonian_baseline();
    failures += test_cfd_differs_from_newtonian();
    failures += test_cfd_gpu_single_beta();
    failures += test_cfd_gpu_nonzero_beta();
    failures += test_cfd_gpu_empty_input();

    std::cout << "\n[" << pass_count << "/" << test_count << " tests passed]\n";
    if (failures) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cout << "All tests PASS\n";
    return 0;
}
