#pragma once

#ifdef _WIN32
    #ifdef AERO_SOLVER_EXPORTS
        #define AERO_SOLVER_API __declspec(dllexport)
    #else
        #define AERO_SOLVER_API __declspec(dllimport)
    #endif
#else
    #define AERO_SOLVER_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Generate a complete aerodynamics table in a single GPU pass.
//
// Parameters:
//   stl_path     - Path to the STL geometry file.
//   csv_path     - Output CSV file path.
//   mach_grid    - Array of Mach numbers.
//   n_mach       - Number of Mach entries.
//   alpha_grid   - Array of angles of attack (degrees).
//   n_alpha      - Number of alpha entries.
//   beta_grid    - Array of sideslip angles (degrees).
//   n_beta       - Number of beta entries.
//   ref_area     - Reference area (m^2).
//   ref_length   - Reference length (m).
//   ref_span     - Reference span (m).
//   com_x        - Center of mass X (m).
//
// Returns: 0 on success, -1 on error.
AERO_SOLVER_API int generate_aero_table_c(
    const char* stl_path,
    const char* csv_path,
    const double* mach_grid, int n_mach,
    const double* alpha_grid, int n_alpha,
    const double* beta_grid, int n_beta,
    float ref_area, float ref_length, float ref_span,
    float com_x);

// Returns a human-readable string for the last error (valid until next call).
AERO_SOLVER_API const char* aero_solver_last_error(void);

#ifdef __cplusplus
}
#endif
