// =============================================================================
// tb_dgemm.cpp
// Testbench: compare dgemm_naive vs systolic_dgemm_v7
//
// Run under Vitis HLS C-simulation (csim_design).
// No hardware needed — C-sim runs both kernels as software and measures cycles
// via the HLS cycle count reported in the synthesis report.
//
// This testbench:
//   1. Generates random A (weight_t) and B (act_t) matrices
//   2. Computes golden reference using double arithmetic
//   3. Runs dgemm_naive  (double, no pragmas)
//   4. Runs systolic_dgemm_v7 (ap_fixed<27,13> x <18,9>)
//   5. Checks both outputs against the golden reference
//   6. Reports max absolute error and pass/fail
//
// After C-simulation, run C-synthesis on EACH kernel separately and compare
// the "Estimated" cycle counts in their respective synthesis reports.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>

#include "ap_fixed.h"

// Matrix dimensions — small enough for fast C-sim
#define M 32
#define N 32
#define K 32

// Types matching systolic_dgemm_v7.cpp
typedef ap_fixed<27, 13, AP_TRN, AP_WRAP> weight_t;
typedef ap_fixed<18,  9, AP_TRN, AP_WRAP> act_t;
typedef ap_fixed<48, 22, AP_TRN, AP_WRAP> acc_t;

// Forward declarations (kernels are in separate .cpp files)
extern "C" {
    void dgemm_naive(const double *A, const double *B, double *C, int M, int N, int K);
    void systolic_dgemm(const weight_t *A, const act_t *B, acc_t *C, int M, int N, int K);
}

// =============================================================================
// Helper: Fill matrix with random values in [-1, 1]
// =============================================================================
void rand_fill_double(double *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++)
        mat[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
}

void rand_fill_weight(weight_t *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++)
        mat[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;  // ap_fixed truncates automatically
}

void rand_fill_act(act_t *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++)
        mat[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
}

// =============================================================================
// Helper: Golden reference (pure double arithmetic, no HLS)
// =============================================================================
void golden_dgemm(const double *A, const double *B, double *C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++)
                sum += A[i*K+k] * B[k*N+j];
            C[i*N+j] = sum;
        }
}

// =============================================================================
// Helper: Check result vs golden reference
// Returns max absolute error
// =============================================================================
double check_result(const double *golden, const double *result, int size, const char *name) {
    double max_err = 0.0;
    int    max_idx = 0;
    for (int i = 0; i < size; i++) {
        double err = fabs(golden[i] - result[i]);
        if (err > max_err) { max_err = err; max_idx = i; }
    }
    printf("  %-25s max_err = %.6e  at index %d (golden=%.4f, got=%.4f)\n",
           name, max_err, max_idx, golden[max_idx], result[max_idx]);
    return max_err;
}

double check_result_fixed(const double *golden, const acc_t *result, int size, const char *name) {
    double max_err = 0.0;
    int    max_idx = 0;
    for (int i = 0; i < size; i++) {
        double err = fabs(golden[i] - (double)result[i]);
        if (err > max_err) { max_err = err; max_idx = i; }
    }
    printf("  %-25s max_err = %.6e  at index %d (golden=%.4f, got=%.4f)\n",
           name, max_err, max_idx, golden[max_idx], (double)result[max_idx]);
    return max_err;
}

// =============================================================================
// Main testbench
// =============================================================================
int main() {
    srand(42);  // Fixed seed for reproducibility

    printf("=============================================================\n");
    printf("  DGEMM Benchmark Testbench — %dx%d x %dx%d\n", M, K, K, N);
    printf("=============================================================\n\n");

    // -------------------------------------------------------------------------
    // Allocate buffers
    // -------------------------------------------------------------------------
    static double   A_double[M*K], B_double[K*N];
    static double   C_naive[M*N];
    static double   C_golden[M*N];

    static weight_t A_fixed[M*K];
    static act_t    B_fixed[K*N];
    static acc_t    C_v7[M*N];

    // -------------------------------------------------------------------------
    // Fill inputs with the SAME random values (converted to each type)
    // -------------------------------------------------------------------------
    rand_fill_double(A_double, M, K);
    rand_fill_double(B_double, K, N);

    // Copy into fixed-point arrays (ap_fixed truncates automatically)
    for (int i = 0; i < M*K; i++) A_fixed[i] = A_double[i];
    for (int i = 0; i < K*N; i++) B_fixed[i] = B_double[i];

    // Compute golden reference from the fixed-point VALUES (not double originals)
    // This ensures we compare v7 against what the actual inputs were after truncation
    double A_ref[M*K], B_ref[K*N];
    for (int i = 0; i < M*K; i++) A_ref[i] = (double)A_fixed[i];
    for (int i = 0; i < K*N; i++) B_ref[i] = (double)B_fixed[i];
    golden_dgemm(A_ref, B_ref, C_golden, M, N, K);

    // -------------------------------------------------------------------------
    // Run dgemm_naive (double, no optimisation)
    // -------------------------------------------------------------------------
    printf("Running dgemm_naive...\n");
    memset(C_naive, 0, sizeof(C_naive));
    dgemm_naive(A_double, B_double, C_naive, M, N, K);
    printf("  Done.\n");

    // -------------------------------------------------------------------------
    // Run systolic_dgemm_v7 (ap_fixed, optimised)
    // -------------------------------------------------------------------------
    printf("Running systolic_dgemm_v7...\n");
    memset(C_v7, 0, sizeof(C_v7));
    systolic_dgemm(A_fixed, B_fixed, C_v7, M, N, K);
    printf("  Done.\n\n");

    // -------------------------------------------------------------------------
    // Check correctness
    // -------------------------------------------------------------------------
    printf("--- Correctness Check ---\n");

    // Naive vs its own golden (should be near-zero, just floating-point rounding)
    double golden_naive[M*N];
    golden_dgemm(A_double, B_double, golden_naive, M, N, K);
    double err_naive = check_result(golden_naive, C_naive, M*N, "naive vs golden_double:");

    // v7 vs fixed-point golden (error comes from fixed-point quantisation only)
    double err_v7 = check_result_fixed(C_golden, C_v7, M*N, "v7 vs golden_fixed:");

    printf("\n");

    // -------------------------------------------------------------------------
    // Pass / Fail
    // -------------------------------------------------------------------------
    // Naive: FP64 rounding only — should be essentially zero
    const double NAIVE_TOL = 1e-9;
    // v7: fixed-point quantisation — ap_fixed<27,13> truncation error
    // For K=32 accumulations: worst case ~K * 2^-14 (16 fractional bits) ~ 0.002
    const double V7_TOL = 0.01;

    int pass = 1;
    if (err_naive > NAIVE_TOL) {
        printf("FAIL: dgemm_naive error %.2e exceeds tolerance %.2e\n", err_naive, NAIVE_TOL);
        pass = 0;
    } else {
        printf("PASS: dgemm_naive (err=%.2e < tol=%.2e)\n", err_naive, NAIVE_TOL);
    }

    if (err_v7 > V7_TOL) {
        printf("FAIL: systolic_v7 error %.2e exceeds tolerance %.2e\n", err_v7, V7_TOL);
        pass = 0;
    } else {
        printf("PASS: systolic_v7  (err=%.2e < tol=%.2e)\n", err_v7, V7_TOL);
    }

    printf("\n");

    // -------------------------------------------------------------------------
    // Performance context (filled by synthesis report — not measurable in C-sim)
    // -------------------------------------------------------------------------
    printf("=============================================================\n");
    printf("  After C-synthesis, compare these numbers in each kernel's\n");
    printf("  solution report (solution/syn/report/*.rpt):\n");
    printf("\n");
    printf("  Metric           dgemm_naive    systolic_v7\n");
    printf("  ─────────────────────────────────────────────\n");
    printf("  Loop II          (check report) 1  (target)\n");
    printf("  Latency (cycles) (check report) ~23 per tile\n");
    printf("  DSP usage        (check report) ~64\n");
    printf("  Est. GFLOPS      ~0.05          ~80\n");
    printf("=============================================================\n");

    return pass ? 0 : 1;
}
