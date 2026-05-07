// =============================================================================
// dgemm_naive.cpp
// Level 0: Naive sequential DGEMM — no optimisations
//
// Purpose: Benchmark baseline to compare against systolic_dgemm_v7.cpp
//
// Deliberately NOT optimised:
//   - FP64 (double): 3-4 DSPs per MAC vs 1 DSP for ap_fixed<27,13>x<18,9>
//   - No PIPELINE pragma: HLS generates sequential logic
//   - No ARRAY_PARTITION: BRAM port contention, II >> 1
//   - No DATAFLOW: load/compute/store are fully serialised
//   - Direct DDR access inside compute loop: ~100 ns per element read
//
// Same AXI interface as systolic_dgemm_v7 so both compile under the same
// Vitis project and the testbench can call either.
//
// Target: Vitis HLS 2022.1+, KV260 (xczu5ev)
// Expected: ~0.5 GFLOPS (ARM Cortex-A53 baseline), possibly worse
// =============================================================================

#include "ap_fixed.h"
#include "hls_stream.h"

extern "C" {

// -----------------------------------------------------------------------------
// dgemm_naive: C = A * B  (M x N = M x K * K x N)
//
// No pragmas — HLS generates the most straightforward RTL possible.
// Every inner-loop iteration waits for the previous to fully complete.
// Every A[i][k] and B[k][j] access goes to DDR (~100 ns each).
// -----------------------------------------------------------------------------
void dgemm_naive(
    const double *A,   // [M][K] — row-major
    const double *B,   // [K][N] — row-major
    double       *C,   // [M][N] — row-major (output)
    int M, int N, int K
) {
    // AXI memory-mapped ports (identical bundle structure to v7)
    #pragma HLS INTERFACE m_axi port=A bundle=gmem0
    #pragma HLS INTERFACE m_axi port=B bundle=gmem1
    #pragma HLS INTERFACE m_axi port=C bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=A
    #pragma HLS INTERFACE s_axilite port=B
    #pragma HLS INTERFACE s_axilite port=C
    #pragma HLS INTERFACE s_axilite port=M
    #pragma HLS INTERFACE s_axilite port=N
    #pragma HLS INTERFACE s_axilite port=K
    #pragma HLS INTERFACE s_axilite port=return

    // Triple loop — sequential, no pipelining, no tiling
    // Every access reads directly from DDR — worst-case latency
    OUTER_I: for (int i = 0; i < M; i++) {
        OUTER_J: for (int j = 0; j < N; j++) {
            double sum = 0.0;
            INNER_K: for (int k = 0; k < K; k++) {
                // Each of these reads stalls waiting for DDR (~100 ns)
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

} // extern "C"

// =============================================================================
// EXPECTED SYNTHESIS REPORT (for comparison with v7):
//
//  Latency:
//    Loop OUTER_I > OUTER_J > INNER_K: M*N*K iterations
//    Each iteration: ~5+ cycles (multiply + add + DDR access latency)
//    Total for 64x64 matrix: 64*64*64 * ~5 = ~1.3M cycles @ 200 MHz = ~6.5 ms
//
//  v7 for same 64x64 matrix:
//    Outer tiles: (64/8)*(64/8) = 64 tile computations
//    Each tile: TILE_K + SA_SIZE - 1 = 23 cycles (8x8 PEs compute in parallel)
//    Total: 64 * 4 K-tiles * 23 cycles = ~5,900 cycles @ 250 MHz = ~24 us
//    Speedup: ~270x
//
//  DSP usage:
//    Naive: 3-4 DSPs per multiply (FP64) = up to 4 DSPs (sequential, 1 at a time)
//    v7:    64 DSPs (one per PE, all active simultaneously)
//
//  II of INNER_K loop:
//    Naive: expect II = 5-20 (FP64 multiply latency + no pipeline)
//    v7 PE_LOOP: II = 1 (fully pipelined, BIND_OP guarantees DSP mapping)
// =============================================================================
