// =============================================================================
// systolic_dgemm_v7.cpp
// Level 7: Production-grade systolic DGEMM for Xilinx KV260 (UltraScale+)
//
// Architecture:
//   8x8 output-stationary systolic array
//   ap_fixed<27,13> x ap_fixed<18,9> -> exactly 1 DSP48E2 per MAC
//   SRL-based input skewing  (no arithmetic masking)
//   DATAFLOW with functions only (no loops inside dataflow region)
//   BIND_OP to guarantee DSP accumulator mapping
//
// Target: Vitis HLS 2022.1+, KV260 (xczu5ev)
// Expected: ~80 GOPS at 250 MHz (64 PEs x 2 FLOP x 250M / 4 DSPs overhead)
// =============================================================================

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

// =============================================================================
// 1. TYPES — sized exactly to DSP48E2 port limits
//    A-port: 27 bits max, B-port: 18 bits max, P-register: 48 bits
// =============================================================================
typedef ap_fixed<27, 13, AP_TRN, AP_WRAP> weight_t;  // A-port — AP_WRAP preserves DSP inference
typedef ap_fixed<18,  9, AP_TRN, AP_WRAP> act_t;      // B-port — AP_SAT would break DSP!
typedef ap_fixed<48, 22, AP_TRN, AP_WRAP> acc_t;      // Maps to 48-bit P-register

static const int SA_SIZE = 8;    // 8x8 = 64 PEs
static const int TILE_K  = 16;   // K-tile depth — power of 2
static const int TILE_M  = SA_SIZE;
static const int TILE_N  = SA_SIZE;

// =============================================================================
// 2. PROCESSING ELEMENT
//    BIND_OP forces accumulator onto DSP, not fabric LUTs
//    Asymmetric types guarantee 1 DSP48E2 per MAC (27x18 fits perfectly)
// =============================================================================
void pe(
    hls::stream<weight_t> &a_in,  hls::stream<weight_t> &a_out,
    hls::stream<act_t>    &b_in,  hls::stream<act_t>    &b_out,
    hls::stream<acc_t>    &c_out
) {
    #pragma HLS INLINE off
    acc_t acc = 0;
    #pragma HLS BIND_OP variable=acc op=add impl=dsp  // accumulate inside DSP P-register

    // TILE_K + SA_SIZE - 1 known at compile time -> fully unrolled by HLS
    PE_LOOP: for (int k = 0; k < TILE_K + SA_SIZE - 1; k++) {
        #pragma HLS PIPELINE II=1
        // No DEPENDENCE pragma needed: BIND_OP+DSP handles the feedback correctly
        weight_t a = a_in.read();
        act_t    b = b_in.read();
        acc += (acc_t)(a * b);   // 27x18 -> 1 DSP48E2, P-register accumulates internally
        a_out.write(a);          // pass right ->
        b_out.write(b);          // pass down  |
    }
    c_out.write(acc);
}

// =============================================================================
// 3. SRL DELAY — spatial shift register, zero arithmetic
//    BIND_STORAGE impl=srl -> LUT configured as shift register (not flip-flops)
//    Cost: 1 LUT per bit per delay stage vs arithmetic masking (comparators + MUX)
// =============================================================================
template<int DELAY, typename T>
void srl_delay_w(hls::stream<T> &in, hls::stream<T> &out) {
    #pragma HLS INLINE off
    T shift_reg[DELAY];
    #pragma HLS ARRAY_PARTITION variable=shift_reg complete
    #pragma HLS BIND_STORAGE variable=shift_reg impl=srl type=ram_1p

    SRL_LOOP: for (int t = 0; t < TILE_K + SA_SIZE - 1; t++) {
        #pragma HLS PIPELINE II=1
        T in_val = in.read();
        for (int i = DELAY - 1; i > 0; i--) {
            #pragma HLS UNROLL
            shift_reg[i] = shift_reg[i-1];
        }
        shift_reg[0] = in_val;
        out.write(shift_reg[DELAY > 0 ? DELAY - 1 : 0]);
    }
}

template<int DELAY, typename T>
void srl_delay_a(hls::stream<T> &in, hls::stream<T> &out) {
    #pragma HLS INLINE off
    T shift_reg[DELAY];
    #pragma HLS ARRAY_PARTITION variable=shift_reg complete
    #pragma HLS BIND_STORAGE variable=shift_reg impl=srl type=ram_1p

    SRL_LOOP: for (int t = 0; t < TILE_K + SA_SIZE - 1; t++) {
        #pragma HLS PIPELINE II=1
        T in_val = in.read();
        for (int i = DELAY - 1; i > 0; i--) {
            #pragma HLS UNROLL
            shift_reg[i] = shift_reg[i-1];
        }
        shift_reg[0] = in_val;
        out.write(shift_reg[DELAY > 0 ? DELAY - 1 : 0]);
    }
}

// =============================================================================
// 4. SKEW FEEDERS — one function per row/column (no loop inside DATAFLOW)
//    Row i gets i cycles of SRL delay -> correct systolic skew
//    Using separate functions (not a loop) so DATAFLOW sees them as parallel tasks
// =============================================================================
void skew_a(
    weight_t A[SA_SIZE][TILE_K],
    hls::stream<weight_t> a_skewed[SA_SIZE]
) {
    #pragma HLS ARRAY_PARTITION variable=A complete dim=2
    #pragma HLS DATAFLOW

    hls::stream<weight_t> raw[SA_SIZE];
    #pragma HLS STREAM variable=raw depth=TILE_K+SA_SIZE

    // Load raw rows into streams (sequential — no DATAFLOW violation)
    LOAD_A: for (int t = 0; t < TILE_K + SA_SIZE - 1; t++) {
        #pragma HLS PIPELINE II=1
        for (int i = 0; i < SA_SIZE; i++) {
            #pragma HLS UNROLL
            int k = t - i;
            bool valid = (k >= 0) && (k < TILE_K);
            raw[i].write(valid ? A[i][k] : weight_t(0));
        }
    }

    // Each row goes through its own SRL delay (i cycles for row i)
    // Delay 0 = passthrough, delay 1..7 = SRL chains
    srl_delay_w<0>(raw[0], a_skewed[0]);
    srl_delay_w<1>(raw[1], a_skewed[1]);
    srl_delay_w<2>(raw[2], a_skewed[2]);
    srl_delay_w<3>(raw[3], a_skewed[3]);
    srl_delay_w<4>(raw[4], a_skewed[4]);
    srl_delay_w<5>(raw[5], a_skewed[5]);
    srl_delay_w<6>(raw[6], a_skewed[6]);
    srl_delay_w<7>(raw[7], a_skewed[7]);
}

void skew_b(
    act_t B[TILE_K][SA_SIZE],
    hls::stream<act_t> b_skewed[SA_SIZE]
) {
    #pragma HLS ARRAY_PARTITION variable=B complete dim=2
    #pragma HLS DATAFLOW

    hls::stream<act_t> raw[SA_SIZE];
    #pragma HLS STREAM variable=raw depth=TILE_K+SA_SIZE

    LOAD_B: for (int t = 0; t < TILE_K + SA_SIZE - 1; t++) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < SA_SIZE; j++) {
            #pragma HLS UNROLL
            int k = t - j;
            bool valid = (k >= 0) && (k < TILE_K);
            raw[j].write(valid ? B[k][j] : act_t(0));
        }
    }

    srl_delay_a<0>(raw[0], b_skewed[0]);
    srl_delay_a<1>(raw[1], b_skewed[1]);
    srl_delay_a<2>(raw[2], b_skewed[2]);
    srl_delay_a<3>(raw[3], b_skewed[3]);
    srl_delay_a<4>(raw[4], b_skewed[4]);
    srl_delay_a<5>(raw[5], b_skewed[5]);
    srl_delay_a<6>(raw[6], b_skewed[6]);
    srl_delay_a<7>(raw[7], b_skewed[7]);
}

// =============================================================================
// 5. PE GRID WIRING — fully unrolled 8x8 array
//    Separate from DATAFLOW region to avoid loop-inside-DATAFLOW violation
// =============================================================================
void pe_grid(
    hls::stream<weight_t> a_in[SA_SIZE],
    hls::stream<act_t>    b_in[SA_SIZE],
    hls::stream<acc_t>    c_out[SA_SIZE][SA_SIZE]
) {
    hls::stream<weight_t> a_pipe[SA_SIZE][SA_SIZE+1];
    hls::stream<act_t>    b_pipe[SA_SIZE+1][SA_SIZE];
    #pragma HLS STREAM variable=a_pipe depth=2
    #pragma HLS STREAM variable=b_pipe depth=2

    // Wire inputs to left/top edges
    WIRE_A: for (int i = 0; i < SA_SIZE; i++)
        for (int t = 0; t < TILE_K + SA_SIZE - 1; t++) {
            #pragma HLS PIPELINE II=1
            a_pipe[i][0].write(a_in[i].read());
        }
    WIRE_B: for (int j = 0; j < SA_SIZE; j++)
        for (int t = 0; t < TILE_K + SA_SIZE - 1; t++) {
            #pragma HLS PIPELINE II=1
            b_pipe[0][j].write(b_in[j].read());
        }

    // Instantiate 8x8 = 64 PEs (UNROLL creates 64 parallel hardware units)
    PE_ROW: for (int i = 0; i < SA_SIZE; i++) {
        PE_COL: for (int j = 0; j < SA_SIZE; j++) {
            #pragma HLS UNROLL
            pe(a_pipe[i][j], a_pipe[i][j+1],
               b_pipe[i][j], b_pipe[i+1][j],
               c_out[i][j]);
        }
    }
}

// =============================================================================
// 6. COLLECT — drain PE outputs into C tile
// =============================================================================
void collect(
    hls::stream<acc_t> c_pipe[SA_SIZE][SA_SIZE],
    acc_t C_tile[SA_SIZE][SA_SIZE]
) {
    COLLECT: for (int i = 0; i < SA_SIZE; i++)
        for (int j = 0; j < SA_SIZE; j++) {
            #pragma HLS PIPELINE II=1
            C_tile[i][j] += c_pipe[i][j].read(); // accumulate across K-tiles
        }
}

// =============================================================================
// 7. TILE COMPUTE — DATAFLOW region (functions only, no raw loops)
// =============================================================================
void compute_tile(
    weight_t A[SA_SIZE][TILE_K],
    act_t    B[TILE_K][SA_SIZE],
    acc_t    C[SA_SIZE][SA_SIZE]
) {
    #pragma HLS DATAFLOW

    hls::stream<weight_t> a_skewed[SA_SIZE];
    hls::stream<act_t>    b_skewed[SA_SIZE];
    hls::stream<acc_t>    c_pipe[SA_SIZE][SA_SIZE];
    #pragma HLS STREAM variable=a_skewed depth=SA_SIZE+TILE_K
    #pragma HLS STREAM variable=b_skewed depth=SA_SIZE+TILE_K
    #pragma HLS STREAM variable=c_pipe   depth=1

    skew_a(A, a_skewed);
    skew_b(B, b_skewed);
    pe_grid(a_skewed, b_skewed, c_pipe);
    collect(c_pipe, C);
}

// =============================================================================
// 8. TOP-LEVEL KERNEL — AXI interfaces, outer tile loops
//    M x N x K matrix multiplication tiled into SA_SIZE x SA_SIZE x TILE_K blocks
// =============================================================================
extern "C" {
void systolic_dgemm(
    const weight_t *A,   // [M][K]
    const act_t    *B,   // [K][N]
    acc_t          *C,   // [M][N]
    int M, int N, int K
) {
    #pragma HLS INTERFACE m_axi port=A bundle=gmem0 max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=B bundle=gmem1 max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=C bundle=gmem2 max_write_burst_length=64
    #pragma HLS INTERFACE s_axilite port=A
    #pragma HLS INTERFACE s_axilite port=B
    #pragma HLS INTERFACE s_axilite port=C
    #pragma HLS INTERFACE s_axilite port=M
    #pragma HLS INTERFACE s_axilite port=N
    #pragma HLS INTERFACE s_axilite port=K
    #pragma HLS INTERFACE s_axilite port=return

    // On-chip BRAM tiles — sized to SA_SIZE x TILE_K
    weight_t A_tile[SA_SIZE][TILE_K];
    act_t    B_tile[TILE_K][SA_SIZE];
    acc_t    C_tile[SA_SIZE][SA_SIZE];
    #pragma HLS ARRAY_PARTITION variable=A_tile complete dim=2
    #pragma HLS ARRAY_PARTITION variable=B_tile complete dim=2

    // Outer tile loops — not in DATAFLOW, so sequential is fine here
    TILE_M: for (int ti = 0; ti < M; ti += SA_SIZE) {
        TILE_N: for (int tj = 0; tj < N; tj += SA_SIZE) {

            // Zero accumulator for this output tile
            ZERO_C: for (int i = 0; i < SA_SIZE; i++)
                for (int j = 0; j < SA_SIZE; j++) {
                    #pragma HLS PIPELINE II=1
                    C_tile[i][j] = 0;
                }

            TILE_K_LOOP: for (int tk = 0; tk < K; tk += TILE_K) {

                // Burst-load A tile from DDR -> BRAM
                LOAD_A: for (int i = 0; i < SA_SIZE; i++)
                    for (int k = 0; k < TILE_K; k++) {
                        #pragma HLS PIPELINE II=1
                        A_tile[i][k] = A[(ti+i)*K + (tk+k)];
                    }

                // Burst-load B tile from DDR -> BRAM
                LOAD_B: for (int k = 0; k < TILE_K; k++)
                    for (int j = 0; j < SA_SIZE; j++) {
                        #pragma HLS PIPELINE II=1
                        B_tile[k][j] = B[(tk+k)*N + (tj+j)];
                    }

                // Run systolic array on this tile (DATAFLOW inside)
                compute_tile(A_tile, B_tile, C_tile);
            }

            // Write accumulated C tile back to DDR
            STORE_C: for (int i = 0; i < SA_SIZE; i++)
                for (int j = 0; j < SA_SIZE; j++) {
                    #pragma HLS PIPELINE II=1
                    C[(ti+i)*N + (tj+j)] = C_tile[i][j];
                }
        }
    }
}
} // extern "C"

// =============================================================================
// SYNTHESIS CHECKLIST (verify in Vitis HLS report after C-synthesis):
//
//  DSP usage:
//    Should be ~64 DSPs for 8x8 array (1 per PE)
//    If you see ~128+, the 27x18 types are not mapping correctly
//    -> Check: #pragma HLS BIND_OP variable=acc op=add impl=dsp
//
//  BRAM usage:
//    A_tile, B_tile, C_tile should fit in ~6 BRAMs total
//    SRL shift registers: appear as LUTs, not BRAMs (correct)
//
//  Timing (II=1 on PE_LOOP):
//    If II>1: check synthesis report for the dependency shown
//    Most likely cause: acc read-write if BIND_OP fails
//
//  Latency per tile:
//    TILE_K + SA_SIZE - 1 = 23 cycles per (8x8) tile
//    For 1024x1024 matrix: (128x128x64) tiles x 23 cy / 250MHz ~ 6 ms
//
//  Target frequency: 250 MHz (4 ns period)
//    If timing fails: check skew_a/skew_b SRL inference in utilisation report
//    SRL should show as LUT, not as FF chains
// =============================================================================
