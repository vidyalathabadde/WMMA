/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// CUDA sample demonstrating a GEMM computation using the Warp Matrix Multiply
// and Accumulate API introduced in CUDA 9.

// In this program, the compute_gemm kernel computes the result of a matrix
// multiplication and addition: D = alpha * A * B + beta * C. The dimensions of
// both C and D matrices are M_GLOBAL x N_GLOBAL. The A matrix is M_GLOBAL x
// K_GLOBAL (row-major), the B matrix is K_GLOBAL x N_GLOBAL (column-major). In
// that kernel, each CTA computes one 128 x 128 tile of the resulting matrix per
// iteration. When the tile is computed, the CTA stores it to the global memory
// and begins a new iteration, selecting a new 128 x 128 tile to compute.
// Each CTA consists of eight warps. For the 128 x 128 tile, each warp computes
// eight 16 x 16 subtiles, organized in a 2 x 4 two-dimensional array. Warps
// compute the 16 x 16 subtiles using nvcuda::wmma::mma_sync operations by
// moving through the K_GLOBAL dimension of the A and B matrices and
// accumulating the intermediate result in the local thread state.

// There are a number of simple optimizations used in the algorithm:
// - The CTA copies the 128 x 128 tile of the C matrix from the global memory to
//   shared memory. After that is done, each warp loads the C matrix fragments
//   from shared memory, thus avoiding a random global memory access.
// - On each internal iteration, the CTA copies a portion of the A and B
//   matrices from global memory to shared memory. After that, all warps in the
//   CTA reuse the A and B data from shared memory, thus reducing the number of
//   data copies from global memory.
// - The portions of the A and B matrices are stored in shared memory with an
//   additional padding (skew) to reduce the number of shared memory access bank
//   conflicts.
//   (See a detailed explanation near the SKEW_HALF macro definition.)
// - When the CTA finishes computing the tiles of the resulting matrix, each
//   warp stores its subtiles to shared memory. The CTA then copies the shared
//   memory contents to global memory, again avoiding redundant random global
//   memory  accesses.
// - Note that the CTA tile size is chosen to maximize the GPU register
//   utilization, but carefully enough to avoid local memory use.

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/device.hpp>
#include <assert.h>
#include <stdio.h>
#include <iostream>
//#include <cuda.h>
// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>
#include <chrono>
using namespace sycl;
using namespace sycl::ext::oneapi;
using namespace sycl::ext::oneapi::experimental::matrix;
// Externally configurable parameters.

#ifndef CPU_DEBUG
// Set this to 1 to verify the correctness of the GPU-computed matrix.
#define CPU_DEBUG 1
#endif

#ifndef SHARED_MEMORY_LIMIT_64K
// Set this to 0 to use more than 64 Kb of shared memory to cache data, to
// improve the performance of the computations on GPU.
// Note that you need a GPU that can have more than 64 Kb of shared memory
// per multiprocessor.
#define SHARED_MEMORY_LIMIT_64K 1
#endif

// GPU configuration.

#define WARP_SIZE 16 //DPAS of PVC SG size is 16

// MMA matrix tile dimensions.

#define M 8// 16  
#define N 16 
#define K 16

#define WMMA_M 8 //in DPAS of PVC, combination supporte for half data type is 8x16x16
#define WMMA_N 16
#define WMMA_K 16

// GEMM configuration.

#define M_TILES 16
#define N_TILES 16
#define K_TILES 16

#define M_GLOBAL (M * M_TILES)
#define N_GLOBAL (N * N_TILES)
#define K_GLOBAL (K * K_TILES)

#define C_LAYOUT wmma::mem_row_major

// Implementation constants.

#define WARPS_PER_BLOCK 16// which is 256/warp_size //8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#if SHARED_MEMORY_LIMIT_64K
// With only 64 Kb shared memory available, we can fit two 8-tile chunks of
// the A and B matrix data, that are 16 * 16 * 8 * 8 * 2 = 32 Kb each
// (i.e. two 8x8 arrays of tiles of 16x16 half-typed elements per CTA).
// But we cannot account the 8 Kb total skew overhead, without which the
// performance would be severely impacted. So we choose to reduce the chunk size
// in half, i.e. the amount of A and B matrix data we cache in shared memory.
// Accordingly, this doubles the number of outer iterations across the global K
// dimension, which only slightly impacts the performance.
#define CHUNK_K 4
#else
#define CHUNK_K 8
#endif

#define CHUNK_LINE_BYTES (CHUNK_K * K * sizeof(sycl::half))
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(sycl::int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define GLOBAL_MEM_STRIDE N_GLOBAL

#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)

// The macro below is used to shift rows of the A matrix and columns of the B matrix
// in shared memory to minimize possible bank conflicts.
// Before performing the nvcuda::wmma::mma_sync operation, the warp must load the matrix
// data using the nvcuda::wmma::load_matrix_sync operation. Although the memory access pattern
// is not specified for that function, each lane in the warp can read one or multiple matrix
// elements from different matrix rows or columns.
// For shared memory, such access can result in bank conflicts if different rows / columns
// of the matrix map to the same bank. By shifting each row and column by a few bytes, we
// make sure that they map to different banks, thus reducing the number of possible bank
// conflicts.
// The number of 16 two-byte "half" elements is chosen as the minimum possible shift because
// we must keep each row and column 256-bit aligned, as required by nvcuda::wmma::load_matrix_sync.
#define SKEW_HALF 16

/*
DPCT1001:43: The statement could not be removed.
*/
/*
DPCT1000:44: Error handling if-stmt was detected but could not be rewritten.
*/
/*
DPCT1010:45: SYCL uses exceptions to report errors and does not use the error
codes. The call was replaced with 0. You need to rewrite this code.
*/
/*
DPCT1009:46: SYCL uses exceptions to report errors and does not use the error
codes. The original code was commented out and a warning string was inserted.
You need to rewrite this code.
*/
#define checkKernelErrors(expr)                                                 \
  do {                                                                          \
    /*expr;                                                                 */      \
                                                                                \
    int  __err = 0;                                                       \
    if (__err != 0) {                                                           \
      printf(                                                                   \
          "Line %d: '%s' failed: %s\n", __LINE__, #expr,                        \
          "cudaGetErrorString is not supported" /*cudaGetErrorString(__err)*/); \
      abort();                                                                  \
    }                                                                           \
  } while (0)

//using namespace nvcuda;

void init_host_matrices(sycl::half *a, sycl::half *b, float *c) {
  for (int i = 0; i < M_GLOBAL; i++) {
    for (int j = 0; j < K_GLOBAL; j++) {
	  // a[i * K_GLOBAL + j] = (sycl::half)(rand() % 3);
        if (i == j)
        a[i * K_GLOBAL + j] = 1;
        else
            a[i * K_GLOBAL + j] = 0;
    }
  }

  for (int i = 0; i < N_GLOBAL; i++) {
    for (int j = 0; j < K_GLOBAL; j++) {
      b[i * K_GLOBAL + j] =  2;/*(sycl::half)(rand() % 3);*/
//      printf("b values: %f\n", b[i * K_GLOBAL + j]);
    }
  }
//  printf("b[12] value: %f\n", b[12]);

  for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
    c[t] = 3/*static_cast<float>(rand() % 3)*/;
  }
}

/*void compute_gemm(const sycl::half *A, const sycl::half *B, const float *C,
                  float *D, float alpha, float beta,
                  const sycl::nd_item<3> &item_ct1, uint8_t *dpct_local, sub_group sg) {
  auto shmem = (sycl::half(*)[CHUNK_K * K + SKEW_HALF])dpct_local;

  // Warp and lane identification
  const unsigned int warpId = item_ct1.get_local_id(2) / WARP_SIZE;
  const unsigned int laneId = item_ct1.get_local_id(2) % WARP_SIZE;

  // Offset in shared memory from which the B matrix is stored.
  const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

  // This pointer is used to access the C and D matrix tiles this warp computes.
  float *shmem_warp_tile_ptr = (float *)&shmem[0][0] +
                               (warpId / 2) * SHMEM_STRIDE * K * 2 +
                               (warpId % 2) * SHMEM_OFFSET;

  // This pointer is used to stream the C and D matrices block-wide tile to and
  // from shared memory.
  float *shmem_warp_stream_ptr =
      (float *)&shmem[0][0] + warpId * SHMEM_STRIDE * K;

  // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
  // each tile computation. Technically this is not generally correct (may
  // result in a loss of precision). Zero still needs to be specially handled
  // though.
  beta /= alpha;

  // Each CTA slides along the 128 x 128 tiles from the top left corner of the
  // matrix to the right and down, and selects the next tile to compute. Once
  // there's no such tile, all warps in this CTA exit.
  for (unsigned int block_pos = item_ct1.get_group(2);;
       block_pos += item_ct1.get_group_range(2)) {
    const unsigned int block_tile_i =
        ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
    const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

    // Stop when there are no more D matrix tiles to compute in this CTA.
    if (block_tile_i >= M_TILES) {
      break;
    }

    // This warp's pointer to the C matrix data to copy memory from to shared
    // memory.
    const size_t gmem_idx =
        (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;
    const float *src_gmem_warp_stream_ptr = &C[gmem_idx];

    // Stream multiple C tiles to shared memory.
#pragma unroll
    for (int i = 0; i < K; i++) {
      typedef sycl::int4 copy_t;

      *((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
          *((copy_t *)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) +
            laneId);
    }

    
    DPCT1065:0: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    
    item_ct1.barrier();

    // These fragments will accumulate the result of A and B matrix fragment
    // multiplications along the K_GLOBAL dimension.
    wmma::fragment<wmma::accumulator, M, N, K, float> c[WARP_COL_TILES]
                                                       [WARP_ROW_TILES];
     joint_matrix<sub_group, float, use::accumulator, M, N>c[WARP_COL_TILES][WARP_ROW_TILES] = {

             joint_matrix<sub_group, float, use::accumulator, M, N>(),

             joint_matrix<sub_group, float, use::accumulator, M, N>(),

             joint_matrix<sub_group, float, use::accumulator, M, N>(),

             joint_matrix<sub_group, float, use::accumulator, M, N>(),

              joint_matrix<sub_group, float, use::accumulator, M, N>(),

              joint_matrix<sub_group, float, use::accumulator, M, N>(),

              joint_matrix<sub_group, float, use::accumulator, M, N>(),

              joint_matrix<sub_group, float, use::accumulator, M, N>()}; 
    // Load the C matrix tiles into fragments from shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
        const float *tile_ptr =
            shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

        //wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
	joint_matrix_load(sg, c[i][j], multi_ptr<float, sycl::access::address_space::global_space>(tile_ptr), SHMEM_STRIDE, layout::row_major);
      }
    }

    
    DPCT1065:1: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    
    item_ct1.barrier();

    // Scale the C matrix.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
        for (int t = 0; t < get_wi_data(sg, c[i][j]).length(); t++) {
          get_wi_data(sg, c[i][j])[t] *= beta;
        }
      }
    }

    // Select what warp copies what matrix to shared memory.
    // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
    const sycl::half *warp_ptr = (warpId < 4)
                                     ? (&A[block_tile_i * M * K_GLOBAL] +
                                        M * K_GLOBAL * (warpId % 4) * 2)
                                     : (&B[block_tile_j * N * K_GLOBAL] +
                                        N * K_GLOBAL * (warpId % 4) * 2);

    // Go through the global K dimension by a fixed step at a time.
#pragma unroll
    for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
      // Copy slices of the A and B matrices to shared memory.
      // The first half of the warps in the CTA copy the A matrix, the rest copy
      // the B matrix.
      size_t shmem_idx =
          warpId < (WARPS_PER_BLOCK / 2)
              ? (M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
              : (N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

      // First half of the warp copies the first row / column of the matrix,
      // the second half of the warp copies the next.
      sycl::int4 *lane_ptr =
          (sycl::int4 *)(warp_ptr + tile_k * K +
                         (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) +
          (laneId % CHUNK_COPY_LINE_LANES);

      // Shift the second half of the warp to the next row / column in the
      // shared memory.
      shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

#pragma unroll
      for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2;
           i++) {
        // Copy 16 bytes at once in each lane.
        *((sycl::int4 *)&shmem[shmem_idx][0] +
          (laneId % CHUNK_COPY_LINE_LANES)) = *lane_ptr;

        // Advance the global memory pointer and the shared memory index.
        lane_ptr = (sycl::int4 *)((sycl::half *)lane_ptr +
                                  K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
        shmem_idx += CHUNK_COPY_LINES_PER_WARP;
      }

      
     DPCT1065:4: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      
      item_ct1.barrier();

      // Compute a grid of C matrix tiles in each warp.
#pragma unroll
      for (int k_step = 0; k_step < CHUNK_K; k_step++) {
        wmma::fragment<wmma::matrix_a, M, N, K, sycl::half, wmma::row_major>
            a[WARP_COL_TILES];
        wmma::fragment<wmma::matrix_b, M, N, K, sycl::half, wmma::col_major>
            b[WARP_ROW_TILES];
	joint_matrix<sub_group, half, use::a, M, K>a[WARP_COL_TILES]= {

             joint_matrix<sub_group, half, use::a, M, K>(),

             joint_matrix<sub_group, half, use::a, M, K>()};
	
	joint_matrix<sub_group, half, use::b, K, N>b[WARP_ROW_TILES]= {

             joint_matrix<sub_group, half, use::b, K, N>(),

	     joint_matrix<sub_group, half, use::b, K, N>(),

	     joint_matrix<sub_group, half, use::b, K, N>(),

             joint_matrix<sub_group, half, use::b, K, N>()};


#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
          size_t shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
          const sycl::half *tile_ptr = &shmem[shmem_idx_a][k_step * K];

          //wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_HALF);
	  joint_matrix_load(sg, a[i], multi_ptr<half, sycl::access::address_space::global_space>(tile_ptr) , K * CHUNK_K + SKEW_HALF);

#pragma unroll
          for (int j = 0; j < WARP_ROW_TILES; j++) {
            if (i == 0) {
              // Load the B matrix fragment once, because it is going to be
              // reused against the other A matrix fragments.
              size_t shmem_idx_b = shmem_idx_b_off +
                                   (WARP_ROW_TILES * N) * (warpId % 2) +
                                   (j * N);
              const sycl::half *tile_ptr = &shmem[shmem_idx_b][k_step * K];

             // wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_HALF);
	     joint_matrix_load(sg, b[j], multi_ptr<half, sycl::access::address_space::global_space>(tile_ptr), K * CHUNK_K + SKEW_HALF);
            }

            //wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
	   joint_matrix_mad(sg, a[i], b[j], c[i][j]);
          }
        }
      }

      
      DPCT1065:5: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      
      item_ct1.barrier();
    }

      // Store the D fragments to shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
        // Uniform, point-wise transformations of ALL fragment elements by ALL
        // threads in the warp are well-defined even though element indices
        // within fragment storage are not defined.
        for (int t = 0; t < get_wi_data(sg, c[i][j]).length(); t++) get_wi_data(sg, c[i][j])[t] *= alpha;

        float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

        //wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
	 joint_matrix_store(sg, c[i][j], multi_ptr< float, sycl::access::address_space::global_space>(tile_ptr),  SHMEM_STRIDE, layout::row_major);

      }
    }

    
    DPCT1065:2: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    
    item_ct1.barrier();

    // Now that shared memory contains all the D tiles, stream them to global
    // memory.
    float *dst_gmem_warp_stream_ptr = &D[gmem_idx];

#pragma unroll
    for (int i = 0; i < K; i++) {
      *((sycl::int4 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) +
        laneId) =
          *((sycl::int4 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
    }

    
    DPCT1065:3: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    
    item_ct1.barrier();
  }
}*/


template <typename T>
void matrix_vnni(unsigned int rows, unsigned int cols, T *src, T *dest, unsigned int vnniFactor = 2)
{
       // printf("inside vnni\n");
for (unsigned int i = 0; i < rows / vnniFactor; i++){
for (unsigned int j = 0; j < cols * vnniFactor; j++) {
for (unsigned int k = 0; k < vnniFactor; k++) {
//printf("inside k loop\n");
dest[i * cols * vnniFactor + j * vnniFactor + k] = src[(i * vnniFactor + k) * cols + j];
//std::cout << "dest values: " << dest[i * cols * vnniFactor + j * vnniFactor + k] << std::endl;
}
}
}
printf("end of vnni function\n");
}

// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16.
//  3) Neither A nor B are transposed.
// Note: This is a less performant version of the compute_gemm kernel. It is
// designed for
//       demonstration purposes only to show the CUDA WMMA API use without
//       relying on availability of the shared memory.
void simple_wmma_gemm(sycl::half *a, sycl::half *b, float *c, float *d,
                      int m_ld, int n_ld, int k_ld, float alpha, float beta,
                      const sycl::nd_item<3> &item_ct1, sub_group sg) {
  // Leading dimensions. Packed with no transpositions.
  int lda = k_ld;
  int ldb = k_ld;
  int ldc = n_ld;

  // Tile using a 2D grid
  int warpM = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
               item_ct1.get_local_id(2)) /
              item_ct1.get_sub_group().get_local_range().get(0);
  int warpN = (item_ct1.get_group(1) * item_ct1.get_local_range(1) +
               item_ct1.get_local_id(1));

  // Declare the fragments
  /*wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, sycl::half,
                 wmma::row_major>
      a_frag;*/
  joint_matrix<sub_group, half, use::a, WMMA_M, WMMA_K, layout::row_major> a_frag;// [2] = {
    //joint_matrix<sub_group, half, use::a, WMMA_M, WMMA_K, layout::row_major>(),
   // joint_matrix<sub_group, half, use::a, WMMA_M, WMMA_K, layout::row_major>()};
  
/*  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, sycl::half,
                 wmma::col_major>
      b_frag;*/
  //in PVC implementation, col major is not supported yet. B has to be in packed layout.
  // before entering the kernel, you need to prepack B, 
  // you will have B and Bvnni
  joint_matrix<sub_group, half, use::b, WMMA_K, WMMA_N, ext::intel::experimental::matrix::layout::packed> b_frag;
  //wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
  joint_matrix<sub_group, float, use::accumulator, WMMA_M, WMMA_N>acc_frag;

  //wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  joint_matrix<sub_group, float, use::accumulator, WMMA_M, WMMA_N>c_frag;
//  wmma::fill_fragment(acc_frag, 0.0f);
  joint_matrix_fill(sg, acc_frag, 0.0f);

  // Loop over k
  for (int i = 0; i < k_ld; i += WMMA_K) {
    int aCol = i;
    int aRow = warpM * WMMA_M;
    int bCol = warpN * N;
    int bRow = i;

    // Bounds checking
    if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
      // Load the inputs
      //wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
        joint_matrix_load(sg, a_frag,  multi_ptr< half, sycl::access::address_space::global_space>(a) + aCol + aRow * lda, lda);
//     wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);
       joint_matrix_load(sg, b_frag, multi_ptr< half, sycl::access::address_space::global_space>(b) + bRow + bCol * ldb, ldb);
      // Perform the matrix multiplication
    //  wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
   joint_matrix_mad(sg, a_frag, b_frag, acc_frag);
    }
  }

  // Load in the current value of c, scale it by beta, and add this our result
  // scaled by alpha
  int cCol = warpN * WMMA_N;
  int cRow = warpM * WMMA_M;
auto wi_data_acc = get_wi_data(sg, acc_frag);
auto wi_data_c = get_wi_data(sg, c_frag);

  if (cRow < m_ld && cCol < n_ld) {
//    wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc,
  //                         wmma::mem_row_major);
     joint_matrix_load(sg, c_frag, multi_ptr< float, sycl::access::address_space::global_space>(c) + cCol + cRow * ldc, ldc, layout::row_major);                         

    for (int i = 0; i < wi_data_c.length(); i++) {
      wi_data_c[i] = alpha * wi_data_acc[i]+ beta * wi_data_c[i];
    }

    // Store the output
/*    wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc,
      ]                     wmma::mem_row_major);*/
    joint_matrix_store(sg, c_frag, multi_ptr< float, sycl::access::address_space::global_space>(d) + cCol + cRow * ldc, ldc, layout::row_major);
  
}
}
/*template <typename T>
void matrix_vnni(unsigned int rows, unsigned int cols, T *src, T *dest, unsigned int vnniFactor = 2) 
{
	printf("inside vnni\n");
for (unsigned int i = 0; i < rows / vnniFactor; i++){
    for (unsigned int j = 0; j < cols; j++) {
      for (unsigned int k = 0; k < vnniFactor; k++) {
	printf("inside k loop\n");
        dest[i * cols * vnniFactor + j * vnniFactor + k] =
            src[(i * vnniFactor + k) * cols + j];
       printf("dest[0] value: %f\n", dest[0]);
       }
    }
  }

}*/
void matMultiplyOnHost(sycl::half *A, sycl::half *vnniB, float *C, float alpha,
                       float beta, int numARows, int numAColumns, int numBRows,
                       int numBColumns, int numCRows, int numCColumns) {
  for (int i = 0; i < numCRows; i++) {
    for (int j = 0; j < numCColumns; j++) {
      float temp = 0.0;

      for (int k = 0; k < numAColumns; k++) {
        temp += (float)A[i * numAColumns + k] * (float)vnniB[j * numBRows + k];
      }

      C[i * numCColumns + j] = temp * alpha + beta * C[i * numCColumns + j];
    }
  }
}

int main(int argc, char **argv) 

{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  printf("Initializing...\n");
  /*auto computeCapability =
      std::stof(dpct::get_default_queue().get_device().get_info<sycl::info::device::backend_version>());*/
 // int dev = findCudaDevice(argc, (const char **)argv);
std::cout << "\nRunning on " << dpct::get_default_queue().get_device().get_info<sycl::info::device::name>()
<<"\n";   

  dpct::device_info deviceProp;
  /*
  DPCT1003:28: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
/*  checkCudaErrors(
      (dpct::dev_mgr::instance().get_device().get_info<sycl::info::device::name>().get_device_info(deviceProp),
       0));*/

  // Tensor cores require a GPU of Volta (SM7X) architecture or higher.
 /* if (computeCapability < 7) {
    printf(
        "cudaTensorCoreGemm requires SM 7.0 or higher to use Tensor "
        "Cores.  Exiting...\n");
    exit(EXIT_WAIVED);
  }*/

  printf("M: %d (%d x %d)\n", M_GLOBAL, M, M_TILES);
  printf("N: %d (%d x %d)\n", N_GLOBAL, N, N_TILES);
  printf("K: %d (%d x %d)\n", K_GLOBAL, K, K_TILES);

  sycl::half *A_h = NULL;
  sycl::half *B_h = NULL;
  float *C_h = NULL;
#if CPU_DEBUG
  float *result_hD = NULL;
  float *result_host = NULL;
#endif

  A_h = (sycl::half *)malloc(sizeof(sycl::half) * M_GLOBAL * K_GLOBAL);
  B_h = (sycl::half *)malloc(sizeof(sycl::half) * K_GLOBAL * N_GLOBAL);
  C_h = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
#if CPU_DEBUG
  result_hD = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
  result_host = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
#endif

  sycl::half *A = NULL;
  sycl::half *B = NULL;
  sycl::half *vnniB = NULL;
  float *C = NULL;
  float *D = NULL;

  /*
  DPCT1003:29: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors(
      (A = (sycl::half *)sycl::malloc_device(
           sizeof(sycl::half) * M_GLOBAL * K_GLOBAL, dpct::get_default_queue()),
       0));
  /*
  DPCT1003:30: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors(
      (B = (sycl::half *)sycl::malloc_shared(
           sizeof(sycl::half) * N_GLOBAL * K_GLOBAL, dpct::get_default_queue()),
       0));

  checkCudaErrors(
      (vnniB = (sycl::half *)sycl::malloc_shared(
           sizeof(sycl::half) * N_GLOBAL * K_GLOBAL, dpct::get_default_queue()),
       0));

  /*
  DPCT1003:31: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors(
      (C = (float *)sycl::malloc_device(sizeof(float) * M_GLOBAL * N_GLOBAL,
                                        dpct::get_default_queue()),
       0));
  /*
  DPCT1003:32: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors(
      (D = (float *)sycl::malloc_device(sizeof(float) * M_GLOBAL * N_GLOBAL,
                                        dpct::get_default_queue()),
       0));

  assert(((unsigned long long)A) % 128 == 0);
  assert(((unsigned long long)B) % 128 == 0);
  assert(((unsigned long long)C) % 128 == 0);
  assert(((unsigned long long)D) % 128 == 0);

  init_host_matrices(A_h, B_h, C_h);

  printf("Preparing data for GPU...\n");

  /*
  DPCT1003:33: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((dpct::get_default_queue()
                       .memcpy(A, A_h, sizeof(sycl::half) * M_GLOBAL * K_GLOBAL)
                       .wait(),
                   0));
  /*
  DPCT1003:34: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((dpct::get_default_queue()
                       .memcpy(B, B_h, sizeof(sycl::half) * N_GLOBAL * K_GLOBAL)
                       .wait(),
                   0));
  /*
  DPCT1003:35: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((dpct::get_default_queue()
                       .memcpy(C, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL)
                       .wait(),
                   0));
  /*
  DPCT1003:36: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((dpct::get_default_queue()
                       .memset(D, 0, sizeof(float) * M_GLOBAL * N_GLOBAL)
                       .wait(),
                   0));

  enum {
    // Compute the right amount of shared memory to request.
    // We need shared memory to hold per-CTA C and D matrix tiles, and to cache
    // per-CTA chunks
    // of the A and B matrices. Therefore, the right amount to request is the
    // maximum of those
    // two numbers.
    SHMEM_SZ = MAX(sizeof(sycl::half) * (BLOCK_COL_TILES * M) *
                       (CHUNK_K * K + SKEW_HALF) * 2,
                   M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N *
                       (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float))
  };

  printf("Required shared memory size: %lu Kb\n", SHMEM_SZ / 1024UL);

  const float alpha = 1.1f;
  const float beta = 1.2f;

  dpct::event_ptr start, stop;
  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;

  /*
  DPCT1003:37: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((start = new sycl::event(), 0));
  /*
  DPCT1003:38: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((stop = new sycl::event(), 0));
  /*
  DPCT1012:39: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  /*
  DPCT1024:40: The original code returned the error code that was further
  consumed by the program logic. This original code was replaced with 0. You may
  need to rewrite the program logic consuming the error code.
  */
  start_ct1 = std::chrono::steady_clock::now();
  checkCudaErrors(0);

  // If enough shared memory available on the GPU use high performant kernel
  /*
  DPCT1019:41: local_mem_size in SYCL is not a complete equivalent of
  sharedMemPerMultiprocessor in CUDA. You may need to adjust the code.
  */
/*  if (deviceProp.get_local_mem_size() >= SHMEM_SZ) {
    printf("Computing... using high performance kernel compute_gemm \n");*/

    
 //   DPCT1007:42: Migration of cudaFuncSetAttribute is not supported.
    
   /* checkCudaErrors(cudaFuncSetAttribute(
        compute_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));*/
    
   /* DPCT1038:6: When the kernel function name is used as a macro argument, the
    migration result may be incorrect. You need to verify the definition of the
    macro.*/
    
    /*(([&]() {
      *stop = dpct::get_default_queue().submit([&](sycl::handler &cgh) {*/
        
/*        DPCT1101:54: 'CHUNK_K * K + SKEW_HALF' expression was replaced with a
        value. Modify the code to use the original expression, provided in
        comments, if it is correct.*/
        
        /*sycl::local_accessor<uint8_t, 2> dpct_local_acc_ct1(
            sycl::range<2>(SHMEM_SZ, CHUNK_K * K + SKEW_HALF) , cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, deviceProp.get_max_compute_units()) *
                    sycl::range<3>(1, 1, THREADS_PER_BLOCK),
                sycl::range<3>(1, 1, THREADS_PER_BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
	    sub_group sg = item_ct1.get_sub_group();
              compute_gemm(A, B, C, D, alpha, beta, item_ct1, dpct_local_acc_ct1.get_pointer(), sg);
            });
      });
    }()));*/
/*#if CPU_DEBUG
    checkCudaErrors(cudaMemcpy(result_hD, D,
                               sizeof(float) * M_GLOBAL * N_GLOBAL,
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(
        (dpct::get_default_queue()
             .memcpy(result_hD, D, sizeof(float) * M_GLOBAL * N_GLOBAL)
             .wait(),
         0));
#endif
  } 
  
  else {*/
  printf("before vnni\n");
   matrix_vnni<half>(K_GLOBAL, N_GLOBAL, B,  vnniB, 2);
   printf("after vnni\n");
    sycl::range<3> gridDim(1, 1, 1);
    sycl::range<3> blockDim(1, 1, 1);
 printf("after sycl blockDim\n");
    // blockDim.x must be a multple of warpSize
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    blockDim[2] = 128;
    blockDim[1] = 4;

    gridDim[2] = (M_GLOBAL + (WMMA_M * blockDim[2] / 32 - 1)) /
                 (WMMA_M * blockDim[2] / 32);
    gridDim[1] = (N_GLOBAL + WMMA_N * blockDim[1] - 1) / (WMMA_N * blockDim[1]);

    printf("Computing... using simple_wmma_gemm kernel\n");
    
    /*DPCT1049:7: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.*/
    
    *stop = dpct::get_default_queue().parallel_for(
        sycl::nd_range<3>(gridDim * blockDim, blockDim),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(WARP_SIZE)]] {
	sub_group sg = item_ct1.get_sub_group();
          simple_wmma_gemm(A, vnniB, C, D, M_GLOBAL, N_GLOBAL, K_GLOBAL, alpha,
                           beta, item_ct1, sg);
        });
#if CPU_DEBUG
/*    checkCudaErrors(cudaMemcpy(result_hD, D,
                               sizeof(float) * M_GLOBAL * N_GLOBAL,
                               cudaMemcpyDeviceToHost));*/
    checkCudaErrors(
        (dpct::get_default_queue()
             .memcpy(result_hD, D, sizeof(float) * M_GLOBAL * N_GLOBAL)
             .wait(),
         0));
#endif
  

  /*
  DPCT1012:47: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  /*
  DPCT1024:48: The original code returned the error code that was further
  consumed by the program logic. This original code was replaced with 0. You may
  need to rewrite the program logic consuming the error code.
  */
  //stop->wait();
  /*stop_ct1 = std::chrono::steady_clock::now();
  checkCudaErrors(0);
  checkCudaErrors(0);*/

#if CPU_DEBUG
  printf("Verifying correctness of the computations...\n");

  memcpy(result_host, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL);

  matMultiplyOnHost(A_h, vnniB, result_host, alpha, beta, M_GLOBAL, K_GLOBAL,
                    K_GLOBAL, N_GLOBAL, M_GLOBAL, N_GLOBAL);

  for (int i = 0; i < N_GLOBAL * M_GLOBAL; i++) {
    if (fabs(result_hD[i] - result_host[i]) > 0.1f)
      printf("mismatch i=%d result_hD=%f result_host=%f\n", i, result_hD[i],
             result_host[i]);
  }
  free(result_hD);
  free(result_host);
#endif

  float milliseconds = 0;

  /*
  DPCT1003:49: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  /*checkCudaErrors((milliseconds = std::chrono::duration<float, std::milli>(
                                      stop_ct1 - start_ct1)
                                      .count(),
                   0));

  printf("Time: %f ms\n", milliseconds);
  printf("TFLOPS: %.2f\n", static_cast<double>((static_cast<double>(M_GLOBAL) *
                                                N_GLOBAL * K_GLOBAL * 2) /
                                               (milliseconds / 1000.)) /
                               1e12);*/

  free(A_h);
  free(B_h);
  free(C_h);
  
  /*DPCT1003:50: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  //checkCudaErrors(
      (sycl::free(A, dpct::get_default_queue()));
  /*
  DPCT1003:51: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  //checkCudaErrors(
      (sycl::free(B, dpct::get_default_queue()));
  /*
  DPCT1003:52: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
//  checkCudaErrors(
      (sycl::free(C, dpct::get_default_queue()));
  /*
  DPCT1003:53: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
//  checkCudaErrors(
      (sycl::free(D, dpct::get_default_queue()));

  return 0;
  }
//}

