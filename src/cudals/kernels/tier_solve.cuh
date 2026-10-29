/**
 * Three-tier SPD solve (thread / warp / block) over GLASS, for the cudals solvers.
 *
 *   Tier::Thread  1 problem/thread   glass::thread::potrf + potrs
 *   Tier::Warp    1 problem/warp     glass::warp::potrf + 2x warp::trsv
 *   Tier::Block   1 problem/block    glass::posv<T,N,1,false,CHECK>
 *
 * GLASS is column-major and the solvers build `A` row-major; for the (symmetric) normal
 * equations the two readings are the same matrix. Do not use this for a non-symmetric A.
 *
 * Derived from GRiM's glass_solve.cuh (A2R Lab, MIT); only the parts cost_lm.cu uses.
 */

#pragma once

#include "glass.cuh"

#include <cstdint>

namespace cudals {

enum class Tier { Thread, Warp, Block };

// Above this size only the block tier fits: the warp tier's per-warp shared double A[N*N]
// overflows the 48KB block budget (a hard ptxas error past 32), and the thread tier's
// per-lane copy collapses occupancy.
constexpr int TIER_CHOICE_MAX_N = 32;

/**
 * Solve `A x = b` at the given tier; on return `b` holds `x` and `A` its Cholesky factor.
 * On a non-positive-definite pivot `b` is zeroed and false returned (an LM caller raises
 * lambda and retries). Residency: thread-local at Thread, shared per warp at Warp, shared
 * per block at Block. `s_fail` is a shared slot at the cooperative tiers.
 */
template <Tier TIER, typename T, uint32_t N>
__device__ __forceinline__ bool tier_posv(T* __restrict__ A, T* __restrict__ b, int* s_fail)
{
    static_assert(TIER == Tier::Block || (int)N <= TIER_CHOICE_MAX_N,
                  "Only Tier::Block is supported above TIER_CHOICE_MAX_N (=32).");

    if constexpr (TIER == Tier::Thread) {
        int fail = 0;
        glass::thread::potrf<T, N, /*CHECK=*/true>(A, &fail);
        if (fail) { for (uint32_t i = 0; i < N; i++) b[i] = static_cast<T>(0); return false; }
        glass::thread::potrs<T, N>(A, b);
        return true;
    } else if constexpr (TIER == Tier::Warp) {
        const uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x) & 31u;
        if (lane == 0) *s_fail = 0;
        __syncwarp();
        glass::warp::potrf<T, N, /*CHECK=*/true>(A, s_fail);
        __syncwarp();
        if (*s_fail) {
            for (uint32_t i = lane; i < N; i += 32u) b[i] = static_cast<T>(0);
            __syncwarp();
            return false;
        }
        glass::warp::trsv<T, N, glass::FillMode::Lower, glass::Diag::NonUnit, /*TRANSPOSE=*/false>(A, b);
        glass::warp::trsv<T, N, glass::FillMode::Lower, glass::Diag::NonUnit, /*TRANSPOSE=*/true >(A, b);
        __syncwarp();
        return true;
    } else {
        // The flags live on the multi-RHS overload, so a flagged single-RHS solve is NRHS=1.
        glass::posv<T, N, /*NRHS=*/1, /*REGULARIZE=*/false, /*CHECK=*/true>(A, b, T(0), s_fail);
        __syncthreads();
        if (*s_fail) {
            const uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x;
            const uint32_t size = blockDim.x * blockDim.y;
            for (uint32_t i = rank; i < N; i += size) b[i] = static_cast<T>(0);
            __syncthreads();
            return false;
        }
        return true;
    }
}

}  // namespace cudals
