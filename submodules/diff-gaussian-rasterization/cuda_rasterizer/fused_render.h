/*
 * Fused forward render + block-local L1/SSIM loss + render backward path.
 */

#ifndef CUDA_RASTERIZER_FUSED_RENDER_H_INCLUDED
#define CUDA_RASTERIZER_FUSED_RENDER_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace FUSED
{
	void render_l1_ssim_backward(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float* bg_color,
		const float2* means2D,
		const float4* conic_opacity,
		const float* colors,
		const float* target,
		int target_height,
		int target_y_offset,
		float loss_normalizer,
		float lambda_l1,
		float lambda_ssim,
		const bool* compute_locally,
		float* loss,
		float3* dL_dmean2D,
		float4* dL_dconic2D,
		float* dL_dopacity,
		float* dL_dcolors);
}

#endif
