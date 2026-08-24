/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <cub/cub.cuh> // order important
#include "forward.h"
#include "auxiliary.h"
#include "timers.cu"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <glm/gtc/type_ptr.hpp>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	// tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;
	// TODO: this is important; it will avoid considering many points that are not in the frustum.

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	// tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS, bool COLLECT_IMPORTANCE>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const uint32_t* __restrict__ per_tile_bucket_offset,
	uint32_t* __restrict__ bucket_to_tile,
	float* __restrict__ sampled_T,
	float* __restrict__ sampled_ar,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	uint32_t* __restrict__ max_contrib,
	uint32_t* __restrict__ n_contrib2loss,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ accum_weights,
	int* __restrict__ projected_area,
	float* __restrict__ max_contribution_area)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;

	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint32_t tile_id = block.group_index().y * horizontal_blocks + block.group_index().x;
	uint2 range = ranges[tile_id];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Per-tile bucket metadata used by per-gaussian backward pass.
	uint32_t bbm = tile_id == 0 ? 0 : per_tile_bucket_offset[tile_id - 1];
	int num_buckets = (toDo + 31) / 32;
	for (int i = 0; i < (num_buckets + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i)
	{
		int bucket_idx = i * BLOCK_SIZE + block.thread_rank();
		if (bucket_idx < num_buckets)
			bucket_to_tile[bbm + bucket_idx] = tile_id;
	}

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	uint32_t thread_n_contrib2loss = 0;
	float C[CHANNELS] = { 0 };
	float max_weight = 0.0f;
	int max_weight_id = -1;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Save sampled forward states every 32 splats for per-gaussian backward.
			if (j % 32 == 0)
			{
				sampled_T[(bbm * BLOCK_SIZE) + block.thread_rank()] = T;
				for (int ch = 0; ch < CHANNELS; ++ch)
					sampled_ar[(bbm * BLOCK_SIZE * CHANNELS) + ch * BLOCK_SIZE + block.thread_rank()] = C[ch];
				++bbm;
			}

			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			thread_n_contrib2loss++;

			const float weight = alpha * T;
			if (COLLECT_IMPORTANCE &&
				(accum_weights != nullptr || projected_area != nullptr || max_contribution_area != nullptr))
			{
				const int gaussian_id = collected_id[j];
				if (accum_weights != nullptr)
					atomicAdd(&accum_weights[gaussian_id], weight);
				if (projected_area != nullptr)
					atomicAdd(&projected_area[gaussian_id], 1);
				if (weight > max_weight)
				{
					max_weight = weight;
					max_weight_id = gaussian_id;
				}
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * weight;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	if (COLLECT_IMPORTANCE && max_contribution_area != nullptr && max_weight_id >= 0)
		atomicAdd(&max_contribution_area[max_weight_id], 1.0f);

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		n_contrib2loss[pix_id] = thread_n_contrib2loss;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}

	// Reduce max contributing index per tile.
	typedef cub::BlockReduce<uint32_t, BLOCK_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_Y> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	last_contributor = BlockReduce(temp_storage).Reduce(last_contributor, cub::Max());
	if (block.thread_rank() == 0)
		max_contrib[tile_id] = last_contributor;
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	const uint32_t* per_tile_bucket_offset,
	uint32_t* bucket_to_tile,
	float* sampled_T,
	float* sampled_ar,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	uint32_t* max_contrib,
	uint32_t* n_contrib2loss,
	const float* bg_color,
	float* out_color,
	float* max_contribution_area)
{
	if (max_contribution_area != nullptr)
		renderCUDA<NUM_CHANNELS, true> << <grid, block >> > (
			ranges,
			point_list,
			per_tile_bucket_offset,
			bucket_to_tile,
			sampled_T,
			sampled_ar,
			W, H,
			means2D,
			colors,
			conic_opacity,
			final_T,
			n_contrib,
			max_contrib,
			n_contrib2loss,
			bg_color,
			out_color,
			nullptr,
			nullptr,
			max_contribution_area);
	else
		renderCUDA<NUM_CHANNELS, false> << <grid, block >> > (
			ranges,
			point_list,
			per_tile_bucket_offset,
			bucket_to_tile,
			sampled_T,
			sampled_ar,
			W, H,
			means2D,
			colors,
			conic_opacity,
			final_T,
			n_contrib,
			max_contrib,
			n_contrib2loss,
			bg_color,
			out_color,
			nullptr,
			nullptr,
			nullptr);
}

void FORWARD::render_importance(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	const uint32_t* per_tile_bucket_offset,
	uint32_t* bucket_to_tile,
	float* sampled_T,
	float* sampled_ar,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	uint32_t* max_contrib,
	uint32_t* n_contrib2loss,
	const float* bg_color,
	float* out_color,
	float* accum_weights,
	int* projected_area,
	float* max_contribution_area)
{
	renderCUDA<NUM_CHANNELS, true> << <grid, block >> > (
		ranges,
		point_list,
		per_tile_bucket_offset,
		bucket_to_tile,
		sampled_T,
		sampled_ar,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		max_contrib,
		n_contrib2loss,
		bg_color,
		out_color,
		accum_weights,
		projected_area,
		max_contribution_area);
}

__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderDepthCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ out_points,
	float* __restrict__ remaining_transmittance,
	const float* __restrict__ means3D,
	const glm::vec3* __restrict__ scales,
	const glm::vec4* __restrict__ rotations,
	const float* __restrict__ projmatrix,
	const glm::vec3* __restrict__ cam_pos)
{
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = {
		block.group_index().x * BLOCK_X,
		block.group_index().y * BLOCK_Y
	};
	const uint2 pix = {
		pix_min.x + block.thread_index().x,
		pix_min.y + block.thread_index().y
	};
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };
	const bool inside = pix.x < W && pix.y < H;
	bool done = !inside;

	const uint32_t tile_id =
		block.group_index().y * horizontal_blocks + block.group_index().x;
	const uint2 range = ranges[tile_id];
	const int rounds = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	float T = 1.0f;
	float max_weight = 0.0f;
	glm::vec3 point_rec = { 0.0f, 0.0f, 0.0f };

	glm::mat4 matrix = glm::make_mat4x4(projmatrix);
	glm::mat4 matrix_inv = glm::inverse(matrix);
	float* projmatrix_inv = glm::value_ptr(matrix_inv);
	const glm::vec3 ray_origin = *cam_pos;

	const float3 p_proj = { Pix2ndc(pixf.x, W), Pix2ndc(pixf.y, H), 1.0f };
	const float3 p_hom = {
		p_proj.x * 1.0000001f,
		p_proj.y * 1.0000001f,
		(100.0f + 0.01f - 1.0f) / (100.0f - 0.01f)
	};
	const float4 p_orig = transformPoint4x4(p_hom, projmatrix_inv);
	const glm::vec3 ray_direction = {
		p_orig.x - ray_origin.x,
		p_orig.y - ray_origin.y,
		p_orig.z - ray_origin.z
	};
	const glm::vec3 normalized_ray_direction = glm::normalize(ray_direction);

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		if (__syncthreads_count(done) == BLOCK_SIZE)
			break;

		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int gaussian_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = gaussian_id;
			collected_xy[block.thread_rank()] = points_xy_image[gaussian_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[gaussian_id];
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f *
				(con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			const float test_T = T * (1.0f - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			const int gaussian_id = collected_id[j];
			const glm::vec4 q = rotations[gaussian_id];
			const float r = q.x;
			const float x = q.y;
			const float y = q.z;
			const float z = q.w;
			const glm::mat3 R = glm::mat3(
				1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
				2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
				2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));

			const glm::vec3 center = {
				means3D[3 * gaussian_id],
				means3D[3 * gaussian_id + 1],
				means3D[3 * gaussian_id + 2]
			};
			const glm::vec3 rotated_origin = R * (ray_origin - center);
			const glm::vec3 rotated_direction = R * normalized_ray_direction;
			const glm::vec3 sigma = scales[gaussian_id] * 3.0f;
			const glm::vec3 direction_scaled = rotated_direction / sigma;
			const glm::vec3 origin_scaled = rotated_origin / sigma;
			const float a = glm::dot(direction_scaled, direction_scaled);
			const float b = 2.0f * glm::dot(direction_scaled, origin_scaled);
			const float depth = -b / (2.0f * a);
			if (depth < 0.0f)
				continue;

			if (max_weight < alpha * T)
			{
				max_weight = alpha * T;
				point_rec = ray_origin + depth * normalized_ray_direction;
			}
			T = test_T;
		}
	}

	if (inside)
	{
		remaining_transmittance[pix_id] = T;
		out_points[pix_id] = point_rec.x;
		out_points[H * W + pix_id] = point_rec.y;
		out_points[2 * H * W + pix_id] = point_rec.z;
	}
}

void FORWARD::render_depth(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float4* conic_opacity,
	float* out_points,
	float* remaining_transmittance,
	const float* means3D,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float* projmatrix,
	const glm::vec3* cam_pos)
{
	renderDepthCUDA << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		conic_opacity,
		out_points,
		remaining_transmittance,
		means3D,
		scales,
		rotations,
		projmatrix,
		cam_pos);
}

template <uint32_t CHANNELS, bool COLLECT_BLUR_STATS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderL1CUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const uint32_t* __restrict__ per_tile_bucket_offset,
	uint32_t* __restrict__ bucket_to_tile,
	float* __restrict__ sampled_T,
	float* __restrict__ sampled_ar,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	uint32_t* __restrict__ max_contrib,
	uint32_t* __restrict__ n_contrib2loss,
	const float* __restrict__ bg_color,
	const float* __restrict__ gt_image,
	int gt_image_y_offset,
	int gt_image_height,
	const bool* __restrict__ loss_compute_locally,
	float lambda_l1,
	float lambda_ssim,
	float* __restrict__ out_loss,
	float* __restrict__ out_color,
	float* __restrict__ dL_dpixels,
	float* __restrict__ max_contribution_area)
{
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };
	const bool inside = pix.x < W && pix.y < H;
	bool done = !inside;
	const int tile_w = min((int)BLOCK_X, W - (int)pix_min.x);
	const int tile_h = min((int)BLOCK_Y, H - (int)pix_min.y);

	const uint32_t tile_id = block.group_index().y * horizontal_blocks + block.group_index().x;
	bool compute_loss = loss_compute_locally == nullptr || loss_compute_locally[tile_id];
	compute_loss = compute_loss
		&& (int)pix_min.y >= gt_image_y_offset
		&& (int)pix_min.y + tile_h <= gt_image_y_offset + gt_image_height;
	const uint2 range = ranges[tile_id];
	const int rounds = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;
	int toDo = range.y - range.x;

	uint32_t bbm = tile_id == 0 ? 0 : per_tile_bucket_offset[tile_id - 1];
	const int num_buckets = (toDo + 31) / 32;
	for (int i = 0; i < (num_buckets + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i)
	{
		const int bucket_idx = i * BLOCK_SIZE + block.thread_rank();
		if (bucket_idx < num_buckets)
			bucket_to_tile[bbm + bucket_idx] = tile_id;
	}

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_features[CHANNELS * BLOCK_SIZE];
	__shared__ float s_x[BLOCK_SIZE];
	__shared__ float s_y[BLOCK_SIZE];
	__shared__ float s_x2[BLOCK_SIZE];
	__shared__ float s_y2[BLOCK_SIZE];
	__shared__ float s_xy[BLOCK_SIZE];

	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	uint32_t thread_n_contrib2loss = 0;
	float C[CHANNELS] = { 0 };
	float max_weight = 0.0f;
	int max_weight_id = -1;

	for (int i = 0; i < rounds; ++i, toDo -= BLOCK_SIZE)
	{
		if (__syncthreads_count(done) == BLOCK_SIZE)
			break;

		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int ch = 0; ch < CHANNELS; ++ch)
				collected_features[ch * BLOCK_SIZE + block.thread_rank()] =
					features[coll_id * CHANNELS + ch];
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); ++j)
		{
			if (j % 32 == 0)
			{
				sampled_T[bbm * BLOCK_SIZE + block.thread_rank()] = T;
				for (int ch = 0; ch < CHANNELS; ++ch)
					sampled_ar[bbm * BLOCK_SIZE * CHANNELS + ch * BLOCK_SIZE + block.thread_rank()] = C[ch];
				++bbm;
			}

			++contributor;
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y)
				- con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			const float test_T = T * (1.0f - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			++thread_n_contrib2loss;
			const float weight = alpha * T;
			if (COLLECT_BLUR_STATS && weight > max_weight)
			{
				max_weight = weight;
				max_weight_id = collected_id[j];
			}
			for (int ch = 0; ch < CHANNELS; ++ch)
				C[ch] += collected_features[ch * BLOCK_SIZE + j] * weight;
			T = test_T;
			last_contributor = contributor;
		}
	}
	if (COLLECT_BLUR_STATS && max_weight_id >= 0)
		atomicAdd(&max_contribution_area[max_weight_id], 1.0f);

	float local_l1_loss = 0.0f;
	float rendered_values[CHANNELS] = { 0.0f };
	float gt_values[CHANNELS] = { 0.0f };
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		n_contrib2loss[pix_id] = thread_n_contrib2loss;
		for (int ch = 0; ch < CHANNELS; ++ch)
		{
			const uint32_t idx = ch * H * W + pix_id;
			const float rendered = C[ch] + T * bg_color[ch];
			out_color[idx] = rendered;
			if (compute_loss)
			{
				const int gt_y = (int)pix.y - gt_image_y_offset;
				const uint32_t gt_idx = ch * gt_image_height * W + gt_y * W + pix.x;
				const float gt = gt_image[gt_idx];
				const float diff = rendered - gt;
				rendered_values[ch] = fminf(fmaxf(rendered, 0.0f), 1.0f);
				gt_values[ch] = gt;
				dL_dpixels[idx] = lambda_l1 *
					(diff > 0.0f ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f));
				local_l1_loss += fabsf(diff);
			}
			else
			{
				dL_dpixels[idx] = 0.0f;
			}
		}
	}

	if (compute_loss && lambda_ssim > 0.0f)
	{
		const int N_valid = tile_w * tile_h;
		const float N_f = (float)N_valid;
		const unsigned r = block.thread_rank();
		if (N_valid >= 4)
		{
			const float C1 = 0.01f * 0.01f;
			const float C2 = 0.03f * 0.03f;
			for (int ch = 0; ch < CHANNELS; ++ch)
			{
				const float x_i = inside ? rendered_values[ch] : 0.0f;
				const float y_i = inside ? gt_values[ch] : 0.0f;
				s_x[r] = x_i;
				s_y[r] = y_i;
				s_x2[r] = x_i * x_i;
				s_y2[r] = y_i * y_i;
				s_xy[r] = x_i * y_i;
				block.sync();

				for (unsigned s = BLOCK_SIZE / 2; s > 0; s >>= 1)
				{
					if (r < s)
					{
						s_x[r] += s_x[r + s];
						s_y[r] += s_y[r + s];
						s_x2[r] += s_x2[r + s];
						s_y2[r] += s_y2[r + s];
						s_xy[r] += s_xy[r + s];
					}
					block.sync();
				}

				const float sum_x = s_x[0];
				const float sum_y = s_y[0];
				const float mu_x = sum_x / N_f;
				const float mu_y = sum_y / N_f;
				const float sigma_x2 = s_x2[0] / N_f - mu_x * mu_x;
				const float sigma_y2 = s_y2[0] / N_f - mu_y * mu_y;
				const float sigma_xy = s_xy[0] / N_f - mu_x * mu_y;
				const float num1 = 2.0f * mu_x * mu_y + C1;
				const float num2 = 2.0f * sigma_xy + C2;
				const float den1 = mu_x * mu_x + mu_y * mu_y + C1;
				const float den2 = sigma_x2 + sigma_y2 + C2;
				const float ssim_ch = num1 * num2 / (den1 * den2);
				if (r == 0)
					atomicAdd(out_loss, lambda_ssim * N_f * (1.0f - ssim_ch));

				if (inside)
				{
					const float inv_den1 = 1.0f / den1;
					const float inv_den2 = 1.0f / den2;
					const float dSSIM_dmu_x = (num2 * inv_den2) * inv_den1 * (2.0f * mu_y)
						- (num1 * num2) * (inv_den1 * inv_den1 * inv_den2) * (2.0f * mu_x);
					const float dSSIM_dsigma_x2 = -ssim_ch * inv_den2;
					const float dSSIM_dsigma_xy = 2.0f * num1 * inv_den1 * inv_den2;
					const float grad = dSSIM_dmu_x
						+ dSSIM_dsigma_x2 * 2.0f * (x_i - mu_x)
						+ dSSIM_dsigma_xy * (y_i - mu_y);
					dL_dpixels[ch * H * W + pix_id] += -lambda_ssim * grad;
				}
				block.sync();
			}
		}
	}

	typedef cub::BlockReduce<uint32_t, BLOCK_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_Y> BlockReduceUint;
	__shared__ typename BlockReduceUint::TempStorage uint_temp_storage;
	last_contributor = BlockReduceUint(uint_temp_storage).Reduce(last_contributor, cub::Max());
	if (block.thread_rank() == 0)
		max_contrib[tile_id] = last_contributor;

	typedef cub::BlockReduce<float, BLOCK_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_Y> BlockReduceFloat;
	__shared__ typename BlockReduceFloat::TempStorage float_temp_storage;
	const float block_l1_loss = BlockReduceFloat(float_temp_storage).Sum(local_l1_loss);
	if (block.thread_rank() == 0)
		atomicAdd(out_loss, lambda_l1 * block_l1_loss);
}

template <uint32_t CHANNELS, bool COLLECT_BLUR_STATS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderL1FusedPerGaussianCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const uint32_t* __restrict__ per_tile_bucket_offset,
	uint32_t* __restrict__ bucket_to_tile,
	float* __restrict__ sampled_T,
	float* __restrict__ sampled_ar,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	uint32_t* __restrict__ max_contrib,
	uint32_t* __restrict__ n_contrib2loss,
	const float* __restrict__ bg_color,
	const float* __restrict__ gt_image,
	int gt_image_y_offset,
	int gt_image_height,
	const bool* __restrict__ loss_compute_locally,
	float lambda_l1,
	float lambda_ssim,
	float* __restrict__ out_loss,
	float* __restrict__ out_color,
	float2* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic_opacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ max_contribution_area)
{
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;

	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	bool inside = pix.x < W && pix.y < H;
	bool done = !inside;
	const int tile_w = min((int)BLOCK_X, W - (int)pix_min.x);
	const int tile_h = min((int)BLOCK_Y, H - (int)pix_min.y);

	uint32_t tile_id = block.group_index().y * horizontal_blocks + block.group_index().x;
	bool compute_loss = loss_compute_locally == nullptr || loss_compute_locally[tile_id];
	compute_loss = compute_loss
		&& (int)pix_min.y >= gt_image_y_offset
		&& (int)pix_min.y + tile_h <= gt_image_y_offset + gt_image_height;
	uint2 range = ranges[tile_id];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	uint32_t bbm = tile_id == 0 ? 0 : per_tile_bucket_offset[tile_id - 1];
	int num_buckets = (toDo + 31) / 32;
	(void)bucket_to_tile;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_features[CHANNELS * BLOCK_SIZE];
	__shared__ float s_x[BLOCK_SIZE];
	__shared__ float s_y[BLOCK_SIZE];
	__shared__ float s_x2[BLOCK_SIZE];
	__shared__ float s_y2[BLOCK_SIZE];
	__shared__ float s_xy[BLOCK_SIZE];

	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	uint32_t thread_n_contrib2loss = 0;
	float C[CHANNELS] = { 0 };
	float max_weight = 0.0f;
	int max_weight_id = -1;

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int ch = 0; ch < CHANNELS; ++ch)
				collected_features[ch * BLOCK_SIZE + block.thread_rank()] =
					features[coll_id * CHANNELS + ch];
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			if (j % 32 == 0)
			{
				sampled_T[(bbm * BLOCK_SIZE) + block.thread_rank()] = T;
				for (int ch = 0; ch < CHANNELS; ++ch)
					sampled_ar[(bbm * BLOCK_SIZE * CHANNELS) + ch * BLOCK_SIZE + block.thread_rank()] = C[ch];
				++bbm;
			}

			contributor++;

			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			thread_n_contrib2loss++;
			const float weight = alpha * T;
			if (COLLECT_BLUR_STATS && weight > max_weight)
			{
				max_weight = weight;
				max_weight_id = collected_id[j];
			}
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += collected_features[ch * BLOCK_SIZE + j] * weight;

			T = test_T;
			last_contributor = contributor;
		}
	}
	if (COLLECT_BLUR_STATS && max_weight_id >= 0)
		atomicAdd(&max_contribution_area[max_weight_id], 1.0f);

	float local_l1_loss = 0.0f;
	float rendered_values[CHANNELS] = { 0.0f };
	float gt_values[CHANNELS] = { 0.0f };
	float pixel_dL[CHANNELS] = { 0.0f };
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		n_contrib2loss[pix_id] = thread_n_contrib2loss;
		for (int ch = 0; ch < CHANNELS; ch++)
		{
			const uint32_t idx = ch * H * W + pix_id;
			const float rendered = C[ch] + T * bg_color[ch];
			out_color[idx] = rendered;
			if (compute_loss)
			{
				const int gt_y = (int)pix.y - gt_image_y_offset;
				const uint32_t gt_idx = ch * gt_image_height * W + gt_y * W + pix.x;
				const float gt = gt_image[gt_idx];
				const float diff = rendered - gt;
				rendered_values[ch] = fminf(fmaxf(rendered, 0.0f), 1.0f);
				gt_values[ch] = gt;
				pixel_dL[ch] = lambda_l1 * (diff > 0.0f ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f));
				local_l1_loss += fabsf(diff);
			}
		}
	}

	if (compute_loss && lambda_ssim > 0.0f)
	{
		const int N_valid = tile_w * tile_h;
		const float N_f = (float)N_valid;
		const unsigned r = block.thread_rank();

		if (N_valid >= 4)
		{
			const float C1 = 0.01f * 0.01f;
			const float C2 = 0.03f * 0.03f;

			for (int ch = 0; ch < CHANNELS; ++ch)
			{
				const float x_i = inside ? rendered_values[ch] : 0.0f;
				const float y_i = inside ? gt_values[ch] : 0.0f;

				s_x[r] = x_i;
				s_y[r] = y_i;
				s_x2[r] = x_i * x_i;
				s_y2[r] = y_i * y_i;
				s_xy[r] = x_i * y_i;
				block.sync();

				for (unsigned s = BLOCK_SIZE / 2; s > 0; s >>= 1)
				{
					if (r < s)
					{
						s_x[r] += s_x[r + s];
						s_y[r] += s_y[r + s];
						s_x2[r] += s_x2[r + s];
						s_y2[r] += s_y2[r + s];
						s_xy[r] += s_xy[r + s];
					}
					block.sync();
				}

				const float sum_x = s_x[0];
				const float sum_y = s_y[0];
				const float sum_x2 = s_x2[0];
				const float sum_y2 = s_y2[0];
				const float sum_xy = s_xy[0];

				const float mu_x = sum_x / N_f;
				const float mu_y = sum_y / N_f;
				const float sigma_x2 = sum_x2 / N_f - mu_x * mu_x;
				const float sigma_y2 = sum_y2 / N_f - mu_y * mu_y;
				const float sigma_xy = sum_xy / N_f - mu_x * mu_y;

				const float num1 = 2.0f * mu_x * mu_y + C1;
				const float num2 = 2.0f * sigma_xy + C2;
				const float den1 = mu_x * mu_x + mu_y * mu_y + C1;
				const float den2 = sigma_x2 + sigma_y2 + C2;
				const float ssim_ch = (num1 * num2) / (den1 * den2);
				const float ssim_component = N_f * (1.0f - ssim_ch);

				if (r == 0)
				{
					atomicAdd(out_loss, lambda_ssim * ssim_component);
				}

				if (inside)
				{
					const float inv_den1 = 1.0f / den1;
					const float inv_den2 = 1.0f / den2;
					const float dSSIM_dmu_x = (num2 * inv_den2) * inv_den1 * (2.0f * mu_y)
						- (num1 * num2) * (inv_den1 * inv_den1 * inv_den2) * (2.0f * mu_x);
					const float dSSIM_dsigma_x2 = -ssim_ch * inv_den2;
					const float dSSIM_dsigma_xy = 2.0f * num1 * inv_den1 * inv_den2;
					const float grad = dSSIM_dmu_x
						+ dSSIM_dsigma_x2 * 2.0f * (x_i - mu_x)
						+ dSSIM_dsigma_xy * (y_i - mu_y);
					pixel_dL[ch] += -lambda_ssim * grad;
				}
				block.sync();
			}
		}
	}

	typedef cub::BlockReduce<uint32_t, BLOCK_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_Y> BlockReduceUint;
	__shared__ typename BlockReduceUint::TempStorage uint_temp_storage;
	last_contributor = BlockReduceUint(uint_temp_storage).Reduce(last_contributor, cub::Max());
	if (block.thread_rank() == 0)
		max_contrib[tile_id] = last_contributor;

	typedef cub::BlockReduce<float, BLOCK_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_Y> BlockReduceFloat;
	__shared__ typename BlockReduceFloat::TempStorage float_temp_storage;
	const float block_l1_loss = BlockReduceFloat(float_temp_storage).Sum(local_l1_loss);
	if (block.thread_rank() == 0)
	{
		atomicAdd(out_loss, lambda_l1 * block_l1_loss);
	}

	// The forward Gaussian cache is dead at this point. Reuse its storage for
	// per-pixel loss gradients consumed by the per-Gaussian phase below.
	block.sync();
	for (int ch = 0; ch < CHANNELS; ++ch)
		collected_features[ch * BLOCK_SIZE + block.thread_rank()] = pixel_dL[ch];
	block.sync();

	if (!compute_loss || num_buckets == 0)
		return;

	// Adapt the original warp-per-bucket PerGaussianRenderCUDA organization to
	// this tile-owned block. Eight warps cooperatively walk all buckets of the
	// current tile, so no grid-wide synchronization is needed between phases.
	auto warp = cg::tiled_partition<32>(block);
	const int lane = warp.thread_rank();
	const int warps_per_block = warp.meta_group_size();
	const int warp_id = warp.meta_group_rank();
	const float ddelx_dx = 0.5f * W;
	const float ddely_dy = 0.5f * H;
	const uint32_t first_bucket = tile_id == 0 ? 0 : per_tile_bucket_offset[tile_id - 1];

	for (int bucket_idx = warp_id; bucket_idx < num_buckets; bucket_idx += warps_per_block)
	{
		if (bucket_idx * 32 >= (int)max_contrib[tile_id])
			continue;

		const uint32_t global_bucket_idx = first_bucket + bucket_idx;
		const int splat_idx_in_tile = bucket_idx * 32 + lane;
		const int splat_idx_global = range.x + splat_idx_in_tile;
		const bool valid_splat = splat_idx_in_tile < (int)(range.y - range.x);

		int gaussian_idx = 0;
		float2 xy = { 0.0f, 0.0f };
		float4 con_o = { 0.0f, 0.0f, 0.0f, 0.0f };
		float color[CHANNELS] = { 0.0f };
		if (valid_splat)
		{
			gaussian_idx = point_list[splat_idx_global];
			xy = points_xy_image[gaussian_idx];
			con_o = conic_opacity[gaussian_idx];
			for (int ch = 0; ch < CHANNELS; ++ch)
				color[ch] = features[gaussian_idx * CHANNELS + ch];
		}

		float grad_mean_x = 0.0f;
		float grad_mean_y = 0.0f;
		float grad_conic_x = 0.0f;
		float grad_conic_y = 0.0f;
		float grad_conic_z = 0.0f;
		float grad_opacity = 0.0f;
		float grad_color[CHANNELS] = { 0.0f };
		float prefix_T = 0.0f;
		float final_pixel_T = 0.0f;
		int pixel_last_contributor = 0;
		float accumulated_remainder[CHANNELS] = { 0.0f };
		float pixel_loss_grad[CHANNELS] = { 0.0f };

		for (int i = 0; i < BLOCK_SIZE + 31; ++i)
		{
			prefix_T = warp.shfl_up(prefix_T, 1);
			final_pixel_T = warp.shfl_up(final_pixel_T, 1);
			pixel_last_contributor = warp.shfl_up(pixel_last_contributor, 1);
			for (int ch = 0; ch < CHANNELS; ++ch)
			{
				accumulated_remainder[ch] = warp.shfl_up(accumulated_remainder[ch], 1);
				pixel_loss_grad[ch] = warp.shfl_up(pixel_loss_grad[ch], 1);
			}

			const int pixel_in_tile = i - lane;
			const uint2 backward_pix = {
				pix_min.x + pixel_in_tile % BLOCK_X,
				pix_min.y + pixel_in_tile / BLOCK_X
			};
			const bool valid_pixel = pixel_in_tile >= 0 && pixel_in_tile < BLOCK_SIZE
				&& backward_pix.x < (uint32_t)W && backward_pix.y < (uint32_t)H;
			const uint32_t backward_pix_id = W * backward_pix.y + backward_pix.x;

			if (lane == 0 && valid_pixel)
			{
				prefix_T = sampled_T[global_bucket_idx * BLOCK_SIZE + pixel_in_tile];
				final_pixel_T = final_T[backward_pix_id];
				pixel_last_contributor = n_contrib[backward_pix_id];
				for (int ch = 0; ch < CHANNELS; ++ch)
				{
					accumulated_remainder[ch] =
						-(out_color[ch * H * W + backward_pix_id] - final_pixel_T * bg_color[ch])
						+ sampled_ar[global_bucket_idx * BLOCK_SIZE * CHANNELS
							+ ch * BLOCK_SIZE + pixel_in_tile];
					pixel_loss_grad[ch] = collected_features[ch * BLOCK_SIZE + pixel_in_tile];
				}
			}

			if (!valid_splat || !valid_pixel || splat_idx_in_tile >= pixel_last_contributor)
				continue;

			const float2 backward_pixf = { (float)backward_pix.x, (float)backward_pix.y };
			const float2 d = { xy.x - backward_pixf.x, xy.y - backward_pixf.y };
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y)
				- con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;
			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			const float dchannel_dcolor = alpha * prefix_T;
			float bg_dot_grad = 0.0f;
			float dL_dalpha = 0.0f;
			for (int ch = 0; ch < CHANNELS; ++ch)
			{
				accumulated_remainder[ch] += prefix_T * alpha * color[ch];
				grad_color[ch] += dchannel_dcolor * pixel_loss_grad[ch];
				dL_dalpha += (color[ch] * prefix_T
					- (-accumulated_remainder[ch]) / (1.0f - alpha)) * pixel_loss_grad[ch];
				bg_dot_grad += bg_color[ch] * pixel_loss_grad[ch];
			}
			dL_dalpha += (-final_pixel_T / (1.0f - alpha)) * bg_dot_grad;
			prefix_T *= (1.0f - alpha);

			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;
			grad_mean_x += dL_dG * dG_ddelx * ddelx_dx;
			grad_mean_y += dL_dG * dG_ddely * ddely_dy;
			grad_conic_x += -0.5f * gdx * d.x * dL_dG;
			grad_conic_y += -0.5f * gdx * d.y * dL_dG;
			grad_conic_z += -0.5f * gdy * d.y * dL_dG;
			grad_opacity += G * dL_dalpha;
		}

		if (valid_splat)
		{
			atomicAdd(&dL_dmean2D[gaussian_idx].x, grad_mean_x);
			atomicAdd(&dL_dmean2D[gaussian_idx].y, grad_mean_y);
			atomicAdd(&dL_dconic_opacity[gaussian_idx].x, grad_conic_x);
			atomicAdd(&dL_dconic_opacity[gaussian_idx].y, grad_conic_y);
			atomicAdd(&dL_dconic_opacity[gaussian_idx].z, grad_conic_z);
			atomicAdd(&dL_dconic_opacity[gaussian_idx].w, grad_opacity);
			for (int ch = 0; ch < CHANNELS; ++ch)
				atomicAdd(&dL_dcolors[gaussian_idx * CHANNELS + ch], grad_color[ch]);
		}
	}
}

void FORWARD::render_l1(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	const uint32_t* per_tile_bucket_offset,
	uint32_t* bucket_to_tile,
	float* sampled_T,
	float* sampled_ar,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	uint32_t* max_contrib,
	uint32_t* n_contrib2loss,
	const float* bg_color,
	const float* gt_image,
	int gt_image_y_offset,
	int gt_image_height,
	const bool* loss_compute_locally,
	float lambda_l1,
	float lambda_ssim,
	float* out_loss,
	float* out_color,
	float* dL_dpixels,
	float* max_contribution_area)
{
	if (max_contribution_area != nullptr)
		renderL1CUDA<NUM_CHANNELS, true><<<grid, block>>>(
			ranges, point_list, per_tile_bucket_offset, bucket_to_tile,
			sampled_T, sampled_ar, W, H, means2D, colors, conic_opacity,
			final_T, n_contrib, max_contrib, n_contrib2loss, bg_color,
			gt_image, gt_image_y_offset, gt_image_height, loss_compute_locally,
			lambda_l1, lambda_ssim, out_loss, out_color, dL_dpixels,
			max_contribution_area);
	else
		renderL1CUDA<NUM_CHANNELS, false><<<grid, block>>>(
			ranges, point_list, per_tile_bucket_offset, bucket_to_tile,
			sampled_T, sampled_ar, W, H, means2D, colors, conic_opacity,
			final_T, n_contrib, max_contrib, n_contrib2loss, bg_color,
			gt_image, gt_image_y_offset, gt_image_height, loss_compute_locally,
			lambda_l1, lambda_ssim, out_loss, out_color, dL_dpixels, nullptr);
}

void FORWARD::render_l1_fused_per_gaussian(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	const uint32_t* per_tile_bucket_offset,
	uint32_t* bucket_to_tile,
	float* sampled_T,
	float* sampled_ar,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	uint32_t* max_contrib,
	uint32_t* n_contrib2loss,
	const float* bg_color,
	const float* gt_image,
	int gt_image_y_offset,
	int gt_image_height,
	const bool* loss_compute_locally,
	float lambda_l1,
	float lambda_ssim,
	float* out_loss,
	float* out_color,
	float2* dL_dmean2D,
	float4* dL_dconic_opacity,
	float* dL_dcolors,
	float* max_contribution_area)
{
	if (max_contribution_area != nullptr)
		renderL1FusedPerGaussianCUDA<NUM_CHANNELS, true> << <grid, block >> > (
			ranges,
			point_list,
			per_tile_bucket_offset,
			bucket_to_tile,
			sampled_T,
			sampled_ar,
			W, H,
			means2D,
			colors,
			conic_opacity,
			final_T,
			n_contrib,
			max_contrib,
			n_contrib2loss,
			bg_color,
			gt_image,
			gt_image_y_offset,
			gt_image_height,
			loss_compute_locally,
			lambda_l1,
			lambda_ssim,
			out_loss,
			out_color,
			dL_dmean2D,
			dL_dconic_opacity,
			dL_dcolors,
			max_contribution_area);
	else
		renderL1FusedPerGaussianCUDA<NUM_CHANNELS, false> << <grid, block >> > (
			ranges,
			point_list,
			per_tile_bucket_offset,
			bucket_to_tile,
			sampled_T,
			sampled_ar,
			W, H,
			means2D,
			colors,
			conic_opacity,
			final_T,
			n_contrib,
			max_contrib,
			n_contrib2loss,
			bg_color,
			gt_image,
			gt_image_y_offset,
			gt_image_height,
			loss_compute_locally,
			lambda_l1,
			lambda_ssim,
			out_loss,
			out_color,
			dL_dmean2D,
			dL_dconic_opacity,
			dL_dcolors,
			nullptr);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + ONE_DIM_BLOCK_SIZE - 1) / ONE_DIM_BLOCK_SIZE, ONE_DIM_BLOCK_SIZE >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}
