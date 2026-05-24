/*
 * Fused forward render + block-local L1/SSIM loss + render backward path.
 */

#include "fused_render.h"
#include "config.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderL1SSIMBackwardCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ target,
	int target_height,
	int target_y_offset,
	float loss_normalizer,
	float lambda_l1,
	float lambda_ssim,
	const bool* __restrict__ compute_locally,
	float* __restrict__ loss,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors)
{
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint32_t block_id = block.group_index().y * horizontal_blocks + block.group_index().x;
	if (!compute_locally[block_id])
		return;

	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const int target_y = (int)pix.y - target_y_offset;
	const bool inside = pix.x < W && pix.y < H && target_y >= 0 && target_y < target_height;
	const int target_pix_id = W * target_y + (int)pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	bool done = !inside;
	const uint2 range = ranges[block_id];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];

	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C_out[C] = { 0 };

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		const int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int ch = 0; ch < C; ch++)
				collected_colors[ch * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + ch];
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			contributor++;

			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
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

			for (int ch = 0; ch < C; ch++)
				C_out[ch] += collected_colors[ch * BLOCK_SIZE + j] * alpha * T;

			T = test_T;
			last_contributor = contributor;
		}
	}

	float thread_loss = 0.0f;
	float dL_dpixel[C] = { 0 };
	const float T_final = T;

	if (inside)
	{
		for (int ch = 0; ch < C; ch++)
		{
			const float rendered = C_out[ch] + T_final * bg_color[ch];
			const float diff = rendered - target[ch * target_height * W + target_pix_id];
			thread_loss += lambda_l1 * fabsf(diff) / loss_normalizer;
			dL_dpixel[ch] = lambda_l1 * (diff > 0.0f ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f)) / loss_normalizer;
		}
	}

	__shared__ float reduction_s[BLOCK_SIZE];
	cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);
	reduction_s[block.thread_rank()] = cg::reduce(tile, thread_loss, cg::plus<float>());
	block.sync();
	if (block.thread_rank() == 0)
	{
		float block_loss = 0.0f;
		for (int i = 0; i < BLOCK_SIZE; i += tile.num_threads())
			block_loss += reduction_s[i];
		atomicAdd(loss, block_loss);
	}

	if (lambda_ssim > 0.0f)
	{
		const int tile_w = min((int)BLOCK_X, W - (int)pix_min.x);
		const int tile_h = min((int)BLOCK_Y, H - (int)pix_min.y);
		const int N_valid = tile_w * tile_h;
		const float N_f = (float)N_valid;
		const unsigned r = block.thread_rank();

		if (N_valid >= 4)
		{
			const float C1 = 0.01f * 0.01f;
			const float C2 = 0.03f * 0.03f;

			float* s_x = reinterpret_cast<float*>(collected_conic_opacity);
			float* s_y = s_x + BLOCK_SIZE;
			float* s_x2 = s_y + BLOCK_SIZE;
			float* s_y2 = s_x2 + BLOCK_SIZE;
			float* s_xy = reinterpret_cast<float*>(collected_colors);

			for (int ch = 0; ch < C; ch++)
			{
				float x_i = 0.0f;
				float y_i = 0.0f;
				if (inside)
				{
					x_i = fminf(fmaxf(C_out[ch] + T_final * bg_color[ch], 0.0f), 1.0f);
					y_i = target[ch * target_height * W + target_pix_id];
				}

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

				if (r == 0)
					atomicAdd(loss, lambda_ssim * N_f * (1.0f - ssim_ch) / loss_normalizer);

				if (inside)
				{
					const float inv_den1 = 1.0f / den1;
					const float inv_den2 = 1.0f / den2;
					const float dSSIM_dmu_x =
						(num2 * inv_den2) * inv_den1 * (2.0f * mu_y) -
						(num1 * num2) * (inv_den1 * inv_den1 * inv_den2) * (2.0f * mu_x);
					const float dSSIM_dsigma_x2 = -ssim_ch * inv_den2;
					const float dSSIM_dsigma_xy = 2.0f * num1 * inv_den1 * inv_den2;
					const float grad =
						dSSIM_dmu_x +
						dSSIM_dsigma_x2 * 2.0f * (x_i - mu_x) +
						dSSIM_dsigma_xy * (y_i - mu_y);

					dL_dpixel[ch] += -lambda_ssim * grad / loss_normalizer;
				}
				block.sync();
			}
		}
	}

	done = !inside;
	toDo = range.y - range.x;
	T = T_final;
	contributor = toDo;
	float accum_rec[C] = { 0 };
	float last_alpha = 0.0f;
	float last_color[C] = { 0 };
	const float ddelx_dx = 0.5f * W;
	const float ddely_dy = 0.5f * H;

	float bg_dot_dpixel = 0.0f;
	if (inside)
		for (int ch = 0; ch < C; ch++)
			bg_dot_dpixel += bg_color[ch] * dL_dpixel[ch];

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int ch = 0; ch < C; ch++)
				collected_colors[ch * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + ch];
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			contributor--;
			if (contributor >= last_contributor)
				continue;

			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.0f - alpha);
			const float dchannel_dcolor = alpha * T;

			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				accum_rec[ch] = last_alpha * last_color[ch] + (1.0f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}
			dL_dalpha *= T;
			last_alpha = alpha;
			dL_dalpha += (-T_final / (1.0f - alpha)) * bg_dot_dpixel;

			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
}

void FUSED::render_l1_ssim_backward(
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
	float* dL_dcolors)
{
	renderL1SSIMBackwardCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		target,
		target_height,
		target_y_offset,
		loss_normalizer,
		lambda_l1,
		lambda_ssim,
		compute_locally,
		loss,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors);
}
