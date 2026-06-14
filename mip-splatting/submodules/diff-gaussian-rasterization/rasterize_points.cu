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

#include <torch/extension.h>

#ifdef snprintf
#undef snprintf
#endif

#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/utils.h"
//#include "cuda_rasterizer/auxiliary.h"
#include <math.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <functional>
#include <stdexcept>

std::function<char *(size_t N)> resizeFunctional(torch::Tensor &t)
{
	auto lambda = [&t](size_t N)
	{
		t.resize_({(long long)N});
		return reinterpret_cast<char *>(t.contiguous().data_ptr());
	};
	return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor &background,
	const torch::Tensor &means3D,
	const torch::Tensor &colors,
	const torch::Tensor &opacity,
	const torch::Tensor &scales,
	const torch::Tensor &rotations,
	const float scale_modifier,
	const torch::Tensor &cov3D_precomp,
	const torch::Tensor& view2gaussian_precomp,
	const torch::Tensor &viewmatrix,
	const torch::Tensor &projmatrix,
	const torch::Tensor &inv_viewprojmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const float kernel_size,
	const torch::Tensor &subpixel_offset,
	const int image_height,
	const int image_width,
	const torch::Tensor &sh,
	const int degree,
	const torch::Tensor &campos,
	const bool prefiltered,
	const nlohmann::json& settings_dict,
	const nlohmann::json& debug_dict,
	const torch::Tensor &filter_3d,
	const bool debug)
{
	if (means3D.ndimension() != 2 || means3D.size(1) != 3)
	{
		AT_ERROR("means3D must have dimensions (num_points, 3)");
	}

	const int P = means3D.size(0);
	const int H = image_height;
	const int W = image_width;
	const long long pixel_count = static_cast<long long>(H) * static_cast<long long>(W);

	if (P < 0)
	{
		throw std::runtime_error("RasterizeGaussiansCUDA received a negative point count.");
	}
	if (H <= 0 || W <= 0)
	{
		std::stringstream ss;
		ss << "RasterizeGaussiansCUDA received a non-positive image size: H=" << H << ", W=" << W;
		throw std::runtime_error(ss.str());
	}
	if (H > 65536 || W > 65536 || pixel_count > 1000000000LL)
	{
		std::stringstream ss;
		ss << "RasterizeGaussiansCUDA received an implausible image size: H=" << H
		   << ", W=" << W << ", pixels=" << pixel_count;
		throw std::runtime_error(ss.str());
	}
	if (subpixel_offset.numel() != 0)
	{
		const long long expected_subpixel = pixel_count * 2;
		if (subpixel_offset.numel() != expected_subpixel)
		{
			std::stringstream ss;
			ss << "RasterizeGaussiansCUDA received a mismatched subpixel offset tensor: numel="
			   << subpixel_offset.numel() << ", expected=" << expected_subpixel
			   << " for H=" << H << ", W=" << W;
			throw std::runtime_error(ss.str());
		}
	}

	auto int_opts = means3D.options().dtype(torch::kInt32);
	auto float_opts = means3D.options().dtype(torch::kFloat32);

	torch::Tensor out_color = torch::full({10, H, W}, 0.0, float_opts);
	torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
	torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
	std::function<char *(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char *(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char *(size_t)> imgFunc = resizeFunctional(imgBuffer);

	int rendered = 0;
	if (P != 0)
	{
		int M = 0;
		if (sh.size(0) != 0)
		{
			M = sh.size(1);
		}

		const float* sh_ptr = sh.numel() ? sh.contiguous().data_ptr<float>() : nullptr;
		const float* colors_ptr = colors.numel() ? colors.contiguous().data_ptr<float>() : nullptr;
		const float* opacity_ptr = opacity.numel() ? opacity.contiguous().data_ptr<float>() : nullptr;
		const float* scales_ptr = scales.numel() ? scales.contiguous().data_ptr<float>() : nullptr;
		const float* rotations_ptr = rotations.numel() ? rotations.contiguous().data_ptr<float>() : nullptr;
		const float* cov3d_ptr = cov3D_precomp.numel() ? cov3D_precomp.contiguous().data_ptr<float>() : nullptr;
		const float* view2gaussian_ptr = view2gaussian_precomp.numel() ? view2gaussian_precomp.contiguous().data_ptr<float>() : nullptr;
		const float* filter3d_ptr = filter_3d.numel() ? filter_3d.contiguous().data_ptr<float>() : nullptr;
		const float* subpixel_ptr = subpixel_offset.numel() ? subpixel_offset.contiguous().data_ptr<float>() : nullptr;

		CudaRasterizer::SplattingSettings settings = settings_dict.get<CudaRasterizer::SplattingSettings>();
		CudaRasterizer::DebugVisualizationData debug_data = debug_dict.get<CudaRasterizer::DebugVisualizationData>();

		rendered = CudaRasterizer::Rasterizer::forward(
			geomFunc,
			binningFunc,
			imgFunc,
			P, degree, M,
			background.contiguous().data<float>(),
			W, H,
			settings,
			debug_data,
			means3D.contiguous().data<float>(),
			sh_ptr,
			colors_ptr,
			opacity_ptr,
			scales_ptr,
			scale_modifier,
			rotations_ptr,
			cov3d_ptr,
			view2gaussian_ptr,
			viewmatrix.contiguous().data<float>(),
			projmatrix.contiguous().data<float>(),
			inv_viewprojmatrix.contiguous().data<float>(),
			filter3d_ptr,
			campos.contiguous().data<float>(),
			tan_fovx,
			tan_fovy,
			kernel_size,
			subpixel_ptr,
			prefiltered,
			out_color.contiguous().data<float>(),
			radii.contiguous().data<int>(),
			debug);
	}
	return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
	const torch::Tensor &background,
	const torch::Tensor &means3D,
	const torch::Tensor &radii,
	const torch::Tensor &opacities,
	const torch::Tensor &colors,
	const torch::Tensor &scales,
	const torch::Tensor &rotations,
	const float scale_modifier,
	const torch::Tensor &cov3D_precomp,
	const torch::Tensor& view2gaussian_precomp,
	const torch::Tensor &viewmatrix,
	const torch::Tensor &projmatrix,
	const torch::Tensor &inv_viewprojmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const float kernel_size,
	const torch::Tensor &subpixel_offset,
	const torch::Tensor &pixel_colors,
	const torch::Tensor &dL_dout_color,
	const torch::Tensor &sh,
	const int degree,
	const torch::Tensor &campos,
	const torch::Tensor &geomBuffer,
	const int R,
	const torch::Tensor &binningBuffer,
	const torch::Tensor &imageBuffer,
	const nlohmann::json& settings_dict,
	const torch::Tensor &filter_3d,
	const bool debug)
{
	const int P = means3D.size(0);
	const int H = dL_dout_color.size(1);
	const int W = dL_dout_color.size(2);

	int M = 0;
	if (sh.size(0) != 0)
	{
		M = sh.size(1);
	}

	const float* sh_ptr = sh.numel() ? sh.contiguous().data_ptr<float>() : nullptr;
	const float* colors_ptr = colors.numel() ? colors.contiguous().data_ptr<float>() : nullptr;
	const float* opacities_ptr = opacities.numel() ? opacities.contiguous().data_ptr<float>() : nullptr;
	const float* scales_ptr = scales.numel() ? scales.contiguous().data_ptr<float>() : nullptr;
	const float* rotations_ptr = rotations.numel() ? rotations.contiguous().data_ptr<float>() : nullptr;
	const float* cov3d_ptr = cov3D_precomp.numel() ? cov3D_precomp.contiguous().data_ptr<float>() : nullptr;
	const float* view2gaussian_ptr = view2gaussian_precomp.numel() ? view2gaussian_precomp.contiguous().data_ptr<float>() : nullptr;
	const float* filter3d_ptr = filter_3d.numel() ? filter_3d.contiguous().data_ptr<float>() : nullptr;
	const float* subpixel_ptr = subpixel_offset.numel() ? subpixel_offset.contiguous().data_ptr<float>() : nullptr;

	torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
	torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
	torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
	torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
	torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
	torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
	torch::Tensor dL_dview2gaussian = torch::zeros({P, 10}, means3D.options());


	CudaRasterizer::SplattingSettings settings = settings_dict.get<CudaRasterizer::SplattingSettings>();


	if (P != 0)
	{
		CudaRasterizer::Rasterizer::backward(P, degree, M, R,
											 background.contiguous().data<float>(),
											 W, H,
											 settings,
											 means3D.contiguous().data<float>(),
											 sh_ptr,
											 opacities_ptr,
											 colors_ptr,
											 view2gaussian_ptr,
											 scales_ptr,
											 scale_modifier,
											 rotations_ptr,
											 cov3d_ptr,
											 viewmatrix.contiguous().data<float>(),
											 projmatrix.contiguous().data<float>(),
											 inv_viewprojmatrix.contiguous().data<float>(),
											 filter3d_ptr,
											 campos.contiguous().data<float>(),
											 tan_fovx,
											 tan_fovy,
											 kernel_size,
											 subpixel_ptr,
											 pixel_colors.contiguous().data<float>(),
											 radii.contiguous().data<int>(),
											 reinterpret_cast<char *>(geomBuffer.contiguous().data_ptr()),
											 reinterpret_cast<char *>(binningBuffer.contiguous().data_ptr()),
											 reinterpret_cast<char *>(imageBuffer.contiguous().data_ptr()),
											 dL_dout_color.contiguous().data<float>(),
											 dL_dmeans2D.contiguous().data<float>(),
											 dL_dconic.contiguous().data<float>(),
											 dL_dopacity.contiguous().data<float>(),
											 dL_dcolors.contiguous().data<float>(),
											 dL_dmeans3D.contiguous().data<float>(),
											 dL_dcov3D.contiguous().data<float>(),
											 dL_dsh.contiguous().data<float>(),
											 dL_dscales.contiguous().data<float>(),
											 dL_drotations.contiguous().data<float>(),
											 dL_dview2gaussian.contiguous().data<float>(),
											 debug);
	}

	return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations, dL_dview2gaussian);
}

torch::Tensor markVisible(
	torch::Tensor &means3D,
	torch::Tensor &viewmatrix,
	torch::Tensor &projmatrix)
{
	const int P = means3D.size(0);

	torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));

	if (P != 0)
	{
		CudaRasterizer::Rasterizer::markVisible(P,
												means3D.contiguous().data<float>(),
												viewmatrix.contiguous().data<float>(),
												projmatrix.contiguous().data<float>(),
												present.contiguous().data<bool>());
	}

	return present;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
IntegrateGaussiansToPointsCUDA(
	const torch::Tensor& background,
	const torch::Tensor& points3D,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& view2gaussian_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const torch::Tensor &inv_viewprojmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const nlohmann::json& settings_dict,
	torch::Tensor& alpha_integrated,
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  if (points3D.ndimension() != 2 || points3D.size(1) != 3) {
    AT_ERROR("points3D must have dimensions (num_points, 3)");
  }
	CudaRasterizer::SplattingSettings settings = settings_dict.get<CudaRasterizer::SplattingSettings>();

  const int PN = points3D.size(0);
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({10, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor out_alpha_integrated = torch::full({PN}, 1.0, float_opts);
  torch::Tensor out_color_integrated = torch::full({PN, 3}, 0.0, float_opts);
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  torch::Tensor pointBuffer = torch::empty({0}, options.device(device));
  torch::Tensor point_binningBuffer = torch::empty({0}, options.device(device));
  
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  std::function<char*(size_t)> pointFunc = resizeFunctional(pointBuffer);
  std::function<char*(size_t)> point_binningFunc = resizeFunctional(point_binningBuffer);
  
  int rendered = 0;
  if(P != 0 && PN != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }
	  CudaRasterizer::DebugVisualizationData debug_data;
	  rendered = CudaRasterizer::Rasterizer::integrate(
	    geomFunc,
		binningFunc,
		imgFunc,
		pointFunc,
		point_binningFunc,
	    PN, P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		settings,
		debug_data,
		points3D.contiguous().data<float>(),
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		view2gaussian_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		inv_viewprojmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		alpha_integrated.contiguous().data<float>(),
		out_color_integrated.contiguous().data<float>(),
		debug);
  }
  return std::make_tuple(rendered, out_color, out_color_integrated, radii, geomBuffer, binningBuffer, imgBuffer);
}

// MCMC
std::tuple<torch::Tensor, torch::Tensor> ComputeRelocationCUDA(
	torch::Tensor& opacity_old,
	torch::Tensor& scale_old,
	torch::Tensor& N,
	torch::Tensor& binoms,
	const int n_max)
{
	const int P = opacity_old.size(0);
  
	torch::Tensor final_opacity = torch::full({P}, 0, opacity_old.options().dtype(torch::kFloat32));
	torch::Tensor final_scale = torch::full({3 * P}, 0, scale_old.options().dtype(torch::kFloat32));
	if(P != 0)
	{
		UTILS::ComputeRelocation(P,
			opacity_old.contiguous().data<float>(),
			scale_old.contiguous().data<float>(),
			N.contiguous().data<int>(),
			binoms.contiguous().data<float>(),
			n_max,
			final_opacity.contiguous().data<float>(),
			final_scale.contiguous().data<float>());
	}
	return std::make_tuple(final_opacity, final_scale);
}

// Efficient Mip
std::tuple<torch::Tensor> Compute3DFilterCUDA(
	torch::Tensor& means3D,
	torch::Tensor& viewmatrices,
	const int W, const int H,
	const float focal_x, const float focal_y,
	torch::Tensor& filter_3D)
{
	const int P = means3D.size(0);
	const int C = viewmatrices.size(0);

	if(P != 0)
	{
		CudaRasterizer::Rasterizer::Compute3DFilter(
			P, C,
			means3D.contiguous().data<float>(),
			viewmatrices.contiguous().data<float>(),
			W, H, focal_x, focal_y,
			filter_3D.contiguous().data<float>()
		);
	}
	return std::make_tuple(filter_3D);
}
