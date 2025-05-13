#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.general_utils import flip_align_view
from scene.NVDIFFREC import extract_env_map

def rendered_world2cam(viewpoint_cam, normal, alpha, bg_color):
    # normal: (3, H, W), alpha: (H, W), bg_color: (3)
    # normal_cam: (3, H, W)
    _, H, W = normal.shape
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf()
    normal_world = normal.permute(1,2,0).reshape(-1, 3) # (HxW, 3)
    normal_cam = torch.cat([normal_world, torch.ones_like(normal_world[...,0:1])], axis=-1) @ torch.inverse(torch.inverse(extrinsic_matrix).transpose(0,1))[...,:3]
    normal_cam = normal_cam.reshape(H, W, 3).permute(2,0,1) # (H, W, 3)
    
    background = bg_color[...,None,None]
    normal_cam = normal_cam*alpha[None,...] + background*(1. - alpha[None,...])

    return normal_cam


# render 360 lighting for a single gaussian
def render_lighting(pc : GaussianModel, resolution=(512, 1024), sampled_index=None):
    if pc.brdf_mode=="envmap":
        lighting = extract_env_map(pc.brdf_mlp, resolution) # (H, W, 3)
        lighting = lighting.permute(2,0,1) # (3, H, W)
    else:
        raise NotImplementedError

    return lighting

def normalize_normal_inplace(normal, alpha):
    # normal: (3, H, W), alpha: (H, W)
    fg_mask = (alpha[None,...]>0.).repeat(3, 1, 1)
    normal = torch.where(fg_mask, torch.nn.functional.normalize(normal, p=2, dim=0), normal)


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, kernel_size, scaling_modifier = 1.0, require_coord : bool = True, require_depth : bool = True):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Set up rasterization configurationne
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size = kernel_size,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        require_coord = require_coord,
        require_depth = require_depth,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation


    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_opacity.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True) # (N, 3)
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.brdf:
            color_delta = None
            delta_normal_norm = None
            if pc.brdf_mode=="envmap":
                gb_pos = pc.get_xyz # (N, 3) 
                view_pos = viewpoint_camera.camera_center.repeat(pc.get_opacity.shape[0], 1) # (N, 3) 

                diffuse = pc.get_diffuse.squeeze(1) # (N, 3) 
                normal, delta_normal = pc.get_normal(dir_pp_normalized=dir_pp_normalized, return_delta=True) # (N, 3) 
                delta_normal_norm = delta_normal.norm(dim=1, keepdim=True)
                specular  = pc.get_specular # (N, 3) 
                roughness = pc.get_roughness # (N, 1) 
                color, brdf_pkg = pc.brdf_mlp.shade(gb_pos[None, None, ...], normal[None, None, ...], diffuse[None, None, ...], specular[None, None, ...], roughness[None, None, ...], view_pos[None, None, ...])

                colors_precomp = color.squeeze() # (N, 3) 
                diffuse_color = brdf_pkg['diffuse'].squeeze() # (N, 3) 
                specular_color = brdf_pkg['specular'].squeeze() # (N, 3) 

                if pc.brdf_dim>0:
                    shs_view = pc.get_brdf_features.view(-1, 3, (pc.brdf_dim+1)**2)
                    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_opacity.shape[0], 1))
                    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                    sh2rgb = eval_sh(pc.brdf_dim, shs_view, dir_pp_normalized)
                    color_delta = sh2rgb
                    colors_precomp += color_delta

            else:
                raise NotImplementedError

        elif pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features

    rendered_image, radii, rendered_expected_coord, rendered_median_coord, rendered_expected_depth, rendered_median_depth, rendered_alpha, rendered_normal = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    normalize_normal_inplace(rendered_normal, rendered_alpha[0]) #????
    if require_depth:
        image = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = None,
                    colors_precomp =  delta_normal_norm.repeat(1, 3),
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = cov3D_precomp)[0]
        return {"render": rendered_image,
            "mask": rendered_alpha,
            "expected_coord": rendered_expected_coord,
            "median_coord": rendered_median_coord,
            "expected_depth": rendered_expected_depth,
            "median_depth": rendered_median_depth,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "normal":rendered_normal,
            "delta_normal_norm":image,
            }
    else: 
        return {"render": rendered_image,
            "mask": rendered_alpha,
            "expected_coord": rendered_expected_coord,
            "median_coord": rendered_median_coord,
            "expected_depth": rendered_expected_depth,
            "median_depth": rendered_median_depth,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "normal":rendered_normal,
            }
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
   
    

# integration is adopted from GOF for marching tetrahedra https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/gaussian_renderer/__init__.py
def integrate(points3D, viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, kernel_size : float, scaling_modifier = 1.0, override_color = None):
    """
    integrate Gaussians to the points, we also render the image for visual comparison. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size = kernel_size,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        require_depth = True,
        require_coord=True
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    depth_plane_precomp = None
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_opacity.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True) # (N, 3)
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.brdf:
            color_delta = None
            delta_normal_norm = None
            if pc.brdf_mode=="envmap":
                gb_pos = pc.get_xyz # (N, 3) 
                view_pos = viewpoint_camera.camera_center.repeat(pc.get_opacity.shape[0], 1) # (N, 3) 

                diffuse = pc.get_diffuse.squeeze(1) # (N, 3) 
                normal, delta_normal = pc.get_normal(dir_pp_normalized=dir_pp_normalized, return_delta=True) # (N, 3) 
                delta_normal_norm = delta_normal.norm(dim=1, keepdim=True)
                specular  = pc.get_specular # (N, 3) 
                roughness = pc.get_roughness # (N, 1) 
                color, brdf_pkg = pc.brdf_mlp.shade(gb_pos[None, None, ...], normal[None, None, ...], diffuse[None, None, ...], specular[None, None, ...], roughness[None, None, ...], view_pos[None, None, ...])

                colors_precomp = color.squeeze() # (N, 3) 
                diffuse_color = brdf_pkg['diffuse'].squeeze() # (N, 3) 
                specular_color = brdf_pkg['specular'].squeeze() # (N, 3) 

                if pc.brdf_dim>0:
                    shs_view = pc.get_brdf_features.view(-1, 3, (pc.brdf_dim+1)**2)
                    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_opacity.shape[0], 1))
                    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                    sh2rgb = eval_sh(pc.brdf_dim, shs_view, dir_pp_normalized)
                    color_delta = sh2rgb
                    colors_precomp += color_delta

            else:
                raise NotImplementedError

        elif pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, alpha_integrated, color_integrated, point_coordinate, point_sdf, radii = rasterizer.integrate(
        points3D = points3D,
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        view2gaussian_precomp=depth_plane_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "alpha_integrated": alpha_integrated,
            "color_integrated": color_integrated,
            "point_coordinate": point_coordinate,
            "point_sdf": point_sdf,
            "visibility_filter" : radii > 0,
            "radii": radii}


