import os
import torch
from random import randint
import sys
from scene import Scene, GaussianModel
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
import matplotlib.pyplot as plt
import math
import numpy as np
from scene.cameras import Camera
from gaussian_renderer import render
import IsoOctree
from scipy.spatial import KDTree
import open3d as o3d
import open3d.core as o3c
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import json


def load_camera(args):
    if os.path.exists(os.path.join(args.source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
    elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming Blender data set!")
        scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
    return cameraList_from_camInfos(scene_info.train_cameras, 1.0, args)
def writeMeshAsPly(mesh, filename):
    print("writing", filename)
    with open(filename, "wt") as f:
        # Write the PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(mesh.vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(mesh.triangles)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        # Write vertices
        for v in mesh.vertices:
            f.write("%f %f %f\n" % (v[0], v[1], v[2]))

        # Write faces
        for t in mesh.triangles:
            f.write("3 %d %d %d\n" % (t[0], t[1], t[2]))

def writeMeshAsObj(vertices, triangles, filename):
    print('writing', filename)
    count_vert = 0
    count_triangle = 0
    with open(filename, 'wt') as f:
        for v in vertices:
            count_vert += 1
            f.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for t in triangles:
            count_triangle += 1
            f.write('f %d %d %d\n' % (t[0]+1, t[1]+1, t[2]+1))
    print("count vert" , count_vert)
    print("count triangle" , count_triangle)

 

def compute_pointcloud_and_normals(depth, normal, viewpoint_cam):
    # Get image size and compute intrinsics
    W, H = viewpoint_cam.image_width, viewpoint_cam.image_height
    fx = W / (2 * math.tan(viewpoint_cam.FoVx / 2.))
    fy = H / (2 * math.tan(viewpoint_cam.FoVy / 2.))
    cx, cy = float(W) / 2, float(H) / 2
    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

    # Convert depth image (assumed to be a numpy array) to an Open3D image
    depth_o3d = o3d.geometry.Image(depth.astype(np.float32))

    # Compute extrinsic transformation
    extrinsic = viewpoint_cam.world_view_transform.T.cpu().numpy()
    R_cam_to_world = extrinsic[:3, :3]  # Extract rotation

    # Create point cloud
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d, intrinsic, extrinsic=extrinsic,
        depth_scale=1.0, depth_trunc=8.0, stride=1
    )

    points_world = np.asarray(pcd.points)  # Nx3

    depth_flat = depth.reshape(-1)

    if normal.shape[0] == 3 and normal.shape[1] == H and normal.shape[2] == W:
        normal = normal.transpose(1, 2, 0)  # -> (H, W, 3)
    normal_flat = normal.reshape(-1, 3)

    valid_mask = depth_flat > 0  # or >= some min_depth

    valid_normals_cam = normal_flat[valid_mask] 

    normal_world = (R_cam_to_world @ valid_normals_cam.T).T
    
    norms = np.linalg.norm(normal_world, axis=1, keepdims=True) + 1e-12
    normal_world /= norms

    pcd.normals = o3d.utility.Vector3dVector(normal_world)
    return points_world, normal_world

def extract_mesh_closest_surface_sample(viewpoint_cam_list, depth_list, normal_list, pixel_stride, max_depth, max_search_distance, k_nearest, subdivision_threshold, debug_ply_file=None, model_path = None):
    all_points = []
    all_normals = []
    
    voxel_downsample=0.001
    nb_neighbors=30
    std_ratio=1.0
    radius=0.02
    min_points=10

    # For each camera, compute a point cloud from the rendered depth
    for cam, depth, normal in zip(viewpoint_cam_list, depth_list, normal_list):
        pts, nrm = compute_pointcloud_and_normals(depth, normal, cam)
        all_points.append(pts)
        all_normals.append(nrm)
    points = np.vstack(all_points)
    normals = np.vstack(all_normals)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    # Stack all points and normals
    points = np.vstack(all_points)
    normals = np.vstack(all_normals)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # Remove outliers using Statistical Outlier Removal (SOR)
    pcd, inlier_indices = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    # Further filter with Radius Outlier Removal
    pcd, inlier_indices = pcd.remove_radius_outlier(nb_points=min_points, radius=radius)

    # Optional: Downsample to keep it dense but not redundant
    if voxel_downsample > 0:
        pcd = pcd.voxel_down_sample(voxel_downsample)



    # Load the teapot mesh
    # teapot_mesh = o3d.io.read_triangle_mesh("teapot_original.obj")
    # teapot_mesh.compute_vertex_normals()

    # # Extract points and normals
    # original_points = np.asarray(teapot_mesh.vertices)
    # original_normals = np.asarray(teapot_mesh.vertex_normals)

    # # Convert to Open3D point cloud
    # original_pcd = o3d.geometry.PointCloud()
    # original_pcd.points = o3d.utility.Vector3dVector(original_points)
    # original_pcd.normals = o3d.utility.Vector3dVector(original_normals)

    # # Sample more points
    # num_desired_samples = len(original_points) * 500  # Increase point density
    # sampled_pcd = teapot_mesh.sample_points_uniformly(number_of_points=num_desired_samples)

    # Alternative: Use Poisson disk sampling (comment out one method)
    # sampled_pcd = original_pcd.farthest_point_down_sample(num_desired_samples)

    # Extract new points and normals
    sampled_points = np.asarray(pcd.points)
    sampled_normals = np.asarray(pcd.normals)

    # Save the sampled point cloud for debugging
    if debug_ply_file is not None:
        o3d.io.write_point_cloud(debug_ply_file, pcd)
        print(f"Saved debug point cloud to {debug_ply_file}")

    # Build KDTree over the sampled points
    tree = KDTree(sampled_points)

    def isoValue(p0):
        _, ii = tree.query(p0, k=k_nearest, distance_upper_bound=max_search_distance)
        ii = [i for i in ii if i < len(sampled_points)]

        if len(ii) == 0:
            return 1.0

        return np.sum(
            [np.dot(sampled_normals[ii[i], :], p0 - sampled_points[ii[i], :]) for i in range(len(ii))]
        )

    def isoValues(points):
        return [isoValue(points[i, :]) for i in range(points.shape[0])]
    mesh =  IsoOctree.buildMeshWithPointCloudHint(
        isoValues,
        sampled_points,
        maxDepth=max_depth,
        subdivisionThreshold=subdivision_threshold,
    )

    out_path = os.path.join(model_path, "recon_closest_teapot_sample.obj")
    writeMeshAsObj(mesh.vertices, mesh.triangles, out_path)
    print("Closest Surface Sample mesh saved to:", out_path)
    # all_points = []
    # all_normals = []
    # # For each camera, compute a point cloud from the rendered depth
    # for cam, depth, normal in zip(viewpoint_cam_list, depth_list, normal_list):
    #     pts, nrm = compute_pointcloud_and_normals(depth, normal, cam)
    #     all_points.append(pts)
    #     all_normals.append(nrm)
    # points = np.vstack(all_points)
    # normals = np.vstack(all_normals)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.normals = o3d.utility.Vector3dVector(normals)

    # # pcd, inlier_indices = pcd.remove_statistical_outlier(
    # #     nb_neighbors=50,  # how many neighbors are considered
    # #     std_ratio=0.005     # points beyond 2 std dev from mean distance are removed
    # # )

    # #pcd, inlier_indices = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
    # points = np.asarray(pcd.points)
    # normals = np.asarray(pcd.normals)
    # if debug_ply_file is not None:
    #     o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    #     o3d.io.write_point_cloud(debug_ply_file, pcd)
    #     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
    #     densities = np.asarray(densities)
    #     density_threshold = np.quantile(densities, 0.1)
    #     vertices_to_remove = densities < density_threshold
    #     mesh.remove_vertices_by_mask(vertices_to_remove)
    #     o3d.visualization.draw_geometries([mesh])
    #     mesh_filename =  os.path.join(model_path, "debug_mesh.obj")
    #     o3d.io.write_triangle_mesh(mesh_filename, mesh)


    #     sampled_pcd = mesh.sample_points_uniformly(number_of_points=len(points))
    #     points =  np.asarray(sampled_pcd.points)
    #     # Save the final point cloud
    #     final_ply_filename = os.path.join(model_path, "filtered_pointcloud.ply")
    #     o3d.io.write_point_cloud(final_ply_filename, sampled_pcd)
    #     print(f"Saved filtered point cloud to {final_ply_filename}")
    #     print(f"Saved mesh to {mesh_filename}")
    #     print("saved ply file")
        
    
    # # Build a KDTree over the sample points.
    # tree = KDTree(points)
    # def isoValue(p0):
    #     # TODO: vectorize / make faster
    #     _, ii = tree.query(p0, k=k_nearest, distance_upper_bound=max_search_distance)
    #     ii = [i for i in ii if i < len(points)]

    #     if len(ii) == 0:
    #         return 1.0

    #     return np.sum(
    #         [np.dot(normals[ii[i], :], p0 - points[ii[i], :]) for i in range(len(ii))]
    #     )

    # def isoValues(points):
    #     return [isoValue(points[i, :]) for i in range(points.shape[0])]

    # # def isoValue(p0):
    # #     _, indices = tree.query(p0, k=k_nearest, distance_upper_bound=max_search_distance)
    # #     indices = [i for i in indices if i < len(points)]
    # #     if len(indices) == 0:
    # #         # Slight positive => 'outside' if no data
    # #         return 0.1

    # #     sdf_num = 0.0
    # #     sdf_den = 0.0
    # #     for i in indices:
    # #         diff = p0 - points[i]
    # #         dist = np.linalg.norm(diff)
    # #         if dist < 1e-8:
    # #             # p0 is basically on a sample point => isoValue ~ 0
    # #             return 0.0
    # #         # Weight closer points more
    # #         weight = 1.0 / dist
    # #         sdf_num += weight * np.dot(normals[i], diff)
    # #         sdf_den += weight

    # #     return sdf_num / sdf_den

    # # def isoValues(points):
    # #     return [isoValue(points[i, :]) for i in range(points.shape[0])]

    # mesh =  IsoOctree.buildMeshWithPointCloudHint(
    #     isoValues,
    #     points,
    #     maxDepth=max_depth,
    #     subdivisionThreshold=subdivision_threshold,
    # )

    # out_path = os.path.join(model_path, "recon_closest_surface_sample.obj")
    # writeMeshAsObj(mesh.vertices, mesh.triangles, out_path)
    # print("Closest Surface Sample mesh saved to:", out_path)

def extract_mesh_projection(viewpoint_cam_list, depth_list, normal_list, pixel_stride, max_depth, max_search_distance, subdivision_threshold=50, max_tsdf_rel=0.05, uncertain_weight=0.1, uncertain_region=0.1, cache=False, debug_ply_file=None, model_path = None):
    # Prepare a list of dictionaries containing camera info and depth maps.
    render_data = []
    for cam, depth, normal in zip(viewpoint_cam_list, depth_list, normal_list):
        W, H = cam.image_width, cam.image_height
        fx = W / (2 * math.tan(cam.FoVx / 2.))
        fy = H / (2 * math.tan(cam.FoVy / 2.))
        intrinsic = np.array([[fx, 0, float(W)/2],
                              [0, fy, float(H)/2],
                              [0, 0, 1]], dtype=np.float64)
        extrinsic = (cam.world_view_transform.T).cpu().numpy().astype(np.float64)
        # Compute camera position from extrinsic.
        cam_pos = np.linalg.inv(extrinsic)[:3,3]
        render_data.append({
            "depth_map": depth,
            "normal_map":normal,
            "resolution": (W, H),
            "intrinsics": intrinsic,
            "extrinsics": extrinsic,
            "camera_position": cam_pos
        })
    # For a hint point cloud, we compute point clouds from each depth image.
    hint_points = []
    for data, cam in zip(render_data, viewpoint_cam_list):
        pts, _ = compute_pointcloud_and_normals(data["depth_map"],data["normal_map"], cam)
        hint_points.append(pts)
    point_cloud_hint = np.vstack(hint_points)
    if debug_ply_file is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_hint)
        o3d.io.write_point_cloud(debug_ply_file, pcd)
    def get_depth_values(data, points):
        num_pts = points.shape[0]
        homog = np.hstack([points, np.ones((num_pts,1))])
        # Transform points to camera coordinates.
        cam_inv = np.linalg.inv(data["extrinsics"])
        cam_pts = (homog @ cam_inv.T)[:, :3]
        # Project points using intrinsics.
        proj = cam_pts @ data["intrinsics"].T
        proj = proj[:, :2] / (cam_pts[:, 2:3] + 1e-6)
        proj_int = np.round(proj).astype(int)
        W, H = data["resolution"]
        valid = (proj_int[:,0] >= 0) & (proj_int[:,0] < W) & (proj_int[:,1] >= 0) & (proj_int[:,1] < H) & (cam_pts[:,2] > 1e-6)
        projected_depth = np.zeros(num_pts)
        normals = np.zeros((num_pts, 3))
        depth_map = data["depth_map"]
        # For simplicity, sample depth from the rendered depth map.
        for i, v in enumerate(valid):
            if v:
                x, y = proj_int[i]
                projected_depth[i] = depth_map[y, x]
        point_depth = cam_pts[:,2]
        return projected_depth, point_depth, normals, valid
    def isoFunc(points):
        values = np.zeros(points.shape[0])
        weights = np.zeros(points.shape[0])
        valid_mask = np.zeros(points.shape[0], dtype=bool)
        valid_no_carve_mask = np.zeros(points.shape[0], dtype=bool)
        for data in render_data:
            proj_depth, point_depth, _, valid = get_depth_values(data, points)
            valid_idx = np.where(valid)[0]
            if valid_idx.size == 0:
                continue
            # Compute normalized TSDF values.
            vts = (proj_depth[valid] - point_depth[valid]) / (max_tsdf_rel * proj_depth[valid] + 1e-6)
            valid1 = vts > -1
            if not np.any(valid1):
                continue
            indices = valid_idx[valid1]
            vts = np.minimum(vts[valid1], 1)
            # Compute rays from camera center.
            rays = points[indices] - data["camera_position"][None, :]
            rays = rays / (np.linalg.norm(rays, axis=1, keepdims=True) + 1e-6)
            # For this simple example, use a constant weight.
            w = 1.0 / (max_search_distance + 1e-6)
            values[indices] += vts * w
            weights[indices] += w
            valid_mask[indices] = True
            valid_no_carve_mask[indices] = (vts < 1)
        final_mask = valid_no_carve_mask
        values[final_mask] /= (weights[final_mask] + 1e-6)
        values[~final_mask] = 1
        return values
    # Assume IsoOctree.buildMeshWithPointCloudHint is available.
    mesh = IsoOctree.buildMeshWithPointCloudHint(isoFunc, point_cloud_hint, maxDepth=max_depth, subdivisionThreshold=subdivision_threshold)
    out_path = os.path.join(model_path, "recon_projection.obj")
    writeMeshAsObj(mesh.vertices, mesh.triangles, out_path)
    print("Closest Surface Sample mesh saved to:", out_path)
    print("Projection-based mesh saved to:", out_path)

def extract_mesh_tsdf(dataset, pipe, viewpoint_cam_list, depth_list, color_list):
    torch.cuda.empty_cache()
    voxel_size = 0.005#0.0025
    o3d_device = o3d.core.Device("CPU:0")
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=('tsdf', 'weight', 'color'),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=voxel_size,
        block_resolution=16,
        block_count=50000,
        device=o3d_device
    )
    for color, depth, cam in zip(color_list, depth_list, viewpoint_cam_list):
        depth_img = o3d.t.geometry.Image(depth)
        depth_img = depth_img.to(o3d_device)
        color_img = o3d.t.geometry.Image(color)
        color_img = color_img.to(o3d_device)
        W, H = cam.image_width, cam.image_height
        fx = W / (2 * math.tan(cam.FoVx / 2.))
        fy = H / (2 * math.tan(cam.FoVy / 2.))
        intrinsic = np.array([[fx, 0, float(W)/2],
                              [0, fy, float(H)/2],
                              [0,  0,  1]], dtype=np.float64)
        intrinsic = o3d.core.Tensor(intrinsic)
        extrinsic = o3d.core.Tensor((cam.world_view_transform.T).cpu().numpy().astype(np.float64))
        frustum_block_coords = vbg.compute_unique_block_coordinates(
            depth_img, intrinsic, extrinsic, 1.0, 8.0
        )
        vbg.integrate(frustum_block_coords, depth_img, color_img, intrinsic, extrinsic, 1.0, 8.0)
    mesh = vbg.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    out_path = os.path.join(dataset.model_path, "recon_tsdf.ply")
    o3d.io.write_triangle_mesh(out_path, mesh.to_legacy())
    print("TSDF mesh saved to:", out_path)

# def extract_mesh(dataset, pipe, checkpoint_iterations=None):
#     gaussians = GaussianModel(dataset.sh_degree)
#     output_path = os.path.join(dataset.model_path,"point_cloud")
#     iteration = 0
#     if checkpoint_iterations is None:
#         for folder_name in os.listdir(output_path):
#             iteration= max(iteration,int(folder_name.split('_')[1]))
#     else:
#         iteration = checkpoint_iterations
#     output_path = os.path.join(output_path,"iteration_" + str(iteration))

#     gaussians.load_ply(os.path.join(output_path,"point_cloud.ply"), os.path.join(output_path, "params.npy"))
#     print(f'Loaded gaussians from {output_path}')
    
#     kernel_size = dataset.kernel_size
    
#     bg_color = [1, 1, 1]
#     background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
#     viewpoint_cam_list = load_camera(dataset)

#     depth_list = []
#     color_list = []
#     alpha_thres = 0.5
#     for viewpoint_cam in viewpoint_cam_list:
#         # Rendering offscreen from that camera 
#         render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size)
#         rendered_img = torch.clamp(render_pkg["render"], min=0, max=1.0).cpu().numpy().transpose(1,2,0)
#         color_list.append(np.ascontiguousarray(rendered_img))
#         depth = render_pkg["median_depth"].clone()
#         if viewpoint_cam.gt_mask is not None:
#             depth[(viewpoint_cam.gt_mask < 0.5)] = 0
#         depth[render_pkg["mask"]<alpha_thres] = 0
#         depth_list.append(depth[0].cpu().numpy())

#     torch.cuda.empty_cache()
#     voxel_size = 0.002
#     o3d_device = o3d.core.Device("CPU:0")
#     vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf', 'weight', 'color'),
#                                             attr_dtypes=(o3c.float32,
#                                                          o3c.float32,
#                                                          o3c.float32),
#                                             attr_channels=((1), (1), (3)),
#                                             voxel_size=voxel_size,
#                                             block_resolution=16,
#                                             block_count=50000,
#                                             device=o3d_device)
#     for color, depth, viewpoint_cam in zip(color_list, depth_list, viewpoint_cam_list):
#         depth = o3d.t.geometry.Image(depth)
#         depth = depth.to(o3d_device)
#         color = o3d.t.geometry.Image(color)
#         color = color.to(o3d_device)
#         W, H = viewpoint_cam.image_width, viewpoint_cam.image_height
#         fx = W / (2 * math.tan(viewpoint_cam.FoVx / 2.))
#         fy = H / (2 * math.tan(viewpoint_cam.FoVy / 2.))
#         intrinsic = np.array([[fx,0,float(W)/2],[0,fy,float(H)/2],[0,0,1]],dtype=np.float64)
#         intrinsic = o3d.core.Tensor(intrinsic)
#         extrinsic = o3d.core.Tensor((viewpoint_cam.world_view_transform.T).cpu().numpy().astype(np.float64))
#         frustum_block_coords = vbg.compute_unique_block_coordinates(
#                                                                         depth, 
#                                                                         intrinsic,
#                                                                         extrinsic, 
#                                                                         1.0, 8.0
#                                                                     )
#         vbg.integrate(
#                         frustum_block_coords, 
#                         depth, 
#                         color,
#                         intrinsic,
#                         extrinsic,  
#                         1.0, 8.0
#                     )

#     mesh = vbg.extract_triangle_mesh()
#     mesh.compute_vertex_normals()
#     o3d.io.write_triangle_mesh(os.path.join(dataset.model_path,"recon_tsdf.ply"),mesh.to_legacy())
#     print("done!")


def extract_mesh(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, method = "tsdf "):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if not skip_train:
            viewpoint_cam_list = scene.getTrainCameras()


    
        depth_list = []
        color_list = []
        normal_list = []
        alpha_thres = 0.5
        for cam in viewpoint_cam_list:
            render_pkg = render(cam, gaussians, pipeline, background, dataset.kernel_size)
            rendered_img = torch.clamp(render_pkg["render"], min=0, max=1.0).cpu().numpy().transpose(1,2,0)
            color_list.append(np.ascontiguousarray(rendered_img))
            depth = render_pkg["median_depth"].clone()
            if cam.gt_mask is not None:
                depth[(cam.gt_mask < 0.5)] = 0
            depth[render_pkg["mask"] < alpha_thres] = 0
            depth_list.append(depth[0].cpu().numpy())
            normal = render_pkg["normal"].clone()
            normal_list.append(normal.cpu().numpy().transpose(1, 2, 0))
    
        # Choose extraction method.
        if method == "tsdf":
            extract_mesh_tsdf(dataset, pipeline, viewpoint_cam_list, depth_list, color_list)
        elif method == "closest_surface_sample":
            # Here pixel_stride, max_depth, etc. are hardcoded or can be made arguments.
            extract_mesh_closest_surface_sample(viewpoint_cam_list, depth_list, normal_list,
                                                pixel_stride=1, max_depth=9, max_search_distance=0.3,
                                                k_nearest=10, subdivision_threshold=50,
                                                debug_ply_file=os.path.join(dataset.model_path, "debug_closest.ply"), model_path = dataset.model_path)
        elif method == "projection":
            extract_mesh_projection(viewpoint_cam_list, depth_list, normal_list,
                                    pixel_stride=1, max_depth=10, max_search_distance=0.3,subdivision_threshold=90,
                                    max_tsdf_rel=0.05, uncertain_weight=0.1, uncertain_region=0.1,
                                    cache=False, debug_ply_file=os.path.join(dataset.model_path, "debug_projection.ply"), model_path = dataset.model_path)
        else:
            raise ValueError(f"Unknown extraction method: {method}")
    
if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    parser.add_argument("--method", type=str, default="tsdf", choices=["tsdf", "closest_surface_sample", "projection"],
                        help="Mesh extraction method to use")
    args = parser.parse_args(sys.argv[1:])
    with torch.no_grad():
        extract_mesh(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, "tsdf")
        
        
    
    