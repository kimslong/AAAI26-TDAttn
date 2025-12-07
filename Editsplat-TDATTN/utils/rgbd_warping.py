import torch

def fov_to_focal(fov, image_size):
    return (image_size / 2) / torch.tan(torch.tensor(fov / 2))

def depth_to_points(depth, color, camera):
    height, width = depth.shape[-2:]
    fx = fov_to_focal(camera.FoVx, width)
    fy = fov_to_focal(camera.FoVy, height)
    cx, cy = width / 2, height / 2
    
    y, x = torch.meshgrid(
        torch.arange(height, device=depth.device), 
        torch.arange(width, device=depth.device),
        indexing='ij'
    )
    x = (x - cx) / fx
    y = (y - cy) / fy
    xyz = torch.stack((x * depth, y * depth, depth.squeeze()), dim=-1)
    
    
    # torch tensor of size (3, H, W) to (H*W, 3)
    color = color.permute(1, 2, 0).reshape(-1, 3)
    xyz = xyz.reshape(-1, 3)
    
    
    return xyz, color

def camera_to_world(points, R, T):
    return (R @ points.T + T).T

def world_to_camera(points, R, T):
    return (R.T @ (points.T - T)).T

def project_points(points, camera):
    height, width = camera.image_height, camera.image_width
    fx = fov_to_focal(camera.FoVx, width)
    fy = fov_to_focal(camera.FoVy, height)
    cx, cy = width / 2, height / 2
    
    x = points[:, 0] / points[:, 2]
    y = points[:, 1] / points[:, 2]
    px = fx * x + cx
    py = fy * y + cy
    return torch.stack((px, py, points[:, 2]), dim=-1)

def world2cam(R, t, translate=torch.tensor([.0, .0, .0])):
    Rt = torch.zeros((4, 4), device=R.device)
    Rt[:3, :3] = R.transpose(0, 1)
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    return Rt.float()

def reproject_rgbd(
    src_camera, 
    dst_camera, 
    src_img: torch.Tensor, 
    src_depth: torch.Tensor,
):
    """
    Reproject src_camera's RGBD image to dst_camera's image plane.
    
    args:
        src_camera:
        dst_camera:
        src_img: torch.Tensor (3, H, W)
            source image.
        src_depth: torch.Tensor (1, H, W)
            source depth image.
        
    returns:
        pixels_dst_valid: torch.Tensor (N, 3)
            contains dst camera pixel space coordinates of reprojected points
        colors_dst_valid: torch.Tensor (N, 3)
            contains rgb color of reprojected points.
            order is same as pixels_dst_valid
    """
    device = src_img.device
    
    src_camera_r = torch.tensor(src_camera.R, dtype=torch.float32, device=device)
    src_camera_t = torch.tensor(src_camera.T, dtype=torch.float32, device=device)
    
    dst_camera_r = torch.tensor(dst_camera.R, dtype=torch.float32, device=device)
    dst_camera_t = torch.tensor(dst_camera.T, dtype=torch.float32, device=device)
    
    src_camera_position = torch.inverse(world2cam(src_camera_r, src_camera_t))[:3, 3:4].reshape(-1, 1)
    dst_camera_position = torch.inverse(world2cam(dst_camera_r, dst_camera_t))[:3, 3:4].reshape(-1, 1)
    
    points_src_cam, colors = depth_to_points(src_depth, src_img, src_camera)
    
    points_world = camera_to_world(points_src_cam, src_camera_r, src_camera_position)
    
    points_dst_cam = world_to_camera(points_world, dst_camera_r, dst_camera_position)
    
    points_dst = project_points(points_dst_cam, dst_camera)
    
    valid_mask = (
        (points_dst_cam[:, 2] > 0) 
        & (points_dst[:, 0] >= 0) & (points_dst[:, 0] < dst_camera.image_width)
        & (points_dst[:, 1] >= 0) & (points_dst[:, 1] < dst_camera.image_height)
    )
    
    points_dst_valid = points_dst[valid_mask]
    colors_dst_valid = colors[valid_mask]
    
    return points_dst_valid, colors_dst_valid

def reprojected2img(
    warped_points_list,
    warped_colors_list,
    dst_camera,
    alpha_blend=True
):
    """
    Construct image from reprojected points and colors using vectorized operations.
    
    args:
        warped_points_list: list of torch.Tensor (N, 3)
            Contains dst camera pixel space coordinates of reprojected points.
        warped_colors_list: list of torch.Tensor (N, 3)
            Contains RGB color of reprojected points (same order as warped_points_list).
        dst_camera: dst camera object with image_height and image_width attributes.
        alpha_blend: bool
            Blend points for forward mapping (if multiple views, blend them).
            
    returns:
        dst_image: torch.Tensor (3, H, W) 
            Reconstructed image.
        depth_map: torch.Tensor (1, H, W)
            Depth map corresponding to the reprojected points.
    """
    if not isinstance(warped_points_list, list):
        warped_points_list = [warped_points_list]
        warped_colors_list = [warped_colors_list]
    
    height, width = dst_camera.image_height, dst_camera.image_width
    device = warped_points_list[0].device

    # print(warped_points_list[0].shape, warped_colors_list[0].shape)

    dst_image = torch.zeros((3, height, width), dtype=torch.float32, device=device)
    depth_map = torch.full((1, height, width), float('inf'), dtype=torch.float32, device=device)

    if not alpha_blend :
        for warped_points, warped_colors in zip(warped_points_list, warped_colors_list):
            valid_mask = (
                (warped_points[:, 2] > 0)
                & (warped_points[:, 0] >= 0) & (warped_points[:, 0] < width)
                & (warped_points[:, 1] >= 0) & (warped_points[:, 1] < height)
            )
            valid_points = warped_points[valid_mask]
            valid_colors = warped_colors[valid_mask]

            pixel_coords = valid_points[:, :2].floor().long()
            
            dst_image[:, pixel_coords[:, 1], pixel_coords[:, 0]] = valid_colors.T
            depth_map[:, pixel_coords[:, 1], pixel_coords[:, 0]] = valid_points[:, 2]

        return dst_image, depth_map

    if alpha_blend:
        N_LAYERS = len(warped_points_list)
        
        dst_image_multi_layer = torch.zeros((N_LAYERS, 3, height, width), dtype=torch.float32, device=device)
        dst_depth_multi_layer = torch.full((N_LAYERS, height, width), float('inf'), dtype=torch.float32, device=device)
        occupancy_multi_layer = torch.zeros((N_LAYERS, height, width), dtype=torch.float32, device=device)
        
        for layer_idx, (warped_points, warped_colors) in enumerate(zip(warped_points_list, warped_colors_list)):
            valid_mask = (
                (warped_points[:, 2] > 0)
                & (warped_points[:, 0] >= 0) & (warped_points[:, 0] < width)
                & (warped_points[:, 1] >= 0) & (warped_points[:, 1] < height)
            )
            valid_points = warped_points[valid_mask]
            valid_colors = warped_colors[valid_mask]
            
            pixel_coords = valid_points[:, :2].floor().long()
            
            dst_image_multi_layer[layer_idx, :, pixel_coords[:, 1], pixel_coords[:, 0]] = valid_colors.T.to(dst_image_multi_layer.dtype)
            dst_depth_multi_layer[layer_idx, pixel_coords[:, 1], pixel_coords[:, 0]] = valid_points[:, 2]
            occupancy_multi_layer[layer_idx, pixel_coords[:, 1], pixel_coords[:, 0]] = 1
            
        sorted_indices = torch.argsort(dst_depth_multi_layer, dim=0, descending=True)
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=device), torch.arange(width, device=device),
            indexing='ij'
        )
        
        dst_image = torch.zeros((3, height, width), dtype=torch.float32, device=device)
        dst_depth = torch.full((1, height, width), float('inf'), dtype=torch.float32, device=device)
        
        for layer_idx in range(N_LAYERS):
            
            
            cur_depth = dst_depth_multi_layer[sorted_indices[layer_idx], y_coords, x_coords]
            cur_image = dst_image_multi_layer[sorted_indices[layer_idx], :, y_coords, x_coords].permute(2, 0, 1) 
            
            valid_mask = (cur_depth < dst_depth).squeeze()
            
            blend_wight_map_back = (cur_depth / (cur_depth + dst_depth)).squeeze()
            blend_wight_map_front = 1 - blend_wight_map_back
            
            dst_depth = torch.where(cur_depth < dst_depth, cur_depth, dst_depth).squeeze()
            
            
            dst_image[:, valid_mask] = (
                cur_image[:, valid_mask] * blend_wight_map_front[valid_mask]
            ) + (
                dst_image[:, valid_mask] * blend_wight_map_back[valid_mask]
            )
        
        return dst_image, dst_depth
    
    
    
    
    
def get_blend_weight_list(distance_arr):
    weight_arr = 1 / distance_arr
    return weight_arr / weight_arr.sum()

def warp_from_nearby_views(
    cam_list,
    edited_img_list,
    cam_proximity_order_mat,
    cam_distance_mat,
    target_cam_idx,
    valid_src_view_idx_list,
    gaussians,
    pipe_args,
    background,
    n_warp_source=5,
) :
    """
    alpha blend with weight increase as views get closer to target view.
    args :
        cam_list : List[scene.Camera]
    edited_img_list : List[np.ndarray]
        list of edited images.
    cam_proximity_order_mat : np.array (N, N)
        cam_proximity_order_mat[i] indicates order of cameras by proximity to i-th camera.
    cam_distance_mat : np.array (N, N)
        distance matrix between cameras.
    target_cam_idx : int
        idx of target camera, where nearby views are warped into.
    valid_src_view_idx_list : List[int]
        list of indices indicating valid source camera idx for warping source.
    n_warp_source : int
        number of source views to warp.
    """
    
    device = gaussians._xyz.device
    
    warping_src_cam_list = []
    distance_list = []
    for src_view_idx in cam_proximity_order_mat[target_cam_idx][1:len(valid_src_view_idx_list)]:
        if src_view_idx in valid_src_view_idx_list:
            warping_src_cam_list.append(src_view_idx)
            distance_list.append(cam_distance_mat[target_cam_idx, src_view_idx])
        if len(warping_src_cam_list) >= n_warp_source:
            break


    per_view_blend_weight_arr = get_blend_weight_list(
        torch.tensor(distance_list, dtype=torch.float32, device=device),
    )
    
    # print(distance_list)
    # print(blend_weight_list)
            
    point_coord_list = []
    point_color_list = []

    dst_cam = cam_list[target_cam_idx]
    for src_view_idx in warping_src_cam_list:
        src_cam = cam_list[src_view_idx]
        
        render_pkg = render(src_cam, gaussians, pipe_args, background)
        
        depth_rendered = render_pkg["depth_3dgs"].detach().squeeze()
        # color = render_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0)
        
        point_coord, point_color = reproject_rgbd(
            src_cam,
            dst_cam,
            # color,
            edited_img_list[src_view_idx].to(device),
            depth_rendered
        )

        point_coord_list.append(point_coord)
        point_color_list.append(point_color)
    
    
    height, width = dst_cam.image_height, dst_cam.image_width
    N_LAYERS = len(warping_src_cam_list)

    dst_image_multi_layer = torch.zeros((N_LAYERS, 3, height, width), dtype=torch.float32, device=device)
    dst_depth_multi_layer = torch.full((N_LAYERS, height, width), np.inf, dtype=torch.float32, device=device)
    blend_wgt_multi_layer = torch.zeros((N_LAYERS, height, width), dtype=torch.float32, device=device)
    
    for layer_idx, (warped_points, warped_colors) in enumerate(zip(point_coord_list, point_color_list)):
        valid_mask = (
            (warped_points[:, 2] > 0)
            & (warped_points[:, 0] >= 0) & (warped_points[:, 0] < width)
            & (warped_points[:, 1] >= 0) & (warped_points[:, 1] < height)
        )
        valid_points = warped_points[valid_mask]
        valid_colors = warped_colors[valid_mask]

        pixel_coords = valid_points[:, :2].floor().long()
        
        dst_image_multi_layer[layer_idx, :, pixel_coords[:, 1], pixel_coords[:, 0]] = valid_colors.T
        dst_depth_multi_layer[layer_idx, pixel_coords[:, 1], pixel_coords[:, 0]] = valid_points[:, 2]
        blend_wgt_multi_layer[layer_idx, pixel_coords[:, 1], pixel_coords[:, 0]] = 1
        

    
    occupancy_mask = blend_wgt_multi_layer.sum(dim=0) > 0
    
    blend_weight_per_channel = per_view_blend_weight_arr[:, None, None].to(device)

    blend_wgt_multi_layer *= blend_weight_per_channel
    
    blend_wgt_channelwise_sum = blend_wgt_multi_layer.sum(dim=0)
    
    blend_wgt_multi_layer[
        :, occupancy_mask
    ] = blend_wgt_multi_layer[
        :, occupancy_mask
    ] / blend_wgt_channelwise_sum[
        occupancy_mask
    ]

    alpha_blended_image = (
        dst_image_multi_layer * blend_wgt_multi_layer.unsqueeze(dim=1)
    ).sum(axis=0)

    return alpha_blended_image




# if __name__ == "__main__":
    
#     # Setup
#     def edit_image(image):
#         pass
#     def find_nearby_camera(camera_list):
#         pass
    
    
#     scene = None
#     render = None
#     gaussians = None
#     pipe_args = None
#     background = None
    
#     camera_list = scene.getTrainCameras() + scene.getTestCameras()
#     cam_proximity_order_mat, (_, _, cam_distance_mat) = find_nearby_camera(camera_list, 1.0)
    
#     valid_src_view_idx_list = list(range(len(camera_list)))
#     edited_img_list = [edit_image(camera.original_image) for camera in camera_list] # should be torch tensor of shape (3, H, W)
    
#     DST_CAM_IDX = 30
#     dst_cam = camera_list[DST_CAM_IDX]
    
    
#     N_WARP_SOURCE = 5
    
#     # Usage 1
#     warping_src_cam_list = []
#     for src_view_idx in cam_proximity_order_mat[DST_CAM_IDX][1:len(valid_src_view_idx_list)]:
#         if src_view_idx in valid_src_view_idx_list:
#             warping_src_cam_list.append(src_view_idx)
#         if len(warping_src_cam_list) >= N_WARP_SOURCE:
#             break

#     point_coord_list = []
#     point_color_list = []

#     for src_view_idx in warping_src_cam_list:
#         src_cam = camera_list[src_view_idx]
        
#         render_pkg = render(src_cam, gaussians, pipe_args, background)
        
#         depth_rendered = render_pkg["depth_3dgs"].detach().squeeze()
#         color = render_pkg["render"].detach()
        
#         point_coord, point_color = reproject_rgbd(
#             src_cam,
#             dst_cam,
#             edited_img_list[src_view_idx].to(color.device),
#             depth_rendered
#         )

#         point_coord_list.append(point_coord)
#         point_color_list.append(point_color)


#     dst_img_t, dst_depth_t = reprojected2img(
#         point_coord_list,
#         point_color_list,
#         dst_cam,
#         alpha_blend=True,
#     )
        
#     # Usage 2
#     dst_img_t = warp_from_nearby_views(
#         cam_list                = camera_list,
#         edited_img_list         = edited_img_list,
#         cam_proximity_order_mat = cam_proximity_order_mat,
#         cam_distance_mat        = cam_distance_mat,
#         target_cam_idx          = DST_CAM_IDX,
#         valid_src_view_idx_list = valid_src_view_idx_list,
#         gaussians               = gaussians,
#         pipe_args               = pipe_args,
#         background              = background,
#         n_warp_source           = N_WARP_SOURCE,
#     )