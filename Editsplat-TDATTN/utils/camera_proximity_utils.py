import numpy as np
# from utils.rgbd_warping import world2cam

def world2cam(
    R, t, translate=np.array([.0, .0, .0]), scale=1.0
):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def find_nearby_camera(
    camera_list,
    angle_amplify_ratio = 1.0
) :
    """
    Given whole list of cameras, calculate src-dst mapping in proximity descent order.
    
    args : 
        camera_list : list of length N
        angle_amplify_ratio : float
            weight ratio of angle compared to distance.
        
    returns : 
        proximity_mat : np.ndarray (N, N)
            j'th camera is i'th camera's proximity_mat[i, j] closest camera.
        (
            pos_distance_mat : np.ndarray (N, N)
            angle_distance_mat : np.ndarray (N, N)
            distance_mat : np.ndarray (N, N)
        )

    """
    
    camera_position_world_array = np.array(list(map(
        lambda cam: np.linalg.inv(world2cam(cam.R, cam.T))[:3, 3:4].reshape(-1),
        camera_list
    )))

    camera_z_array = np.array(list(map(lambda x: x.R @ np.array([0,0,1]), camera_list)))
     
    pos_distance_mat = np.linalg.norm(
        camera_position_world_array[:, None, :] - camera_position_world_array[None, :, :],
        axis=-1
    )

    angle_distance_mat = np.arccos(
        np.clip(
            camera_z_array[:, None, :] @ camera_z_array[None, :, :].transpose(0,2,1),
            -1, 1
        )
    ).squeeze()

    distance_mat = pos_distance_mat + angle_distance_mat * angle_amplify_ratio
    
    proximity_mat = distance_mat.argsort(axis=-1)

    return proximity_mat, (pos_distance_mat, angle_distance_mat, distance_mat)



if __name__ == "__main__":
    
    # usage
    # for visualization, refer to a_notebooks/find_nearby_camera.ipynb
    
    
    scene = None

    
    train_camera_list = scene.getTrainCameras()
    proximity_mat, _ = find_nearby_camera(train_camera_list)


    # get N of closest camera for camera 0
    N = 5
    nearby_camera_idx_list = proximity_mat[0, :N].tolist()
    
    
    
    