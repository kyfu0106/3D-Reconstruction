import open3d as o3d
import numpy as np
import copy
import pandas as pd
import argparse
import math
from sklearn.neighbors import NearestNeighbors

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--floor', type=int, default=1, help='1:first floor, 2: second floor')
    parser.add_argument('--icp', type=int, default=1, help='1:open3d icp, 2:my icp')
    args = parser.parse_args()
    return args

def best_fit_transform(A, B):
    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    W = AA.T @ BB
    U, S, Vt = np.linalg.svd(W)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = Vt.T @ U.T

    # translation
    t = centroid_B.T - (R @ centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T


def nearest_neighbor(src, dst):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def my_icp(A, B, init_pose=None, max_iterations=50, tolerance=1e-7):
    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = init_pose @ src

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = T @ src

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    
    # calculate final transformation
    T = best_fit_transform(A, src[:m,:].T)
   
    return T


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(source, target, voxel_size):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    
    return source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def depth_image_to_point_cloud(color_img, depth_img, f): 
    colors = np.zeros((512*512, 3))
    points = np.zeros((512*512, 3))
    u = np.array([range(512)]*512).reshape(512,512) - 256
    v = np.array([[i]*512 for i in range(512)]).reshape(512,512) - 256
    z = np.asarray(depth_img) / 1000
 
    colors = (np.asarray(color_img)/255).reshape(512*512, 3)
    points[:, 0] = (u * z / f).reshape(512*512)
    points[:, 1] = (v * z / f).reshape(512*512)
    points[:, 2] = z.reshape(512*512)
    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.select_by_index(np.where(points[:, 2] != 0)[0])
    
    pcd.transform(np.array(([[1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, -1, 0],
                            [0, 0, 0, 1]])))
                          
    #o3d.visualization.draw_geometries([pcd])
    return pcd

if __name__ == '__main__':
    args = parse_args()
    f = 1 / (2/512 * np.tan(np.pi/180*90/2))
    transformation_matrices = []
    estimate_camera_pose = []
    all_pcds_list = []
    if args.floor == 1:
        total_imgs = 210 #210
        floor = 'floor1'
        truncate_threshold = 0.000035  #-0.0013
        traj_y_offset = 0
    else:
        total_imgs = 171
        floor = 'floor2'
        truncate_threshold = 0.000035    #0.000035 
        traj_y_offset = -0.0002

    df = pd.read_csv(f'./data_task2_{floor}/camera_pose.csv')
    gt_camera_pose = np.concatenate((np.squeeze(df['x'][:].values/255*10/1.5).reshape(-1,1),
                                np.squeeze(df['y'][:].values/255*10/1.5).reshape(-1,1), 
                                np.squeeze(df['z'][:].values/255*10/1.5).reshape(-1,1)), 1)

    gt_camera_q = np.concatenate((np.squeeze(df['rw'][:].values).reshape(-1,1),
                                np.squeeze(df['rx'][:].values).reshape(-1,1),
                                np.squeeze(df['ry'][:].values).reshape(-1,1), 
                                np.squeeze(df['rz'][:].values).reshape(-1,1)), 1)

    for i in range(total_imgs):    
        print(f'Align image{i+1} to image{i}')
        src_color_img = o3d.io.read_image(f'./data_task2_{floor}/{i+1}_color.png')
        src_depth_img = o3d.io.read_image(f'./data_task2_{floor}/{i+1}_depth.png')
        source = depth_image_to_point_cloud(src_color_img, src_depth_img, f)
        
        tar_color_img = o3d.io.read_image(f'./data_task2_{floor}/{i}_color.png')
        tar_depth_img = o3d.io.read_image(f'./data_task2_{floor}/{i}_depth.png')
        target = depth_image_to_point_cloud(tar_color_img, tar_depth_img, f) 
        if i == 0:
            all_pcds_list.append(target)
            estimate_camera_pose.append([0,0,0])
            
        
        voxel_size = 0.0018
        source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source, target, voxel_size)
        result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        
        # o3d icp: p2l
        if args.icp == 1:
            distance_threshold = voxel_size * 0.4 
            result_icp = o3d.pipelines.registration.registration_icp(
                source_down, target_down, distance_threshold, result_ransac.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
            transformation_matrices.append(result_icp.transformation)
        # my icp: p2p
        else:
            result_icp = my_icp(np.asarray(source_down.points), np.asarray(target_down.points), init_pose=result_ransac.transformation)
            transformation_matrices.append(result_icp)

        # transform source at Ti to T0
        transformed_source = source
        transformed_matrix =  np.identity(4)
        for index in range(i, -1, -1):
            transformed_source = transformed_source.transform(transformation_matrices[index])
            transformed_matrix = transformation_matrices[index] @ transformed_matrix
        all_pcds_list.append(transformed_source)
        estimate_camera_pose.append([transformed_matrix[0,3], transformed_matrix[1,3], transformed_matrix[2,3]])

    # remove ceiling
    truncate_all_pcds_list = []
    for pcds in all_pcds_list:
        points = np.asarray(pcds.points)
        pcd_sel = pcds.select_by_index(np.where(points[:, 1] < truncate_threshold)[0])
        truncate_all_pcds_list.append(pcd_sel)
    
    # put traj of ground truth camera pose 
    gt_camera_pose = [a + [0, traj_y_offset, 0] for a in gt_camera_pose]
    lines = []
    colors = []
    for j in range(len(gt_camera_pose)-1):
        lines.append([j, j+1]) 
        colors.append([0, 0, 0])
    line_set_gt = o3d.geometry.LineSet()
    line_set_gt.points = o3d.utility.Vector3dVector(gt_camera_pose)
    line_set_gt.lines = o3d.utility.Vector2iVector(lines)
    line_set_gt.colors = o3d.utility.Vector3dVector(colors)
    truncate_all_pcds_list.append(line_set_gt)
    
    
    # put traj of estimate camera pose
    estimate_camera_pose = np.array(estimate_camera_pose)
    estimate_camera_pose = [a + [gt_camera_pose[0][0], gt_camera_pose[0][1], gt_camera_pose[0][2]] for a in estimate_camera_pose]    
    lines = []
    colors = []
    for j in range(len(estimate_camera_pose)-1):
        lines.append([j, j+1]) 
        colors.append([1, 0, 0])
    line_set_est = o3d.geometry.LineSet()
    line_set_est.points = o3d.utility.Vector3dVector(estimate_camera_pose)
    line_set_est.lines = o3d.utility.Vector2iVector(lines)
    line_set_est.colors = o3d.utility.Vector3dVector(colors)
    truncate_all_pcds_list.append(line_set_est)
    
    # L2 distance
    L2 = 0
    for i in range(len(gt_camera_pose)):
        L2 += math.sqrt((gt_camera_pose[i][0] - estimate_camera_pose[i][0])**2 + (gt_camera_pose[i][1] - estimate_camera_pose[i][1])**2 + (gt_camera_pose[i][2] - estimate_camera_pose[i][2])**2 )
    L2 = L2 / len(gt_camera_pose)
    print('======================')
    
    if args.icp == 1:
        print(f"L2 distance ({floor}, open3d icp): {L2}")
    else:
        print(f"L2 distance ({floor}, my icp): {L2}")
    o3d.visualization.draw_geometries(truncate_all_pcds_list)        
    