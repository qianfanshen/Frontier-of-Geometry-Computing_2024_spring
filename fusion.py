# Copyright (c) 2018 Andy Zeng

import numpy as np
import skimage.measure

from numba import njit, prange
from skimage import measure
import trimesh
import scipy.spatial.transform

class TSDFVolume:
  """Volumetric TSDF Fusion of RGB-D Images.
  """
  def __init__(self, vol_bnds, voxel_size):
    """Constructor.

    Args:
      vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
        xyz bounds (min/max) in meters.
      voxel_size (float): The volume discretization in meters.
    """
    vol_bnds = np.asarray(vol_bnds)
    assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

    # Define voxel volume parameters
    self.vol_bnds = vol_bnds
    self.voxel_size = float(voxel_size)
    self.trunc_margin = 5 * self.voxel_size  # truncation on SDF
    self.vol_origin = self.vol_bnds[:,0].copy(order='C').astype(np.float32)

    # Define voxel grid dimensions
    self.dim = np.ceil((self.vol_bnds[:,1] - self.vol_bnds[:,0]) / self.voxel_size).copy(order='C').astype(int)
    self.dim_x, self.dim_y, self.dim_z = self.dim
    # Initialize voxel volume
    self.tsdf_vol = np.ones(self.dim).astype(np.float32)
    # for computing the cumulative moving average of weights per voxel
    self.weight_vol = np.zeros(self.dim).astype(np.float32)
    self.color_vol = np.zeros(self.dim).astype(np.float32)

    # Get voxel grid coordinates
    grid_x, grid_y, grid_z = np.meshgrid(range(self.dim_x), range(self.dim_y), range(self.dim_z), indexing='ij')
    #self.vox_coords = np.empty((self.dim_x, self.dim_y, self.dim_z, 3), dtype=np.float32 )
    self.vox_coords = np.concatenate([grid_x.reshape(1, -1), grid_y.reshape(1, -1), grid_z.reshape(1, -1)], axis=0).astype(int).T
    print("Voxel volume size: {} x {} x {} - # points: {:,}".format(
      self.dim_x, self.dim_y, self.dim_z,
      self.dim_x * self.dim_y * self.dim_z)
    )

  def integrate(self, color_im, depth_im, cam_intr, cam_pose, obs_weight=1.):
    """Integrate an RGB-D frame into the TSDF volume.

    Args:
      depth_im (ndarray): A depth image of shape (H, W).
      cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
      cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
      obs_weight (float): The weight to assign for the current observation. A higher
        value
      color_im(ndarray): An RGB image of  shape(H, W, 3)
    """
    # Fold image into single-channel
    color_im = color_im.astype(np.float32)
    color_im = np.floor(color_im[...,0] * 256*256 + color_im[...,1] * 256 + color_im[..., 2])

    # Task2: Convert voxel grid coordinates to pixel coordinates
    # voxel to world
    #self.vox_coords = self.vox_coords.astype(np.float32)
    wor_pts = np.empty_like(self.vox_coords, dtype=np.float32)
    wor_pts = self.vol_origin.reshape(1,3) + self.voxel_size * self.vox_coords
    # world to camera
    cam_pts_homo = np.concatenate((wor_pts, np.ones((wor_pts.shape[0], 1), dtype=np.float32)), axis=1).transpose((1, 0))
    cam_pts = np.dot(np.linalg.inv(cam_pose), cam_pts_homo)[:3, :]
    # camera to pixel coordinates
    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]
    pix = np.dot(cam_intr, cam_pts)
    pix_z = cam_pts[2, :]
    pix_x = np.round((cam_pts[0, :] * fx / pix_z + cx)).astype(np.uint32)
    pix_y = np.round((cam_pts[1, :] * fy / pix_z + cy)).astype(np.uint32)
    # pix = np.stack((pix_x, pix_y), axis= 1)
    # Eliminate pixels outside depth images
    valid_pix = np.logical_and(pix_x >= 0,np.logical_and(pix_x < depth_im.shape[1], np.logical_and(pix_y >= 0, pix_y < depth_im.shape[0])))
    #valid_cam_pts = cam_pts[valid_pix]
    valid_pix_x = pix_x[valid_pix].astype(np.int32)
    valid_pix_y = pix_y[valid_pix].astype(np.int32)
    # Sample depth values
    dep_val = np.zeros(pix_x.shape)
    dep_val[valid_pix] = depth_im[valid_pix_y, valid_pix_x]

    # Task3: Compute TSDF for current frame
    dep_diff = dep_val - pix_z
    valid_pts = np.logical_and(dep_val > 0, np.logical_and(dep_diff >= -self.trunc_margin, dep_diff <= self.trunc_margin))
    distance = np.maximum(-1, np.minimum(1.0, dep_diff / self.trunc_margin))

    # Task4: Integrate TSDF into voxel volume
    # self.vox_coords = self.vox_coords.astype(int)
    valid_vox_x = self.vox_coords[valid_pts, 0]
    valid_vox_y = self.vox_coords[valid_pts, 1]
    valid_vox_z = self.vox_coords[valid_pts, 2]
    weight_old = self.weight_vol[valid_vox_x, valid_vox_y, valid_vox_z]
    tsdf_vals = self.tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z]
    valid_dist = distance[valid_pts]
    #tsdf_vals_new = self.tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z]
    #weight_new = self.weight_vol[valid_vox_x, valid_vox_y, valid_vox_z]
    weight_new = weight_old + obs_weight
    tsdf_vals_new = (weight_old * tsdf_vals + obs_weight * valid_dist) / weight_new
    self.weight_vol[valid_vox_x, valid_vox_y, valid_vox_z] = weight_new
    self.tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vals_new

    # Integerate color
    color_old = self.color_vol[valid_vox_x, valid_vox_y, valid_vox_z]
    r_old = np.floor(color_old / (256 * 256))
    g_old = np.floor((color_old - r_old * 256 * 256)/256)
    b_old = color_old - r_old * 256 * 256 - g_old * 256
    color_new = np.zeros(self.vox_coords.shape[0])
    color_new[valid_pix] = color_im[valid_pix_y, valid_pix_x]
    color_new = color_new[valid_pts]
    r_new = np.floor(color_new / (256 * 256))
    g_new = np.floor((color_new - r_new * 256 * 256) / 256)
    b_new = color_new - r_new * 256 * 256 - g_new * 256
    r_new = np.minimum(255., np.round((weight_old * r_old + obs_weight * r_new) / weight_new))
    g_new = np.minimum(255., np.round((weight_old * g_old + obs_weight * g_new) / weight_new))
    b_new = np.minimum(255., np.round((weight_old * b_old + obs_weight * b_new) / weight_new))
    self.color_vol[valid_vox_x, valid_vox_y, valid_vox_z] = r_new * 256 * 256 + g_new * 256 + b_new



  def get_mesh(self):
    verts, faces, norms, vals = skimage.measure.marching_cubes(self.tsdf_vol, level=0)
    verts_indices = np.round(verts).astype(int)
    verts = verts_indices * self.voxel_size + self.vol_origin
    verts_color = self.color_vol[verts_indices[:, 0], verts_indices[:, 1], verts_indices[:, 2]]
    r_colors = np.floor(verts_color / (256 * 256))
    g_colors = np.floor((verts_color - r_colors * 256 * 256) / 256)
    b_colors = verts_color - r_colors * 256 * 256 - g_colors * 256
    colors = np.floor(np.asarray([r_colors,g_colors,b_colors])).T
    colors = colors.astype(np.uint8)
    # rotate to right direction
    gravity_dir = np.loadtxt("data/gravity-direction.txt")
    current_dir = np.array([0, -1, 1])
    rot_vec = np.cross(gravity_dir, current_dir)
    rot_angle = np.arccos(np.dot(current_dir, gravity_dir) / (np.linalg.norm(current_dir) * np.linalg.norm(gravity_dir)))
    rotation = scipy.spatial.transform.Rotation.from_rotvec(rot_vec * rot_angle)
    verts = rotation.apply(verts)
    return verts, faces, norms, colors



def cam_to_world(depth_im, cam_intr, cam_pose):
  height, width = depth_im.shape
  u, v = np.meshgrid(np.arange(width), np.arange(height))
  z = depth_im.flatten()
  x = (u.flatten() - cam_intr[0, 2]) * z / cam_intr[0, 0]
  y = (v.flatten() - cam_intr[1, 2]) * z / cam_intr[1, 1]

  points_cam = np.vstack((x, y, z, np.ones_like(z)))
  world_pts = np.dot(cam_pose, points_cam)
  world_pts = world_pts[:3, :] / world_pts[3, :]
  vertices = np.transpose(world_pts)
  pointcloud = trimesh.PointCloud(vertices)
  pointcloud.export("pointcloud.ply")
  return vertices