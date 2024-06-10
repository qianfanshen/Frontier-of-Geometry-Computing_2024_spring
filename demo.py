"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time

import cv2
import numpy as np

import fusion


if __name__ == "__main__":

  print("Estimating voxel volume bounds...")
  n_imgs = 1000
  cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')
  vol_bnds = np.zeros((3,2))
  for i in range(n_imgs):
    # Read depth image and camera pose
    depth_im = cv2.imread("data/frame-%06d.depth.png"%(i),-1).astype(float)
    depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
    depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
    cam_pose = np.loadtxt("data/frame-%06d.pose.txt"%(i))  # 4x4 rigid transformation matrix

    # Task1: Convert depth image to world coordinates
    view_frust_pts = fusion.cam_to_world(depth_im, cam_intr, cam_pose)
    # Extend voxel volume bounds
    vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=0))
    vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=0))


  # Initialize TSDF voxel volume
  print("Initializing voxel volume...")
  tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.02)

  # Loop through images and fuse them together
  t0_elapse = time.time()
  for i in range(n_imgs):
    print("Fusing frame %d/%d"%(i+1, n_imgs))

    # Read rgb image
    color_im = cv2.cvtColor(cv2.imread("data/frame-%06d.color.jpg"%(i)), cv2.COLOR_BGR2RGB)
    #print("Color image shape:", color_im.shape)
    # Read depth image and camera pose
    depth_im = cv2.imread("data/frame-%06d.depth.png"%(i),-1).astype(float)
    depth_im /= 1000.
    depth_im[depth_im == 65.535] = 0
    cam_pose = np.loadtxt("data/frame-%06d.pose.txt"%(i))

    # Integrate observation into voxel volume
    tsdf_vol.integrate(color_im, depth_im, cam_intr, cam_pose, obs_weight=1.)

  fps = n_imgs / (time.time() - t0_elapse)
  print("Average FPS: {:.2f}".format(fps))

  # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
  verts, faces, normals, colors = tsdf_vol.get_mesh()
  with open("mesh_color.ply", "w") as ply_file:
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex {}\n".format(len(verts)))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face {}\n".format(len(faces)))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")
    for i in range(len(verts)):
      ply_file.write("{} {} {} {} {} {}\n".format(
        verts[i][0], verts[i][1], verts[i][2],
        colors[i][0], colors[i][1], colors[i][2]))
    for face in faces:
      ply_file.write("3 {} {} {}\n".format(face[0], face[1], face[2]))
  print("Mesh saved to", "mesh_col.ply")
