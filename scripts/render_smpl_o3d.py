import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

import open3d as o3d
import open3d.visualization.rendering as rendering
import imageio
from tqdm import tqdm
import joblib
import numpy as np
import torch

from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)
import random

from smpl_sim.smpllib.smpl_mujoco import SMPL_BONE_ORDER_NAMES as joint_names
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

paused, reset, recording, image_list, writer, control, curr_zoom = False, False, False, [], None, None, 0.01


def main():
    render = rendering.OffscreenRenderer(2560, 960)
    # render.scene.set_clear_color(np.array([0, 0, 0, 1]))
    ############ Load SMPL Data ############
    pkl_dir = "output/renderings/smpl_im_comp_8-2023-02-05-15:36:14.pkl"
    mujoco_joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
    Name = pkl_dir.split("/")[-1].split(".")[0]
    pkl_data = joblib.load(pkl_dir)
    data_dir = "data/smpl"
    mujoco_2_smpl = [mujoco_joint_names.index(q) for q in joint_names if q in mujoco_joint_names]
    smpl_parser_n = SMPL_Parser(model_path=data_dir, gender="neutral")
    smpl_parser_m = SMPL_Parser(model_path=data_dir, gender="male")
    smpl_parser_f = SMPL_Parser(model_path=data_dir, gender="female")

    data_seq = pkl_data['0_0']
    pose_quat, trans = data_seq['body_quat'].numpy()[::2], data_seq['trans'].numpy()[::2]
    skeleton_tree = SkeletonTree.from_dict(data_seq['skeleton_tree'])
    offset = skeleton_tree.local_translation[0]
    root_trans_offset = trans - offset.numpy()
    gender, beta = data_seq['betas'][0], data_seq['betas'][1:]

    if gender == 0:
        smpl_parser = smpl_parser_n
    elif gender == 1:
        smpl_parser = smpl_parser_m
    else:
        smpl_parser = smpl_parser_f

    sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat), torch.from_numpy(trans), is_local=True)

    global_rot = sk_state.global_rotation
    B, J, N = global_rot.shape
    pose_quat = (sRot.from_quat(global_rot.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5])).as_quat().reshape(B, -1, 4)
    B_down = pose_quat.shape[0]
    new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat), torch.from_numpy(trans), is_local=False)
    local_rot = new_sk_state.local_rotation
    pose_aa = sRot.from_quat(local_rot.reshape(-1, 4).numpy()).as_rotvec().reshape(B_down, -1, 3)
    pose_aa = pose_aa[:, mujoco_2_smpl, :].reshape(B_down, -1)
    root_trans_offset[..., :2] = root_trans_offset[..., :2] - root_trans_offset[0:1, :2]
    with torch.no_grad():
        vertices, joints = smpl_parser.get_joints_verts(pose=torch.from_numpy(pose_aa), th_trans=torch.from_numpy(root_trans_offset), th_betas=torch.from_numpy(beta[None,]))
        # vertices, joints = smpl_parser.get_joints_verts(pose=torch.from_numpy(pose_aa), th_betas=torch.from_numpy(beta[None,]))
    vertices = vertices.numpy()
    faces = smpl_parser.faces
    smpl_mesh = o3d.geometry.TriangleMesh()
    smpl_mesh.vertices = o3d.utility.Vector3dVector(vertices[0])
    smpl_mesh.triangles = o3d.utility.Vector3iVector(faces)
    # smpl_mesh.compute_triangle_normals()
    smpl_mesh.compute_vertex_normals()

    groun_plane = rendering.MaterialRecord()
    groun_plane.base_color = [1, 1, 1, 1]
    # groun_plane.shader = "defaultLit"

    box = o3d.geometry.TriangleMesh()
    ground_size = 10
    box = box.create_box(width=ground_size, height=1, depth=ground_size)
    box.compute_triangle_normals()
    # box.compute_vertex_normals()
    box.translate(np.array([-ground_size / 2, -1, -ground_size / 2]))
    box.rotate(sRot.from_euler('x', 90, degrees=True).as_matrix(), center=(0, 0, 0))
    render.scene.add_geometry("box", box, groun_plane)

    # cyl.compute_vertex_normals()
    # cyl.translate([-2, 0, 1.5])

    ending_color = rendering.MaterialRecord()
    ending_color.base_color = np.array([35, 102, 218, 256]) / 256
    ending_color.shader = "defaultLit"

    render.scene.add_geometry("cyl", smpl_mesh, ending_color)
    eye_level = 1
    render.setup_camera(60.0, [0, 0, eye_level], [0, -3, eye_level], [0, 0, 1])  # center (lookat), eye (pos), up

    # render.scene.scene.set_sun_light([0, 1, 0], [1.0, 1.0, 1.0], 100000)
    # render.scene.scene.enable_sun_light(True)
    # render.scene.scene.enable_light_shadow("sun", True)

    for i in tqdm(range(0, 50, 5)):
        smpl_mesh.vertices = o3d.utility.Vector3dVector(vertices[i])
        color_rgb = np.array([35, 102, 218, 256]) / 256 * (1 - i / 50)
        color_rgb[-1] = 1
        ending_color.base_color = color_rgb
        render.scene.add_geometry(f"cly_{i}", smpl_mesh, ending_color)
        break

    # render.scene.show_axes(True)
    img = render.render_to_image()
    cv2.imwrite("output/renderings/iccv2023/test_data.png", np.asarray(img)[..., ::-1])
    plt.figure(dpi=400)
    plt.imshow(img)
    plt.show()

    # writer = imageio.get_writer("output/renderings/test_data.mp4", fps=30, macro_block_size=None)

    # for i in tqdm(range(B_down)):

    #     smpl_mesh.vertices = o3d.utility.Vector3dVector(vertices[i])

    #     render.scene.remove_geometry('cyl')
    #     render.scene.add_geometry("cyl", smpl_mesh, color)
    #     img = render.render_to_image()
    #     writer.append_data(np.asarray(img))

    # writer.close()


if __name__ == "__main__":
    main()