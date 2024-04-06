import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import torch 
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import joblib
from tqdm import tqdm
import argparse
import cv2
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--path", type=str, default="sample_data/amass_db_smplh.pt")
    args = parser.parse_args()
    
    process_split = "train"
    upright_start = True
    robot_cfg = {
            "mesh": False,
            "rel_joint_lm": True,
            "upright_start": upright_start,
            "remove_toe": False,
            "real_weight": True,
            "real_weight_porpotion_capsules": True,
            "real_weight_porpotion_boxes": True, 
            "replace_feet": True,
            "masterfoot": False,
            "big_ankle": True,
            "freeze_hand": False, 
            "box_body": False,
            "master_range": 50,
            "body_params": {},
            "joint_params": {},
            "geom_params": {},
            "actuator_params": {},
            "model": "smpl",
        }

    smpl_local_robot = LocalRobot(robot_cfg,)
    all_pkls = glob.glob("AMASS_data/**/*.npz", recursive=True)
    amass_occlusion = joblib.load("sample_data/amass_copycat_occlusion_v3.pkl")
    amass_full_motion_dict = {}
    amass_splits = {
        'vald': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
        'test': ['Transitions_mocap', 'SSM_synced'],
        'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'KIT',  'EKUT', 'TCD_handMocap', "BMLhandball", "DanceDB", "ACCAD", "BMLmovi", "BioMotionLab_NTroje", "Eyes_Japan_Dataset", "DFaust_67"]   # Adding ACCAD
    }
    process_set = amass_splits[process_split]
    length_acc = []
    for data_path in tqdm(all_pkls):
        bound = 0
        splits = data_path.split("/")[7:]
        key_name_dump = "0-" + "_".join(splits).replace(".npz", "")
        
        if (not splits[0] in process_set):
            continue
        
        if key_name_dump in amass_occlusion:
            issue = amass_occlusion[key_name_dump]["issue"]
            if (issue == "sitting" or issue == "airborne") and "idxes" in amass_occlusion[key_name_dump]:
                bound = amass_occlusion[key_name_dump]["idxes"][0]  # This bounded is calucaled assuming 30 FPS.....
                if bound < 10:
                    print("bound too small", key_name_dump, bound)
                    continue
            else:
                print("issue irrecoverable", key_name_dump, issue)
                continue
            
        entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))
        
        if not 'mocap_framerate' in  entry_data:
            continue
        framerate = entry_data['mocap_framerate']

        if "0-KIT_442_PizzaDelivery02_poses" == key_name_dump:
            bound = -2
        
        skip = int(framerate/30)
        root_trans = entry_data['trans'][::skip, :]
        pose_aa = np.concatenate([entry_data['poses'][::skip, :66], np.zeros((root_trans.shape[0], 6))], axis = -1)
        betas = entry_data['betas']
        gender = entry_data['gender']
        N = pose_aa.shape[0]
        
        if bound == 0:
            bound = N
            
        root_trans = root_trans[:bound]
        pose_aa = pose_aa[:bound]
        N = pose_aa.shape[0]
        if N < 10:
            continue
    
        smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
        pose_aa_mj = pose_aa.reshape(N, 24, 3)[:, smpl_2_mujoco]
        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 24, 4)

        beta = np.zeros((16))
        gender_number, beta[:], gender = [0], 0, "neutral"
        # print("using neutral model")
        smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None)
        smpl_local_robot.write_xml(f"phc/data/assets/mjcf/{robot_cfg['model']}_humanoid.xml")
        skeleton_tree = SkeletonTree.from_mjcf(f"phc/data/assets/mjcf/{robot_cfg['model']}_humanoid.xml")
        root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]

        new_sk_state = SkeletonState.from_rotation_and_root_translation(
                    skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here. 
                    torch.from_numpy(pose_quat),
                    root_trans_offset,
                    is_local=True)
        
        if robot_cfg['upright_start']:
            pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(N, -1, 4)  # should fix pose_quat as well here...

            new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False)
            pose_quat = new_sk_state.local_rotation.numpy()


        pose_quat_global = new_sk_state.global_rotation.numpy()
        pose_quat = new_sk_state.local_rotation.numpy()
        fps = 30

        new_motion_out = {}
        new_motion_out['pose_quat_global'] = pose_quat_global
        new_motion_out['pose_quat'] = pose_quat
        new_motion_out['trans_orig'] = root_trans
        new_motion_out['root_trans_offset'] = root_trans_offset
        new_motion_out['beta'] = beta
        new_motion_out['gender'] = gender
        new_motion_out['pose_aa'] = pose_aa
        new_motion_out['fps'] = fps

        amass_full_motion_dict[key_name_dump] = new_motion_out
        
    import ipdb; ipdb.set_trace()
    if upright_start:
        joblib.dump(amass_full_motion_dict, "data/amass/amass_train_take6_upright.pkl", compress=True)
    else:
        joblib.dump(amass_full_motion_dict, "data/amass/amass_train_take6.pkl", compress=True)
    # joblib.dump(amass_full_motion_dict, "data/amass/amass_test_take6.pkl", compress=True)
    # joblib.dump(amass_full_motion_dict, "data/amass_x/singles/total_capture.pkl", compress=True)
    # joblib.dump(amass_full_motion_dict, "data/amass_x/upright/singles/total_capture.pkl", compress=True)