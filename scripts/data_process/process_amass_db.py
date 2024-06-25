import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import numpy as np
import glob
import pickle as pk
import joblib
import torch
import argparse

from tqdm import tqdm
from smpl_sim.utils.transform_utils import (
    convert_aa_to_orth6d,
    convert_orth_6d_to_aa,
    vertizalize_smpl_root,
    rotation_matrix_to_angle_axis,
    rot6d_to_rotmat,
)
from scipy.spatial.transform import Rotation as sRot
from smpl_sim.smpllib.smpl_parser import SMPL_Parser
from smpl_sim.utils.flags import flags

np.random.seed(1)
left_right_idx = [
    0,
    2,
    1,
    3,
    5,
    4,
    6,
    8,
    7,
    9,
    11,
    10,
    12,
    14,
    13,
    15,
    17,
    16,
    19,
    18,
    21,
    20,
    23,
    22,
]


def left_to_rigth_euler(pose_euler):
    pose_euler[:, :, 0] = pose_euler[:, :, 0] * -1
    pose_euler[:, :, 2] = pose_euler[:, :, 2] * -1
    pose_euler = pose_euler[:, left_right_idx, :]
    return pose_euler


def flip_smpl(pose, trans=None):
    """
    Pose input batch * 72
    """
    curr_spose = sRot.from_rotvec(pose.reshape(-1, 3))
    curr_spose_euler = curr_spose.as_euler("ZXY", degrees=False).reshape(pose.shape[0], 24, 3)
    curr_spose_euler = left_to_rigth_euler(curr_spose_euler)
    curr_spose_rot = sRot.from_euler("ZXY", curr_spose_euler.reshape(-1, 3), degrees=False)
    curr_spose_aa = curr_spose_rot.as_rotvec().reshape(pose.shape[0], 24, 3)
    if trans != None:
        pass
        # target_root_mat = curr_spose.as_matrix().reshape(pose.shape[0], 24, 3, 3)[:, 0]
        # root_mat = curr_spose_rot.as_matrix().reshape(pose.shape[0], 24, 3, 3)[:, 0]
        # apply_mat = np.matmul(target_root_mat[0], np.linalg.inv(root_mat[0]))

    return curr_spose_aa.reshape(-1, 72)


def sample_random_hemisphere_root():
    rot = np.random.random() * np.pi * 2
    pitch = np.random.random() * np.pi / 3 + np.pi
    r = sRot.from_rotvec([pitch, 0, 0])
    r2 = sRot.from_rotvec([0, rot, 0])
    root_vec = (r * r2).as_rotvec()
    return root_vec


def sample_seq_length(seq, tran, seq_length=150):
    if seq_length != -1:
        num_possible_seqs = seq.shape[0] // seq_length
        max_seq = seq.shape[0]

        start_idx = np.random.randint(0, 10)
        start_points = [max(0, max_seq - (seq_length + start_idx))]

        for i in range(1, num_possible_seqs - 1):
            start_points.append(i * seq_length + np.random.randint(-10, 10))

        if num_possible_seqs >= 2:
            start_points.append(max_seq - seq_length - np.random.randint(0, 10))

        seqs = [seq[i:(i + seq_length)] for i in start_points]
        trans = [tran[i:(i + seq_length)] for i in start_points]
    else:
        seqs = [seq]
        trans = [tran]
        start_points = []
    return seqs, trans, start_points


def get_random_shape(batch_size):
    shape_params = torch.rand(1, 10).repeat(batch_size, 1)
    s_id = torch.tensor(np.random.normal(scale=1.5, size=(3)))
    shape_params[:, :3] = s_id
    return shape_params



def count_consec(lst):
    consec = [1]
    for x, y in zip(lst, lst[1:]):
        if x == y - 1:
            consec[-1] += 1
        else:
            consec.append(1)
    return consec



def fix_height_smpl_vanilla(pose_aa, th_trans, th_betas, gender, seq_name):
    # no filtering, just fix height
    gender = gender.item() if isinstance(gender, np.ndarray) else gender
    if isinstance(gender, bytes):
        gender = gender.decode("utf-8")

    if gender == "neutral":
        smpl_parser = smpl_parser_n
    elif gender == "male":
        smpl_parser = smpl_parser_m
    elif gender == "female":
        smpl_parser = smpl_parser_f
    else:
        print(gender)
        raise Exception("Gender Not Supported!!")

    batch_size = pose_aa.shape[0]
    verts, jts = smpl_parser.get_joints_verts(pose_aa[0:1], th_betas.repeat((1, 1)), th_trans=th_trans[0:1])

    # vertices = verts[0].numpy()
    gp = torch.min(verts[:, :, 2])

    # if gp < 0:
    th_trans[:, 2] -= gp

    return th_trans

def process_qpos_list(qpos_list):
    amass_res = {}
    removed_k = []
    pbar = qpos_list
    for (k, v) in tqdm(pbar):
        # print("=" * 20)
        k = "0-" + k
        seq_name = k
        betas = v["betas"]
        gender = v["gender"]
        amass_fr = v["mocap_framerate"]
        skip = int(amass_fr / target_fr)
        amass_pose = v["poses"][::skip]
        amass_trans = v["trans"][::skip]

        bound = amass_pose.shape[0]
        if k in amass_occlusion:
            issue = amass_occlusion[k]["issue"]
            if (issue == "sitting" or issue == "airborne") and "idxes" in amass_occlusion[k]:
                bound = amass_occlusion[k]["idxes"][0]  # This bounded is calucaled assuming 30 FPS.....
                if bound < 10:
                    print("bound too small", k, bound)
                    continue
            else:
                print("issue irrecoverable", k, issue)
                continue

        seq_length = amass_pose.shape[0]
        if seq_length < 10:
            continue
        with torch.no_grad():
            amass_pose = amass_pose[:bound]
            batch_size = amass_pose.shape[0]
            amass_pose = np.concatenate([amass_pose[:, :66], np.zeros((batch_size, 6))], axis=1) # We use SMPL and not SMPLH
            
            pose_aa = torch.tensor(amass_pose)  # After sampling the bound
            
            amass_trans = torch.tensor(amass_trans[:bound])  # After sampling the bound
            betas = torch.from_numpy(betas)
            

            amass_trans = fix_height_smpl_vanilla(
                pose_aa=pose_aa,
                th_betas=betas,
                th_trans=amass_trans,
                gender=gender,
                seq_name=k,
            )

            pose_seq_6d = convert_aa_to_orth6d(torch.tensor(pose_aa)).reshape(batch_size, -1, 6)

            amass_res[seq_name] = {
                "pose_aa": pose_aa.numpy(),
                "pose_6d": pose_seq_6d.numpy(),
                # "qpos": qpos,
                "trans": amass_trans.numpy(),
                "beta": betas.numpy(),
                "seq_name": seq_name,
                "gender": gender,
            }

        if flags.debug and len(amass_res) > 10:
            break
    print(removed_k)
    return amass_res


amass_splits = {
    'valid': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
    'test': ['Transitions_mocap', 'SSM_synced'],
    'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BML', 'EKUT', 'TCD_handMocap', "BMLhandball", "DanceDB", "ACCAD", "BMLmovi", "BioMotionLab", "Eyes", "DFaust"]  # Adding ACCAD
}

amass_split_dict = {}
for k, v in amass_splits.items():
    for d in v:
        amass_split_dict[d] = k

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--path", type=str, default="sample_data/amass_db_smplh.pt")
    args = parser.parse_args()

    np.random.seed(0)
    flags.debug = args.debug
    take_num = "copycat_take6"
    amass_seq_data = {}
    seq_length = -1

    target_fr = 30
    video_annot = {}
    counter = 0
    seq_counter = 0
    db_dataset = args.path
    amass_db = joblib.load(db_dataset)
    amass_occlusion = joblib.load("sample_data/amass_copycat_occlusion_v3.pkl")


    qpos_list = list(amass_db.items())
    np.random.seed(0)
    np.random.shuffle(qpos_list)
    smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral", use_pca=False, create_transl=False)
    smpl_parser_m = SMPL_Parser(model_path="data/smpl", gender="male", use_pca=False, create_transl=False)
    smpl_parser_f = SMPL_Parser(model_path="data/smpl", gender="female", use_pca=False, create_transl=False)

    amass_seq_data = process_qpos_list(qpos_list)
     

    train_data = {}
    test_data = {}
    valid_data = {}
    for k, v in amass_seq_data.items():
        start_name = k.split("-")[1]
        found = False
        for dataset_key in amass_split_dict.keys():
            if start_name.lower().startswith(dataset_key.lower()):
                found = True
                split = amass_split_dict[dataset_key]
                if split == "test":
                    test_data[k] = v
                elif split == "valid":
                    valid_data[k] = v
                else:
                    train_data[k] = v
        if not found:
            print(f"Not found!! {start_name}")

    joblib.dump(train_data, f"sample_data/amass_{take_num}_train.pkl")
    joblib.dump(test_data, f"sample_data/amass_{take_num}_test.pkl")
    joblib.dump(valid_data, f"sample_data/amass_{take_num}_valid.pkl")
