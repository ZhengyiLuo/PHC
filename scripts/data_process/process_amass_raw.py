# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())


import os
import joblib
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm
from pathlib import Path

dict_keys = ["betas", "dmpls", "gender", "mocap_framerate", "poses", "trans"]

# extract SMPL joints from SMPL-H model
joints_to_use = np.array(
    [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        37,
    ]
)
joints_to_use = np.arange(0, 156).reshape((-1, 3))[joints_to_use].reshape(-1)

all_sequences = [
    "ACCAD",
    "BMLmovi",
    "BioMotionLab_NTroje",
    "CMU",
    "DFaust_67",
    "EKUT",
    "Eyes_Japan_Dataset",
    "HumanEva",
    "KIT",
    "MPI_HDM05",
    "MPI_Limits",
    "MPI_mosh",
    "SFU",
    "SSM_synced",
    "TCD_handMocap",
    "TotalCapture",
    "Transitions_mocap",
    "BMLhandball",
    "DanceDB"
]

def read_data(folder, sequences):
    # sequences = [osp.join(folder, x) for x in sorted(os.listdir(folder)) if osp.isdir(osp.join(folder, x))]

    if sequences == "all":
        sequences = all_sequences

    db = {}
    print(folder)
    for seq_name in sequences:
        print(f"Reading {seq_name} sequence...")
        seq_folder = osp.join(folder, seq_name)

        datas = read_single_sequence(seq_folder, seq_name)
        db.update(datas)
        print(seq_name, "number of seqs", len(datas))

    return db


def read_single_sequence(folder, seq_name):
    subjects = os.listdir(folder)

    datas = {}

    for subject in tqdm(subjects):
        if not osp.isdir(osp.join(folder, subject)):
            continue
        actions = [
            x for x in os.listdir(osp.join(folder, subject)) if x.endswith(".npz") 
        ]

        for action in actions:
            fname = osp.join(folder, subject, action)

            if fname.endswith("shape.npz"):
                continue

            data = dict(np.load(fname))
            # data['poses'] = pose = data['poses'][:, joints_to_use]

            # shape = np.repeat(data['betas'][:10][np.newaxis], pose.shape[0], axis=0)
            # theta = np.concatenate([pose,shape], axis=1)
            vid_name = f"{seq_name}_{subject}_{action[:-4]}"

            datas[vid_name] = data
            # thetas.append(theta)

    return datas


def read_seq_data(folder, nsubjects, fps):
    subjects = os.listdir(folder)
    sequences = {}

    assert nsubjects < len(subjects), "nsubjects should be less than len(subjects)"

    for subject in subjects[:nsubjects]:
        actions = os.listdir(osp.join(folder, subject))

        for action in actions:
            data = np.load(osp.join(folder, subject, action))
            mocap_framerate = int(data["mocap_framerate"])
            sampling_freq = mocap_framerate // fps
            sequences[(subject, action)] = data["poses"][
                0::sampling_freq, joints_to_use
            ]

    train_set = {}
    test_set = {}

    for i, (k, v) in enumerate(sequences.items()):
        if i < len(sequences.keys()) - len(sequences.keys()) // 4:
            train_set[k] = v
        else:
            test_set[k] = v

    return train_set, test_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", type=str, help="dataset directory", default="data/amass"
    )
    parser.add_argument(
        "--out_dir", type=str, help="dataset directory", default="out"
    )
    parser.add_argument(
        '--sequences', type=str, nargs='+', help='which sequences to use', default=all_sequences
    )

    args = parser.parse_args()
    out_path = Path(args.out_dir)
    out_path.mkdir(exist_ok=True)
    db_file = osp.join(out_path, "amass_db_smplh.pt")

    db = read_data(args.dir, sequences=args.sequences)
     
    
    print(f"Saving AMASS dataset to {db_file}")
    joblib.dump(db, db_file)
