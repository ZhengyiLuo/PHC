import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())
# os.system("export REPLICATE_API_TOKEN=e47c32b4a1208437d0c5c02d85afb297353bab1b")

import replicate
import joblib

model = replicate.models.get("daanelson/motion_diffusion_model")
version = model.versions.get("3e2218c061c18b2a7388dd91b6677b6515529d4db4d719a6513a23522d23cfa7")

# https://replicate.com/daanelson/motion_diffusion_model/versions/3e2218c061c18b2a7388dd91b6677b6515529d4db4d719a6513a23522d23cfa7#input
inputs = {
    # Prompt
    'prompt': "the person walked forward and is picking up his toolbox.",

    # How many
    'num_repetitions': 3,

    # Choose the format of the output, either an animation or a json file
    # of the animation data.                The json format is: {"thetas":
    # [...], "root_translation": [...], "joint_map": [...]}, where
    # "thetas"                 is an [nframes x njoints x 3] array of
    # joint rotations in degrees, "root_translation" is an [nframes x 3]
    # array of (X, Y, Z) positions of the root, and "joint_map" is a list
    # mapping the SMPL joint index to the                corresponding
    # HumanIK joint name
    # 'output_format': "json_file",
    'output_format': "animation",
}

# https://replicate.com/daanelson/motion_diffusion_model/versions/3e2218c061c18b2a7388dd91b6677b6515529d4db4d719a6513a23522d23cfa7#output-schema
output = version.predict(**inputs)
import ipdb

ipdb.set_trace()

joblib.dump(output, "data/mdm/res.pk")