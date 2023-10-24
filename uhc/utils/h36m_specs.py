import numpy as np
names = ['Hips', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'Site_RToe', 'LeftUpLeg', 'LeftLeg', 'LeftFoot',
         'LeftToeBase', 'Site_LToe', 'Spine', 'Spine1', 'Neck', 'Head', 'Site_Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm',
         'LeftHand', 'LeftHandThumb', 'Site_LThumb', 'LeftWrist', 'Site_LWrist', 'RightShoulder', 'RightArm', 'RightForeArm',
         'RightHand', 'RightHandThumb', 'Site_RThumb', 'RightWrist', 'Site_RWrist']
offsets = [[   0.      ,    0.      ,    0.      ],
           [-12.7193936,    0.      ,    0.      ],
           [   0.      , -43.4291009,    0.      ],
           [   0.      , -44.8767017,    0.      ],
           [   0.      ,    0.      ,  15.1507021],
           [   0.      ,    0.      ,   7.4999997],
           [ 12.7193940,    0.      ,    0.      ],
           [   0.      , -43.4291013,    0.      ],
           [   0.      , -44.8767017,    0.      ],
           [   0.      ,    0.      ,  15.1507012],
           [   0.      ,    0.      ,   7.5000011],
           [   0.      ,    0.1     ,    0.      ],
           [   0.      ,  24.5913012,    0.      ],
           [   0.      ,  24.8462965,    0.      ],
           [   0.      ,   9.2752478,    0.      ],
           [   0.      ,  11.4999962,    0.      ],
           [   0.      ,  24.8462965,    0.      ],
           [   0.      ,  12.4881980,    0.      ],
           [   0.      ,  25.9758047,    0.      ],
           [   0.      ,  24.5542024,    0.      ],
           [   0.      ,    0.      ,    0.      ],
           [   0.      ,    0.      ,   10.000000],
           [   0.      ,  9.99999671,    0.      ],
           [   0.      ,    0.      ,    0.      ],
           [   0.      ,  24.8462965,    0.      ],
           [   0.      ,  12.4882004,    0.      ],
           [   0.      ,  25.9757994,    0.      ],
           [   0.      ,  24.5541986,    0.      ],
           [   0.      ,    0.      ,    0.      ],
           [   0.      ,    0.      ,    9.999997],
           [   0.      ,  13.7500031,    0.      ],
           [   0.      ,    0.      ,    0.      ]]
offsets = {names[i]: x for i, x in enumerate(offsets)}
parents = [-1,  0,  1,  2,  3,  4,  0,  6,  7,  8,  9,  0, 11, 12, 13, 14, 12,
           16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30]
parents = {names[i]: names[x] for i, x in enumerate(parents)}
parents[names[0]] = None
bone_addr = {
    'Hips': (0, 6),
    'RightUpLeg': (6, 9),
    'RightLeg': (10, 11),
    'RightFoot': (12, 15),
    'RightToeBase': (15, 18),
    'LeftUpLeg': (18, 21),
    'LeftLeg': (22, 23),
    'LeftFoot': (24, 27),
    'LeftToeBase': (27, 30),
    'Spine': (30, 33),
    'Spine1': (33, 36),
    'Neck': (36, 39),
    'Head': (39, 42),
    'LeftShoulder': (42, 45),
    'LeftArm': (45, 48),
    'LeftForeArm': (49, 50),
    'LeftHand': (51, 54),
    'LeftHandThumb': (54, 57),
    'LeftWrist': (57, 60),
    'RightShoulder': (60, 63),
    'RightArm': (63, 66),
    'RightForeArm': (67, 68),
    'RightHand': (69, 72),
    'RightHandThumb': (72, 75),
    'RightWrist': (75, 78)
}
joint_shuffle_ind = np.array([1, 2, 0])
exclude_bones = {'Thumb', 'Site', 'Wrist', 'Toe'}
channels = ['z', 'x', 'y']
spec_channels = {'LeftForeArm': ['x'], 'RightForeArm': ['x'],
                 'LeftLeg': ['x'], 'RightLeg': ['x']}
# make offsets symmetric
for bone in names:
    if 'Left' in bone:
        symm_bone = bone.replace('Left', 'Right')
        offset_left = offsets[bone]
        offset_right = offsets[symm_bone]
        sign_left = offset_left / (np.abs(offset_left) + 1e-12)
        sign_right = offset_right / (np.abs(offset_right) + 1e-12)
        new_offset = (np.abs(offset_left) + np.abs(offset_right)) * 0.5
        offsets[bone] = sign_left * new_offset
        offsets[symm_bone] = sign_right * new_offset
print(offsets)

