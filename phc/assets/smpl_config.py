# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a simple Cartpole robot."""


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##
 

SMPL_Upright_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"phc/data/assets/usd/smpl/smpl_0_humanoid_upright.usda",
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95), # Default initial state of SMPL with the upright configuration
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["L_Hip_.", "R_Hip_.", "L_Knee_.", "R_Knee_.", "L_Ankle_.", "R_Ankle_.", "L_Toe_.", "R_Toe_."],
            effort_limit=500,
            velocity_limit=100.0,
            stiffness={
                "L_Hip_.": 800,
                "R_Hip_.": 800,
                "L_Knee_.": 800,
                "R_Knee_.": 800,
                "L_Ankle_.": 800,
                "R_Ankle_.": 800,
                "L_Toe_.": 500,
                "R_Toe_.": 500,
            },
            damping={
                "L_Hip_.": 80,
                "R_Hip_.": 80,
                "L_Knee_.": 80,
                "R_Knee_.": 80,
                "L_Ankle_.": 80,
                "R_Ankle_.": 80,
                "L_Toe_.": 50,
                "R_Toe_.": 50,
                
            },
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["Torso_.", "Spine_.", "Chest_.", "Neck_.", "Head_.", "L_Thorax_.", "R_Thorax_."],
            effort_limit=500,
            velocity_limit=100.0,
            stiffness={
                "Torso_.": 1000,
                "Spine_.": 1000,
                "Chest_.": 1000,
                "Neck_.": 500,
                "Head_.": 500,
                "L_Thorax_.": 500,
                "R_Thorax_.": 500,
            },
            damping={
                "Torso_.": 100,
                "Spine_.": 100,
                "Chest_.": 100,
                "Neck_.": 50,
                "Head_.": 50,
                "L_Thorax_.": 50,
                "R_Thorax_.": 50,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=["L_Shoulder_.", "R_Shoulder_.", "L_Elbow_.", "R_Elbow_.", "L_Wrist_.", "R_Wrist_.", "L_Hand_.", "R_Hand_."],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                "L_Shoulder_.": 500,
                "R_Shoulder_.": 500,
                "L_Elbow_.": 300,
                "R_Elbow_.": 300,
                "L_Wrist_.": 300,
                "R_Wrist_.": 300,
                "L_Hand_.": 300,
                "R_Hand_.": 300,
            },
            damping={
                "L_Shoulder_.": 50,
                "R_Shoulder_.": 50,
                "L_Elbow_.": 30,
                "R_Elbow_.": 30,
                "L_Wrist_.": 30,
                "R_Wrist_.": 30,
            },
        ),
    },
)

SMPL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"phc/data/assets/usd/smpl/smpl_0_humanoid.usda",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9), # Default initial state of SMPL with the upright configuration
        rot=(0.5, 0.5, 0.5, 0.5),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["L_Hip_.", "R_Hip_.", "L_Knee_.", "R_Knee_.", "L_Ankle_.", "R_Ankle_.", "L_Toe_.", "R_Toe_."],
            effort_limit=500,
            velocity_limit=100.0,
            stiffness={
                "L_Hip_.": 800,
                "R_Hip_.": 800,
                "L_Knee_.": 800,
                "R_Knee_.": 800,
                "L_Ankle_.": 800,
                "R_Ankle_.": 800,
                "L_Toe_.": 500,
                "R_Toe_.": 500,
            },
            damping={
                "L_Hip_.": 80,
                "R_Hip_.": 80,
                "L_Knee_.": 80,
                "R_Knee_.": 80,
                "L_Ankle_.": 80,
                "R_Ankle_.": 80,
                "L_Toe_.": 50,
                "R_Toe_.": 50,
                
            },
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["Torso_.", "Spine_.", "Chest_.", "Neck_.", "Head_.", "L_Thorax_.", "R_Thorax_."],
            effort_limit=500,
            velocity_limit=100.0,
            stiffness={
                "Torso_.": 1000,
                "Spine_.": 1000,
                "Chest_.": 1000,
                "Neck_.": 500,
                "Head_.": 500,
                "L_Thorax_.": 500,
                "R_Thorax_.": 500,
            },
            damping={
                "Torso_.": 100,
                "Spine_.": 100,
                "Chest_.": 100,
                "Neck_.": 50,
                "Head_.": 50,
                "L_Thorax_.": 50,
                "R_Thorax_.": 50,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=["L_Shoulder_.", "R_Shoulder_.", "L_Elbow_.", "R_Elbow_.", "L_Wrist_.", "R_Wrist_.", "L_Hand_.", "R_Hand_."],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                "L_Shoulder_.": 500,
                "R_Shoulder_.": 500,
                "L_Elbow_.": 300,
                "R_Elbow_.": 300,
                "L_Wrist_.": 300,
                "R_Wrist_.": 300,
                "L_Hand_.": 300,
                "R_Hand_.": 300,
            },
            damping={
                "L_Shoulder_.": 50,
                "R_Shoulder_.": 50,
                "L_Elbow_.": 30,
                "R_Elbow_.": 30,
                "L_Wrist_.": 30,
                "R_Wrist_.": 30,
            },
        ),
    },
)


SMPL_LIMIT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"phc/data/assets/usd/smpl/smpl_0_humanoid_limit.usda",
        
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9), # Default initial state of SMPL with the upright configuration
        rot=(0.5, 0.5, 0.5, 0.5),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators = {
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["L_Hip_.", "R_Hip_.", "L_Knee_.", "R_Knee_.", "L_Ankle_.", "R_Ankle_.", "L_Toe_.", "R_Toe_."],
            effort_limit={
                "L_Hip_x": 360, "L_Hip_y": 360, "L_Hip_z": 360,
                "R_Hip_x": 360, "R_Hip_y": 360, "R_Hip_z": 360,
                "L_Knee_x": 360, "L_Knee_y": 360, "L_Knee_z": 360,
                "R_Knee_x": 360, "R_Knee_y": 360, "R_Knee_z": 360,
                "L_Ankle_x": 180, "L_Ankle_y": 180, "L_Ankle_z": 180,
                "R_Ankle_x": 180, "R_Ankle_y": 180, "R_Ankle_z": 180,
                "L_Toe_x": 144, "L_Toe_y": 144, "L_Toe_z": 144,
                "R_Toe_x": 144, "R_Toe_y": 144, "R_Toe_z": 144,
            },
            velocity_limit=100.0,
            stiffness={
                "L_Hip_x": 287.33, "L_Hip_y": 219.91, "L_Hip_z": 186.37,
                "R_Hip_x": 287.33, "R_Hip_y": 219.91, "R_Hip_z": 186.37,
                "L_Knee_x": 278.11, "L_Knee_y": 18.00, "L_Knee_z": 18.00,
                "R_Knee_x": 278.11, "R_Knee_y": 18.00, "R_Knee_z": 18.00,
                "L_Ankle_x": 72.00, "L_Ankle_y": 76.49, "L_Ankle_z": 7.19,
                "R_Ankle_x": 72.00, "R_Ankle_y": 76.49, "R_Ankle_z": 7.19,
                "L_Toe_x": 71.99, "L_Toe_y": 5.76, "L_Toe_z": 5.76,
                "R_Toe_x": 71.99, "R_Toe_y": 5.76, "R_Toe_z": 5.76,
            },
            damping={
                "L_Hip_x": 15, "L_Hip_y": 10, "L_Hip_z": 10,
                "R_Hip_x": 15, "R_Hip_y": 10, "R_Hip_z": 10,
                "L_Knee_x": 8, "L_Knee_y": 8, "L_Knee_z": 8,
                "R_Knee_x": 8, "R_Knee_y": 8, "R_Knee_z": 8,
                "L_Ankle_x": 6, "L_Ankle_y": 3, "L_Ankle_z": 6,
                "R_Ankle_x": 6, "R_Ankle_y": 3, "R_Ankle_z": 6,
                "L_Toe_x": 2, "L_Toe_y": 2, "L_Toe_z": 2,  # Default to 2
                "R_Toe_x": 2, "R_Toe_y": 2, "R_Toe_z": 2,  # Default to 2
            },
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["Torso_.", "Spine_.", "Chest_.", "Neck_.", "Head_.", "L_Thorax_.", "R_Thorax_."],
            effort_limit={
                "Torso_x": 360, "Torso_y": 360, "Torso_z": 360,
                "Spine_x": 360, "Spine_y": 360, "Spine_z": 360,
                "Chest_x": 360, "Chest_y": 360, "Chest_z": 360,
                "Neck_x": 72, "Neck_y": 72, "Neck_z": 72,
                "Head_x": 48, "Head_y": 48, "Head_z": 48,
                "L_Thorax_x": 240, "L_Thorax_y": 240, "L_Thorax_z": 240,
                "R_Thorax_x": 240, "R_Thorax_y": 240, "R_Thorax_z": 240,
            },
            velocity_limit=100.0,
            stiffness={
                "Torso_x": 174.00, "Torso_y": 240.02, "Torso_z": 144.01,
                "Spine_x": 174.00, "Spine_y": 240.02, "Spine_z": 144.01,
                "Chest_x": 174.00, "Chest_y": 240.02, "Chest_z": 144.01,
                "Neck_x": 27.00, "Neck_y": 25.20, "Neck_z": 25.20,
                "Head_x": 18.00, "Head_y": 16.80, "Head_z": 16.80,
                "L_Thorax_x": 16.20, "L_Thorax_y": 48.00, "L_Thorax_z": 75.00,
                "R_Thorax_x": 16.20, "R_Thorax_y": 48.00, "R_Thorax_z": 75.00,
            },
            damping={
                "Torso_x": 15, "Torso_y": 20, "Torso_z": 20,
                "Spine_x": 15, "Spine_y": 8, "Spine_z": 12,
                "Chest_x": 15, "Chest_y": 8, "Chest_z": 12,
                "Neck_x": 10, "Neck_y": 10, "Neck_z": 10,
                "Head_x": 2, "Head_y": 2, "Head_z": 2,  # Default to 2
                "L_Thorax_x": 20, "L_Thorax_y": 20, "L_Thorax_z": 20,
                "R_Thorax_x": 20, "R_Thorax_y": 20, "R_Thorax_z": 20,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=["L_Shoulder_.", "R_Shoulder_.", "L_Elbow_.", "R_Elbow_.", "L_Wrist_.", "R_Wrist_.", "L_Hand_.", "R_Hand_."],
            effort_limit={
                "L_Shoulder_x": 96, "L_Shoulder_y": 96, "L_Shoulder_z": 96,
                "R_Shoulder_x": 96, "R_Shoulder_y": 96, "R_Shoulder_z": 96,
                "L_Elbow_x": 144, "L_Elbow_y": 144, "L_Elbow_z": 144,
                "R_Elbow_x": 144, "R_Elbow_y": 144, "R_Elbow_z": 144,
                "L_Wrist_x": 24, "L_Wrist_y": 24, "L_Wrist_z": 24,
                "R_Wrist_x": 24, "R_Wrist_y": 24, "R_Wrist_z": 24,
                "L_Hand_x": 24, "L_Hand_y": 24, "L_Hand_z": 24,
                "R_Hand_x": 24, "R_Hand_y": 24, "R_Hand_z": 24,
            },
            velocity_limit=100.0,
            stiffness={
                "L_Shoulder_x": 81.60, "L_Shoulder_y": 79.20, "L_Shoulder_z": 76.80,
                "R_Shoulder_x": 81.60, "R_Shoulder_y": 79.20, "R_Shoulder_z": 76.80,
                "L_Elbow_x": 13.68, "L_Elbow_y": 123.90, "L_Elbow_z": 66.24,
                "R_Elbow_x": 13.68, "R_Elbow_y": 123.90, "R_Elbow_z": 66.24,
                "L_Wrist_x": 21.25, "L_Wrist_y": 3.60, "L_Wrist_z": 4.80,
                "R_Wrist_x": 21.25, "R_Wrist_y": 3.60, "R_Wrist_z": 4.80,
                "L_Hand_x": 3.60, "L_Hand_y": 10.80, "L_Hand_z": 19.20,
                "R_Hand_x": 3.60, "R_Hand_y": 10.80, "R_Hand_z": 19.20,
            },
            damping={
                "L_Shoulder_x": 6, "L_Shoulder_y": 6, "L_Shoulder_z": 6,
                "R_Shoulder_x": 6, "R_Shoulder_y": 6, "R_Shoulder_z": 6,
                "L_Elbow_x": 5, "L_Elbow_y": 5, "L_Elbow_z": 5,
                "R_Elbow_x": 5, "R_Elbow_y": 5, "R_Elbow_z": 5,
                "L_Wrist_x": 2, "L_Wrist_y": 2, "L_Wrist_z": 2,  # Default to 2
                "R_Wrist_x": 2, "R_Wrist_y": 2, "R_Wrist_z": 2,  # Default to 2
                "L_Hand_x": 2, "L_Hand_y": 2, "L_Hand_z": 2,  # Default to 2
                "R_Hand_x": 2, "R_Hand_y": 2, "R_Hand_z": 2,  # Default to 2
            },
        ),
    }, 
)

SMPLX_Upright_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"phc/data/assets/usd/smpl/smplx_0_humanoid_upright.usda",
        
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["L_Hip_.", "R_Hip_.", "L_Knee_.", "R_Knee_.", "L_Ankle_.", "R_Ankle_.", "L_Toe_.", "R_Toe_."],
            effort_limit=500,
            velocity_limit=100.0,
            stiffness={
                "L_Hip_.": 800,
                "R_Hip_.": 800,
                "L_Knee_.": 800,
                "R_Knee_.": 800,
                "L_Ankle_.": 800,
                "R_Ankle_.": 800,
                "L_Toe_.": 500,
                "R_Toe_.": 500,
            },
            damping={
                "L_Hip_.": 80,
                "R_Hip_.": 80,
                "L_Knee_.": 80,
                "R_Knee_.": 80,
                "L_Ankle_.": 80,
                "R_Ankle_.": 80,
                "L_Toe_.": 50,
                "R_Toe_.": 50,
                
            },
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["Torso_.", "Spine_.", "Chest_.", "Neck_.", "Head_.", "L_Thorax_.", "R_Thorax_."],
            effort_limit=500,
            velocity_limit=100.0,
            stiffness={
                "Torso_.": 1000,
                "Spine_.": 1000,
                "Chest_.": 1000,
                "Neck_.": 500,
                "Head_.": 500,
                "L_Thorax_.": 500,
                "R_Thorax_.": 500,
            },
            damping={
                "Torso_.": 100,
                "Spine_.": 100,
                "Chest_.": 100,
                "Neck_.": 50,
                "Head_.": 50,
                "L_Thorax_.": 50,
                "R_Thorax_.": 50,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=["L_Shoulder_.", "R_Shoulder_.", "L_Elbow_.", "R_Elbow_.", "L_Wrist_.", "R_Wrist_."],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                "L_Shoulder_.": 500,
                "R_Shoulder_.": 500,
                "L_Elbow_.": 300,
                "R_Elbow_.": 300,
                "L_Wrist_.": 300,
                "R_Wrist_.": 300,
            },
            damping={
                "L_Shoulder_.": 50,
                "R_Shoulder_.": 50,
                "L_Elbow_.": 30,
                "R_Elbow_.": 30,
                "L_Wrist_.": 30,
                "R_Wrist_.": 30,
            },
        ),
        
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                "L_Index1_.", "L_Index2_.", "L_Index3_.",
                "L_Middle1_.", "L_Middle2_.", "L_Middle3_.",
                "L_Pinky1_.", "L_Pinky2_.", "L_Pinky3_.",
                "L_Ring1_.", "L_Ring2_.", "L_Ring3_.",
                "L_Thumb1_.", "L_Thumb2_.", "L_Thumb3_.",
                "R_Index1_.", "R_Index2_.", "R_Index3_.",
                "R_Middle1_.", "R_Middle2_.", "R_Middle3_.",
                "R_Pinky1_.", "R_Pinky2_.", "R_Pinky3_.",
                "R_Ring1_.", "R_Ring2_.", "R_Ring3_.",
                "R_Thumb1_.", "R_Thumb2_.", "R_Thumb3_."
            ],
            effort_limit=100,
            velocity_limit=10.0,
            stiffness={
                "L_Index1_.": 100,
                "L_Index2_.": 100,
                "L_Index3_.": 100,
                "L_Middle1_.": 100,
                "L_Middle2_.": 100,
                "L_Middle3_.": 100,
                "L_Pinky1_.": 100,
                "L_Pinky2_.": 100,
                "L_Pinky3_.": 100,
                "L_Ring1_.": 100,
                "L_Ring2_.": 100,
                "L_Ring3_.": 100,
                "L_Thumb1_.": 100,
                "L_Thumb2_.": 100,
                "L_Thumb3_.": 100,
                "R_Index1_.": 100,
                "R_Index2_.": 100,
                "R_Index3_.": 100,
                "R_Middle1_.": 100,
                "R_Middle2_.": 100,
                "R_Middle3_.": 100,
                "R_Pinky1_.": 100,
                "R_Pinky2_.": 100,
                "R_Pinky3_.": 100,
                "R_Ring1_.": 100,
                "R_Ring2_.": 100,
                "R_Ring3_.": 100,
                "R_Thumb1_.": 100,
                "R_Thumb2_.": 100,
                "R_Thumb3_.": 100,
            },
            damping={
                "L_Index1_.": 10,
                "L_Index2_.": 10,
                "L_Index3_.": 10,
                "L_Middle1_.": 10,
                "L_Middle2_.": 10,
                "L_Middle3_.": 10,
                "L_Pinky1_.": 10,
                "L_Pinky2_.": 10,
                "L_Pinky3_.": 10,
                "L_Ring1_.": 10,
                "L_Ring2_.": 10,
                "L_Ring3_.": 10,
                "L_Thumb1_.": 10,
                "L_Thumb2_.": 10,
                "L_Thumb3_.": 10,
                "R_Index1_.": 10,
                "R_Index2_.": 10,
                "R_Index3_.": 10,
                "R_Middle1_.": 10,
                "R_Middle2_.": 10,
                "R_Middle3_.": 10,
                "R_Pinky1_.": 10,
                "R_Pinky2_.": 10,
                "R_Pinky3_.": 10,
                "R_Ring1_.": 10,
                "R_Ring2_.": 10,
                "R_Ring3_.": 10,
                "R_Thumb1_.": 10,
                "R_Thumb2_.": 10,
                "R_Thumb3_.": 10,
            },
        ),
    },
)


SMPLX_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"phc/data/assets/usd/smpl/smplx_0_humanoid.usda",
        
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        rot=(0.5, 0.5, 0.5, 0.5),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["L_Hip_.", "R_Hip_.", "L_Knee_.", "R_Knee_.", "L_Ankle_.", "R_Ankle_.", "L_Toe_.", "R_Toe_."],
            effort_limit=500,
            velocity_limit=100.0,
            stiffness={
                "L_Hip_.": 800,
                "R_Hip_.": 800,
                "L_Knee_.": 800,
                "R_Knee_.": 800,
                "L_Ankle_.": 800,
                "R_Ankle_.": 800,
                "L_Toe_.": 500,
                "R_Toe_.": 500,
            },
            damping={
                "L_Hip_.": 80,
                "R_Hip_.": 80,
                "L_Knee_.": 80,
                "R_Knee_.": 80,
                "L_Ankle_.": 80,
                "R_Ankle_.": 80,
                "L_Toe_.": 50,
                "R_Toe_.": 50,
                
            },
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["Torso_.", "Spine_.", "Chest_.", "Neck_.", "Head_.", "L_Thorax_.", "R_Thorax_."],
            effort_limit=500,
            velocity_limit=100.0,
            stiffness={
                "Torso_.": 1000,
                "Spine_.": 1000,
                "Chest_.": 1000,
                "Neck_.": 500,
                "Head_.": 500,
                "L_Thorax_.": 500,
                "R_Thorax_.": 500,
            },
            damping={
                "Torso_.": 100,
                "Spine_.": 100,
                "Chest_.": 100,
                "Neck_.": 50,
                "Head_.": 50,
                "L_Thorax_.": 50,
                "R_Thorax_.": 50,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=["L_Shoulder_.", "R_Shoulder_.", "L_Elbow_.", "R_Elbow_.", "L_Wrist_.", "R_Wrist_."],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                "L_Shoulder_.": 500,
                "R_Shoulder_.": 500,
                "L_Elbow_.": 300,
                "R_Elbow_.": 300,
                "L_Wrist_.": 300,
                "R_Wrist_.": 300,
            },
            damping={
                "L_Shoulder_.": 50,
                "R_Shoulder_.": 50,
                "L_Elbow_.": 30,
                "R_Elbow_.": 30,
                "L_Wrist_.": 30,
                "R_Wrist_.": 30,
            },
        ),
        
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                "L_Index1_.", "L_Index2_.", "L_Index3_.",
                "L_Middle1_.", "L_Middle2_.", "L_Middle3_.",
                "L_Pinky1_.", "L_Pinky2_.", "L_Pinky3_.",
                "L_Ring1_.", "L_Ring2_.", "L_Ring3_.",
                "L_Thumb1_.", "L_Thumb2_.", "L_Thumb3_.",
                "R_Index1_.", "R_Index2_.", "R_Index3_.",
                "R_Middle1_.", "R_Middle2_.", "R_Middle3_.",
                "R_Pinky1_.", "R_Pinky2_.", "R_Pinky3_.",
                "R_Ring1_.", "R_Ring2_.", "R_Ring3_.",
                "R_Thumb1_.", "R_Thumb2_.", "R_Thumb3_."
            ],
            effort_limit=10,
            velocity_limit=10.0,
            stiffness={
                "L_Index1_.": 10,
                "L_Index2_.": 10,
                "L_Index3_.": 10,
                "L_Middle1_.": 10,
                "L_Middle2_.": 10,
                "L_Middle3_.": 10,
                "L_Pinky1_.": 10,
                "L_Pinky2_.": 10,
                "L_Pinky3_.": 10,
                "L_Ring1_.": 10,
                "L_Ring2_.": 10,
                "L_Ring3_.": 10,
                "L_Thumb1_.": 10,
                "L_Thumb2_.": 10,
                "L_Thumb3_.": 10,
                "R_Index1_.": 10,
                "R_Index2_.": 10,
                "R_Index3_.": 10,
                "R_Middle1_.": 10,
                "R_Middle2_.": 10,
                "R_Middle3_.": 10,
                "R_Pinky1_.": 10,
                "R_Pinky2_.": 10,
                "R_Pinky3_.": 10,
                "R_Ring1_.": 10,
                "R_Ring2_.": 10,
                "R_Ring3_.": 10,
                "R_Thumb1_.": 10,
                "R_Thumb2_.": 10,
                "R_Thumb3_.": 10,
            },
            damping={
                "L_Index1_.": 1,
                "L_Index2_.": 1,
                "L_Index3_.": 1,
                "L_Middle1_.": 1,
                "L_Middle2_.": 1,
                "L_Middle3_.": 1,
                "L_Pinky1_.": 1,
                "L_Pinky2_.": 1,
                "L_Pinky3_.": 1,
                "L_Ring1_.": 1,
                "L_Ring2_.": 1,
                "L_Ring3_.": 1,
                "L_Thumb1_.": 1,
                "L_Thumb2_.": 1,
                "L_Thumb3_.": 1,
                "R_Index1_.": 1,
                "R_Index2_.": 1,
                "R_Index3_.": 1,
                "R_Middle1_.": 1,
                "R_Middle2_.": 1,
                "R_Middle3_.": 1,
                "R_Pinky1_.": 1,
                "R_Pinky2_.": 1,
                "R_Pinky3_.": 1,
                "R_Ring1_.": 1,
                "R_Ring2_.": 1,
                "R_Ring3_.": 1,
                "R_Thumb1_.": 1,
                "R_Thumb2_.": 1,
                "R_Thumb3_.": 1,
            },
        ),
    },
)