import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"phc/data/assets/usd/unitree/g1_29dof_with_hand.usda",
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
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_*",
                ".*_knee_joint",
                ".*_ankle_*",
            ],
            effort_limit={
                ".*_hip_*": 88.0, 
                ".*_knee_joint": 139.0,
                ".*_ankle_*": 50.0,
                },
            velocity_limit={
                ".*_hip_*": 32.0,
                ".*_knee_joint": 20.0,
                ".*_ankle_*": 37.0,
                },
            stiffness= 0,
            damping= 0,
            armature=0.03,
            friction=0.03,
        ),
        "body": IdealPDActuatorCfg(
            joint_names_expr=[
                "waist_*",
            ],
            effort_limit={
                ".*waist_yaw_joint*": 88.0, 
                ".*waist_roll_joint*": 88.0, 
                ".*waist_pitch_joint*": 88.0, 
                },
            velocity_limit={
                ".*waist_yaw_joint*": 32.0,
                ".*waist_roll_joint": 37.0,
                ".*waist_pitch_joint*": 37.0,
                },
            stiffness= 0,
            damping= 0,
            armature=0.03,
            friction=0.03,
        ),
        "arm": IdealPDActuatorCfg(
            joint_names_expr=[
                "*_shoulder_*", "*_elbow_*", "*_wrist_*",
            ],
            effort_limit={
                ".*_shoulder_*": 88.0,
                ".*_elbow_*": 88.0,
                ".*right_wrist_roll_joint*": 25.0, 
                ".*right_wrist_pitch_joint*": 5.0, 
                ".*right_wrist_yaw_joint*": 5.0, 
                },
            velocity_limit={
                ".*_shoulder_*": 37.0, 
                ".*_elbow_*": 37.0, 
                ".*right_wrist_roll_joint*": 37.0,
                ".*right_wrist_pitch_joint*": 22.0,
                ".*right_wrist_yaw_joint*": 22.0,
                },
            stiffness= 0,
            damping= 0,
            armature=0.03,
            friction=0.03,
        ),
        "hand": IdealPDActuatorCfg(
            joint_names_expr=[
                "*hand*", 
            ],
            effort_limit={
                ".*hand*": 50.0,
                },
            velocity_limit={
                ".*hand*": 50.0, 
                },
            stiffness= 0,
            damping= 0,
            armature=0.03,
            friction=0.03,
        ),
    },
)