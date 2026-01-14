import torch
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import ContactSensorCfg, CameraCfg

from pathlib import Path

# 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

USD_PATH = PROJECT_ROOT / "assets" / "robot_env" / "E0509_Fin.usd"
assert USD_PATH.exists(), f"USD not found: {USD_PATH}"

USD_PATH_G = PROJECT_ROOT / "assets" / "robot_env" / "assets"/ "GroundPlane.usd"
assert USD_PATH_G.exists(), f"USD not found: {USD_PATH_G}"

# 로봇 설정
robot_cfg: ArticulationCfg = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD_PATH),
        activate_contact_sensors=True, 
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, 
            max_depenetration_velocity=1.0
        ),
        # articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        #     enabled_self_collisions=True,
        #     solver_position_iteration_count=4,
        #     solver_velocity_iteration_count=1, # 0에서 1로 수정하여 물리 안정성 확보
        # ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8, # 여기서 직접 8로 수정
            solver_velocity_iteration_count=2, # 여기서 직접 2로 수정
        ),
        
    ), # <--- [해결] SyntaxError 방지를 위한 쉼표 추가
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.7),
        joint_pos={
            "joint_1": -2.148,
            "joint_2": -0.176,
            "joint_3": -1.990,
            "joint_4": 1.273,
            "joint_5": -1.991,
            "joint_6": 2.461,
            "rh_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    # actuators={
    #     "arm_and_gripper": ImplicitActuatorCfg(
    #         joint_names_expr=[".*"],
    #         stiffness=800.0,
    #         damping=40.0,
    #     ),
    # },
    actuators={
    "arm": ImplicitActuatorCfg(
        joint_names_expr=["joint_[1-6]"],
        stiffness=800.0,
        damping=80.0,
        ),
    "gripper": ImplicitActuatorCfg(
        joint_names_expr=["rh_.*"],
        stiffness=1500.0,  # 800 -> 2000으로 대폭 강화 (꽉 쥐는 힘)
        damping=100.0,     # 떨림 방지
        ),
    },    
)

# 접촉 센서 설정
contact_sensor_LF = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/robot/e0509_with_gripper/rh_p12_rn_l2",
    update_period=0.0, 
    history_length=1,
    debug_vis=True,
    filter_prim_paths_expr=[], # 약병 감지를 위해 필터 비움
)

contact_sensor_RF = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/robot/e0509_with_gripper/rh_p12_rn_r2",
    update_period=0.0, 
    history_length=1,
    debug_vis=True,
    filter_prim_paths_expr=[],
)

# 환경 요소 설정
ground_cfg = AssetBaseCfg(
    prim_path="/World/GroundPlane", 
    spawn=sim_utils.UsdFileCfg(usd_path=str(USD_PATH_G))
)

contact_sensor_arm_body = ContactSensorCfg(
    # 정규표현식을 사용하여 joint 1~6 및 그리퍼 본체(base)만 포함
    # 손가락 끝인 l2, r2는 이름 패턴에서 제외되도록 설정
    prim_path="{ENV_REGEX_NS}/robot/e0509_with_gripper/(link_[1-6]|rh_p12_rn_base)",
    update_period=0.0,
    history_length=1,
    debug_vis=True,
)


# # 카메라 설정 (모든 환경 동적 참조)
# camera_rgb_cfg = CameraCfg(
#     prim_path="{ENV_REGEX_NS}/robot/e0509_with_gripper/rh_p12_rn_base/Camera_Mount/d435i/d435i/RSD435i/Camera_OmniVision_OV9782_Color",
#     update_period=0,
#     height=128,
#     width=128,
#     data_types=["rgb"],
#     spawn=None,
# )

# camera_depth_cfg = CameraCfg(
#     prim_path="{ENV_REGEX_NS}/robot/e0509_with_gripper/rh_p12_rn_base/Camera_Mount/d435i/d435i/RSD435i/Camera_Pseudo_Depth",
#     update_period=0,
#     height=128,
#     width=128,
#     data_types=["depth"],
#     spawn=None,
# )
