import torch
import math
import custom_lab.envs.mdp as mdp
import custom_lab.envs.medicine_cfg as mc
import custom_lab.envs.e0509_cfg as ec

from isaaclab.utils import configclass
from isaaclab.sim.spawners.lights import DomeLightCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import (
    joint_pos_rel, joint_vel_rel, time_out, reset_joints_by_offset, last_action, action_rate_l2,
    joint_vel_l2, is_alive,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import (
    ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg,
    RewardTermCfg, TerminationTermCfg, EventTermCfg, CurriculumTermCfg
)
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

# ========================================
# Curriculum Schedules
# ========================================


ee_to_pregrasp_schedule_iter = [
    (0,   80.0),    
    (300, 100.0),  
    (500, 50.0),  
    (800, 10.0),   
]

# Object Grasping (물체 파지)
object_grasping_schedule_iter = [
    (0,   0.0), 
    (300, 20.0),   
    (400, 150.0), 
    (700, 80.0),
]

lift_object_schedule_iter = [
    (0, 0.0),
    (500, 50.0),  
    (650, 300.0),
]

# Collision Penalty (충돌 페널티)
collision_penalty_schedule_iter = [
    (0,  -0.1),    
    (100, -5.0),  
    (300, -15.0),  # 현재 적용된 값
    (600, -25.0),
]

# Action Rate Penalty (행동 변화율 페널티)
action_rate_schedule_iter = [
    (0, -0.05),
    (400, -0.1),
    (700, -0.2),   
]

# ------------------ Scene ------------------

@configclass
class E0509SceneCfg(InteractiveSceneCfg):
    env_spacing: float = 2.5
    robot = ec.robot_cfg
    bacchus = mc.bacchus_cfg
    cold = mc.cold_cfg
    kitty = mc.kitty_cfg
    ground = ec.ground_cfg
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=DomeLightCfg(intensity=400.0)
    )
    ee_contact_LF = ec.contact_sensor_LF
    ee_contact_RF = ec.contact_sensor_RF
    contact_sensor_arm_body = ec.contact_sensor_arm_body

# ------------------ Observation ------------------

@configclass
class E0509ObservationCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        concatenate_terms: bool = True
        target_pos = ObservationTermCfg(
            func=mdp.get_object_pos,
            params={"object_cfg": SceneEntityCfg("bacchus")},
        )
        ee_pos = ObservationTermCfg(func=mdp.get_ee_pos, params={"robot_cfg": None})
        ee_rot = ObservationTermCfg(func=mdp.get_ee_quat, params={"robot_cfg": None})
        ee_to_object = ObservationTermCfg(
            func=mdp.get_ee_to_object_vec,
            params={"robot_cfg": None, "object_cfg": None}
        )
        last_action = ObservationTermCfg(func=last_action)
        
        joint_vel = ObservationTermCfg(
            func=joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_[1-6]"])},
        )
        
        joint_pos = ObservationTermCfg(
            func=joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_[1-6]"])},
        )
            
    policy: PolicyCfg = PolicyCfg()

# ------------------ Action ------------------

@configclass
class E0509ActionCfg:
    ee = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["joint_[1-6]"],
        body_name="rh_p12_rn_base",
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.1),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        scale=(0.02,) * 6,
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        ),
    )
    gripper = mdp.E0509GripperActionTermCfg(asset_name="robot")

# ------------------ Reward ------------------

@configclass
class E0509RewardCfg:
    ee_to_pregrasp_pose = RewardTermCfg(
        func=mdp.ee_to_pregrasp_pose_distance,
        weight=30.0, # Curriculum에서 관리
        params={"pos_sigma": 0.2, "rot_sigma": 15.0},
    )
    object_grasping = RewardTermCfg(
        func=mdp.object_grasping_reward,
        weight=0.0, # Curriculum에서 관리
        params={"sensor_names": ["ee_contact_LF", "ee_contact_RF"],}
    )
    lift_object = RewardTermCfg(
        func=mdp.lift_object_reward,
        weight=0.0, # Curriculum에서 관리
        params={
            "sensor_names": ["ee_contact_LF", "ee_contact_RF"],
            "table_height": 0.72,
            "full_lift_height": 0.03, # [수정] 3cm 목표 반영
            "hold_steps": 15,    # [수정] 15스텝 유지
        }
    )
    collision = RewardTermCfg(
        func=mdp.get_contact_binary_penalty,
        weight=-10.0,
        params={"sensor_name": ["contact_sensor_arm_body"], "threshold": 5.0}
    )
    
    action_rate = RewardTermCfg(
        func=action_rate_l2, 
        weight=-0.001, # 아주 작은 페널티로 시작
    )
    
# ------------------ Termination ------------------

@configclass
class E0509TerminationCfg:
    time_out = TerminationTermCfg(func=time_out, time_out=True)
    
    extreme_collision = TerminationTermCfg(
        func=mdp.bad_collision_termination, # 임계값 이상의 힘이 감지되면 True 반환하는 함수
        params={"sensor_name": ["contact_sensor_arm_body"], "threshold": 400.0}
    )
    lift_success = TerminationTermCfg(
        func=mdp.lift_success_termination,
        params={"hold_steps_threshold": 15} # 1.5초(15 steps) 유지 시 성공 종료    
    )

# ------------------ Event ------------------
@configclass
class E0509EventCfg:
    # 1. 물체 위치 리셋 
    randomize_object = EventTermCfg(
        func=mdp.reset_medicine_positions,
        mode="reset",
        # 특정 asset_cfg를 주지 않아도 함수 내부 medicines 리스트에서 처리함
        params={}, 
    )
    # 2. 로봇 관절 리셋
    reset_robot = EventTermCfg(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_lift_buffer = EventTermCfg(
        func=mdp.reset_lift_hold_buffer, 
        mode="reset",
        params={}, 
    )
        
# ------------------ Curriculum ------------------      
class LiftWeightCurriculum:
    def __init__(self, term_name: str, schedule: list[tuple[int, float]], resume_iter: int = 0):
        self.term_name = term_name
        self.schedule = schedule
        self._last_iter = None
        self.resume_iter = resume_iter  # resume offset
        
    def __call__(self, env, env_ids=None, **kwargs):  
        
        # RSL-RL에서는 env.common_step_counter가 항상 증가
        steps_per_iter = env.num_envs * env.cfg.decimation
        curr_iter = env.common_step_counter // steps_per_iter

        actual_iter = curr_iter + self.resume_iter

        if self._last_iter == actual_iter:
            return
        self._last_iter = actual_iter

        return mdp.modify_reward_weight_by_iteration(
            env,
            term_name=self.term_name,
            schedule=self.schedule,
            iteration=actual_iter,
        )
        
        
@configclass
class E0509CurriculumCfg:
    # 각 Term을 변수명에 할당해야 Manager가 인식합니다.
    # ee_pose_weight = CurriculumTermCfg(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "ee_to_pregrasp_pose", "schedule": ee_to_pregrasp_schedule}
    # )
    # grasp_weight = CurriculumTermCfg(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "object_grasping", "schedule": object_grasping_schedule}
    # )
    # lift_weight = CurriculumTermCfg(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "lift_object", "schedule": lift_object_schedule}
    # )
    # collision_weight = CurriculumTermCfg(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "collision", "schedule": collision_penalty_schedule}
    # )
    # action_rate_weight = CurriculumTermCfg(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "action_rate", "schedule": action_rate_schedule}
    # )
    
    ee_pose_weight = CurriculumTermCfg(
        func=LiftWeightCurriculum(
            term_name ="ee_to_pregrasp_pose",
            schedule = ee_to_pregrasp_schedule_iter,
            resume_iter = 300
        )
    )
    grasp_weight = CurriculumTermCfg(
        func=LiftWeightCurriculum(
            term_name = "object_grasping",
            schedule = object_grasping_schedule_iter,
            resume_iter = 300
        )
    )
    lift_weight = CurriculumTermCfg(
        func=LiftWeightCurriculum(
            term_name = "lift_object",
            schedule = lift_object_schedule_iter,
            resume_iter = 300
        )
    )
    collision_weight = CurriculumTermCfg(
        func=LiftWeightCurriculum(
            term_name = "collision",
            schedule = collision_penalty_schedule_iter,
            resume_iter = 300
        )
    )
    action_rate_weight = CurriculumTermCfg(
        func=LiftWeightCurriculum(
            term_name = "action_rate",
            schedule = action_rate_schedule_iter,
            resume_iter = 300
        )
    )

# ------------------ Task ------------------

@configclass
class E0509TaskCfg(ManagerBasedRLEnvCfg):
    target_medicine_name: str = "bacchus"
    decimation: int = 10
    episode_length_s: float = 10.0
    
    scene: E0509SceneCfg = E0509SceneCfg()
    observations: E0509ObservationCfg = E0509ObservationCfg()
    actions: E0509ActionCfg = E0509ActionCfg()
    rewards: E0509RewardCfg = E0509RewardCfg()
    curriculum: E0509CurriculumCfg = E0509CurriculumCfg()
    events = E0509EventCfg() 
    terminations = E0509TerminationCfg()

    def __post_init__(self):
        super().__post_init__()

        # 1. 공통 Asset 설정
        obj_cfg = SceneEntityCfg(self.target_medicine_name)
        # 로봇 End-Effector 위치/보상 계산용
        robot_ee_cfg = SceneEntityCfg("robot", body_names=["rh_p12_rn_base"])
        
        # 2. Observations 업데이트
        policy_obs = self.observations.policy
        policy_obs.ee_to_object.params.update({"object_cfg": obj_cfg, "robot_cfg": robot_ee_cfg})
        policy_obs.ee_pos.params.update({"robot_cfg": robot_ee_cfg})
        policy_obs.ee_rot.params.update({"robot_cfg": robot_ee_cfg})
        
        # 3. Rewards 업데이트
        reward_cfg = self.rewards
        reward_cfg.ee_to_pregrasp_pose.params.update({
            "robot_cfg": robot_ee_cfg,
            "object_cfg": obj_cfg
        })
        reward_cfg.object_grasping.params.update({
            "robot_cfg": robot_ee_cfg,
            "object_cfg": obj_cfg
        })
        reward_cfg.lift_object.params.update({
            "robot_cfg": robot_ee_cfg,
            "object_cfg": obj_cfg,
        })
        
        reward_cfg.collision.params = {
            "sensor_name": ["contact_sensor_arm_body"] # 'sensor_names'에서 's' 제거, 로봇 팔 본체 센서 이름으로 변경
        }

        # 4. Simulation 설정
        self.sim.dt = 0.01
        self.sim.render_dt = 0.01