import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg

from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.camera import TiledCameraCfg
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from .e0509_cfg import E0509_GRIPPER_CFG
from .mdp.actions import E0509GripperActionTermCfg

@configclass
class E0509SceneCfg(InteractiveSceneCfg):
    # num_envs를 여기서 특정 숫자로 고정하지 않습니다.
    # 대신 기본값만 명시하여 외부에서 덮어쓸 수 있게 합니다.
    num_envs: int = 1 
    env_spacing: float = 2.5
    
    # 로봇 설정 (기존 유지)
    robot = E0509_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DistantLightCfg())
    
@configclass
class E0509ActionCfg:
    """명세서 ③항: Action 설정"""
    arm = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["joint_[1-6]"], scale=0.5, use_default_offset=True
    )
    gripper = E0509GripperActionTermCfg(asset_name="robot")

@configclass
class E0509ObservationCfg:
    """명세서 ③항: Observation 설정"""
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
    policy: PolicyCfg = PolicyCfg()

@configclass
class E0509TaskCfg(ManagerBasedRLEnvCfg):
    """최종 태스크 메인 설정"""
    decimation = 2
    episode_length_s = 10.0
    
    scene: E0509SceneCfg = E0509SceneCfg() #숫자를 비우는 것이 포인트
    observations: E0509ObservationCfg = E0509ObservationCfg()
    actions: E0509ActionCfg = E0509ActionCfg()
    
    @configclass
    class RewardCfg: pass
    rewards = RewardCfg()

    @configclass
    class TerminationCfg: pass
    terminations = TerminationCfg()

    def __post_init__(self):
        self.sim.dt = 0.01 
        self.sim.render_dt = 0.01
