import torch
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.assets import Articulation
from isaaclab.utils import configclass

@configclass
class E0509GripperActionTerm(ActionTerm):
    def __init__(self, cfg: "E0509GripperActionTermCfg", env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        
        # 실제 존재하는 그리퍼 조인트 이름 패턴 매칭
        gripper_joint_names = ["rh_.*"]
        self.joint_indices, _ = self.robot.find_joints(gripper_joint_names)
        
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def action_dim(self) -> int: return 1

    @property
    def raw_actions(self) -> torch.Tensor: return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor: return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        # 그리퍼 가동 범위 매핑 (0.0 ~ 1.1 라디안)
        normalized_action = (actions + 1.0) / 2.0
        self._processed_actions[:] = normalized_action * 1.1

    def apply_actions(self):
        targets = self._processed_actions.expand(-1, len(self.joint_indices))
        self.robot.set_joint_position_target(targets, joint_ids=self.joint_indices)

    def reset(self, env_ids=None):
        if env_ids is None: env_ids = slice(None)
        self._raw_actions[env_ids] = -1.0 #0.0
        self._processed_actions[env_ids] = 0.0

@configclass
class E0509GripperActionTermCfg(ActionTermCfg):
    class_type: type = E0509GripperActionTerm
    asset_name: str = "robot"
