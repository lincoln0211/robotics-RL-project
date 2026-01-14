import torch
from isaaclab.envs import ManagerBasedEnv
import custom_lab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv

def time_limit_exceeded(env: ManagerBasedEnv, max_episode_length: int = 500):
    return env.episode_length_buf >= max_episode_length

def bad_collision_termination(env: ManagerBasedRLEnv, sensor_name: list, threshold: float = 50.0) -> torch.Tensor:
    """매우 강한 충돌이 발생하면 에피소드를 즉시 종료합니다."""
    total_force = mdp.get_total_contact_force(env, sensor_name)
    return total_force > threshold

def lift_success_termination(
    env: ManagerBasedRLEnv, 
    hold_steps_threshold: int = 15, # lift_object_reward와 동일한 값 권장
    env_ids: torch.Tensor | None = None
) -> torch.Tensor:
    """리프트 유지 버퍼가 목표치에 도달하면 에피소드를 종료합니다."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    
    # lift_object_reward에서 사용하는 버퍼를 참조
    if hasattr(env, "lift_hold_buffer"):
        # 목표 step 이상 유지했는지 확인
        return env.lift_hold_buffer[env_ids] >= hold_steps_threshold
    return torch.zeros_like(env_ids, dtype=torch.bool)

