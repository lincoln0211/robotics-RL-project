import torch
import numpy as np
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.envs import ManagerBasedRLEnv

SHELF_PARAMS = {
    "center_x": -0.335,
    "shelf_y_offset": 0.40,     # 이 값을 수정하여 전체적인 좌우 위치를 조절하세요.
    "slot_y_width": 0.08,
    "floor_heights": [0.87, 1.05, 1.23],
}

def get_6_slot_positions(device):
    x = SHELF_PARAMS["center_x"]
    y_base = SHELF_PARAMS["shelf_y_offset"]
    y_slot = SHELF_PARAMS["slot_y_width"]
    heights = SHELF_PARAMS["floor_heights"]
    slots = []
    for h in heights:
        slots.append([x, y_base - y_slot, h])
        slots.append([x, y_base + y_slot, h])
    return torch.tensor(slots, device=device)


def reset_medicine_positions(env, env_ids, asset_cfg: SceneEntityCfg = None):
    # 1. 위치 슬롯 정보 (get_6_slot_positions가 (6, 3) 텐서를 반환한다고 가정)
    shelf_slots = get_6_slot_positions(env.device) 
    num_resets = len(env_ids)
    medicines = ["bacchus", "cold", "kitty"]
    
    # 2. 중복 없는 랜덤 인덱스 추출 (num_resets, 3)
    # 각 환경마다 0~5 중 3개를 무작위로 뽑음
    shuffled_indices = torch.stack([torch.randperm(6, device=env.device)[:3] for _ in range(num_resets)])

    for j, med_name in enumerate(medicines):
        if med_name in env.scene.keys():
            asset = env.scene[med_name]
            origins = env.scene.env_origins[env_ids]
            
            # 해당 약병이 배치될 슬롯 좌표들 추출
            selected_slots = shelf_slots[shuffled_indices[:, j]] # (num_resets, 3)

            # 3. 회전(Quaternion) 계산 - 배치 처리를 위해 텐서로 생성
            roll = torch.zeros(num_resets, device=env.device)
            pitch = torch.zeros(num_resets, device=env.device)
            yaw = torch.zeros(num_resets, device=env.device)

            if med_name == "kitty":
                roll[:] = -torch.pi / 2.0  # 모든 환경의 kitty에 대해 roll 적용

            # quat_from_euler_xyz는 (N, 4)를 반환해야 함
            target_quat = quat_from_euler_xyz(roll, pitch, yaw)

            # 4. Root State 적용
            # default_root_state를 복사하여 초기화 (속도 등 0 포함)
            root_state = asset.data.default_root_state[env_ids].clone()
            
            # 위치 설정 (Slot 위치 + 환경별 원점)
            root_state[:, :3] = selected_slots + origins
            # 회전 설정
            root_state[:, 3:7] = target_quat
            # 속도 초기화 (정지 상태)
            root_state[:, 7:] = 0.0

            # 5. 시뮬레이션에 쓰기
            asset.write_root_state_to_sim(root_state, env_ids)
            
            # [중요] 필터링된 환경들에 대해 내부 버퍼 데이터도 리셋
            if hasattr(asset, "reset"):
                asset.reset(env_ids)
                
                
def reset_lift_hold_buffer(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """리셋 시 리프팅 유지 버퍼를 0으로 초기화합니다."""
    if hasattr(env, "lift_hold_buffer"):
        env.lift_hold_buffer[env_ids] = 0   