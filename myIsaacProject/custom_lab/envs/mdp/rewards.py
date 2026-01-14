import torch
import torch.nn.functional as F
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import quat_apply, quat_conjugate
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import ContactSensor

# ======================================================
# Pre-grasp pose reward (MAIN GUIDANCE SIGNAL)
# ======================================================
def ee_to_pregrasp_pose_distance(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    pos_sigma: float = 0.2,
    rot_sigma: float = 10.0,
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]

    ee_pos = get_ee_pos(env, robot_cfg, env_ids)
    ee_quat = get_ee_quat(env, robot_cfg, env_ids)
    obj_pos = obj.data.root_pos_w[env_ids, :3]
    obj_quat = obj.data.root_quat_w[env_ids]

    # Pregrasp target: object top
    offset_obj = torch.tensor([0.0, 0.0, 0.03], device=env.device).repeat(len(env_ids), 1)
    pregrasp_pos_w = obj_pos + quat_apply(obj_quat, offset_obj)

    # ---------------- Rotation reward ---------------
    # z축
    ee_z_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(len(env_ids), 1)
    ee_z_world = quat_apply(ee_quat, ee_z_local)
    
    rel_pos = obj_pos - ee_pos
    rel_pos[:, 2] = 0.0  # Z축 차이를 무시함 (수평 벡터로 만듦)
    target_dir = F.normalize(rel_pos, dim=-1)
    
    cos_angle_z = torch.clamp(torch.sum(ee_z_world * target_dir, dim=-1), -1.0, 1.0)
    angle_deg_z = torch.acos(cos_angle_z) * 180.0 / torch.pi
    rot_reward_z = torch.exp(-(angle_deg_z / rot_sigma) ** 2)

    # up-vector
    ee_up_local = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(len(env_ids), 1)
    ee_up_world = quat_apply(ee_quat, ee_up_local)
    world_up = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(len(env_ids), 1)
    cos_angle_up = torch.clamp(torch.sum(ee_up_world * world_up, dim=-1), -1.0, 1.0)
    angle_deg_up = torch.acos(cos_angle_up) * 180.0 / torch.pi
    rot_reward_up = torch.exp(-(angle_deg_up / 15.0) ** 2)  # 약 15도 이내 유지

    rot_reward = rot_reward_z * rot_reward_up

    # ---------------- Position reward ----------------
    pos_err = torch.norm(ee_pos - pregrasp_pos_w, dim=-1)
    pos_reward = torch.exp(-(pos_err / pos_sigma) ** 2)

    # ---------------- Strict gating ----------------
    #alignment_gate = torch.where(rot_reward > 0.5, rot_reward, rot_reward * 0.1)
    alignment_gate = torch.pow(rot_reward, 3)
    # ---------------- Disable after grasp ----------------
    if hasattr(env, "is_physically_grasped"):
        pos_reward = torch.where(env.is_physically_grasped[env_ids], 0.0, pos_reward)

    return pos_reward * alignment_gate 


# ======================================================
# Proximity Shaping + Finger Alignment + Force & Geometry Check + Stability Check
# ======================================================
def object_grasping_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    sensor_names: list[str],
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    ee_pos = get_ee_pos(env, robot_cfg, env_ids)
    ee_quat = get_ee_quat(env, robot_cfg, env_ids)
    obj_pos = obj.data.root_pos_w[env_ids, :3]

    l_idx, _ = robot.find_bodies("rh_p12_rn_l2")
    r_idx, _ = robot.find_bodies("rh_p12_rn_r2")
    l_pos = robot.data.body_pos_w[env_ids, l_idx[0]]
    r_pos = robot.data.body_pos_w[env_ids, r_idx[0]]

    finger_dist = torch.norm(l_pos - r_pos, dim=-1)
    curr_ee_dist = torch.norm(obj_pos - ee_pos, dim=-1)
    force_obj = get_contact_force_with_object(env, sensor_names, object_cfg, env_ids)

    # 물체와 EE의 거리가 13cm(0.13) 이상일 때 그리퍼를 닫으면 강한 페널티
    ee_far = curr_ee_dist > 0.13
    gripper_closed = finger_dist < 0.045  # 그리퍼가 닫힌 상태 정의
    
    # ---------------- Unified grasp condition ----------------
    is_physically_grasped = (
        (force_obj > 0.5)
        & (finger_dist < 0.05)
        & (curr_ee_dist < 0.08)
    )
    
    stay_grasped_reward = is_physically_grasped.float() * 2.0
    
    # store globally for other rewards
    if not hasattr(env, "is_physically_grasped"):
        env.is_physically_grasped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env.is_physically_grasped[env_ids] = is_physically_grasped
    
    # 미리 닫는 행위에 대한 페널티 (Pre-emptive closing penalty)
    closing_penalty = torch.where(ee_far & gripper_closed, -5.0, 0.0)

    # ---------------- One-time grasp bonus ----------------
    if not hasattr(env, "grasped_once"):
        env.grasped_once = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    new_grasp = is_physically_grasped & (~env.grasped_once[env_ids])
    env.grasped_once[env_ids] |= new_grasp

    grasp_bonus = new_grasp.float() * 10.0

    # ---------------- Distance shaping reward (10~12cm) ----------------
    # 11cm 기준, sigma=2cm
    #range_reward = torch.exp(-((curr_ee_dist - 0.11)/0.02)**2)
    range_reward = torch.exp(-((curr_ee_dist - 0.05) / 0.15)**2)
    # ---------------- Final reward ----------------
    # grasp bonus + range reward
    final_reward = grasp_bonus + range_reward * 5.0 + closing_penalty + stay_grasped_reward # scale factor 조정 가능
    
    return final_reward
    
    
# ======================================================
# Stable Grasp + Height Shaping + Force Direction Gating + Hold Buffer
# ======================================================
def lift_object_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    sensor_names: list[str],
    table_height: float = 0.72,
    min_lift_height: float = 0.01,   # 1 cm
    full_lift_height: float = 0.03,  # 3 cm
    hold_steps: int = 15,             # 15 steps 유지
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos = obj.data.root_pos_w[env_ids]
    obj_vel = obj.data.root_lin_vel_w[env_ids]

    current_height = obj_pos[:, 2] - table_height
    is_grasped = env.is_physically_grasped[env_ids]

    # ---------------- Height shaping ----------------
    # 1 cm부터 시작, 3 cm에서 1.0
    height_reward = torch.where(
        (is_grasped) & (current_height > min_lift_height),
        torch.clamp(
            (current_height - min_lift_height)
            / (full_lift_height - min_lift_height),
            0.0,
            1.0,
        ),
        torch.zeros_like(current_height),
    )

    # ---------------- Lift velocity reward ----------------
    vel_reward = is_grasped.float() * torch.clamp(obj_vel[:, 2], min=0.0) * 3.0

    # ---------------- Hold buffer ----------------
    if not hasattr(env, "lift_hold_buffer"):
        env.lift_hold_buffer = torch.zeros(
            env.num_envs, dtype=torch.int32, device=env.device
        )

    # 성공 조건용 valid lift: 3 cm 이상 + grasp 유지
    valid_lift = is_grasped & (current_height >= full_lift_height)

    env.lift_hold_buffer[env_ids] = torch.where(
        valid_lift,
        env.lift_hold_buffer[env_ids] + 1,
        torch.zeros_like(env.lift_hold_buffer[env_ids]),
    )

    hold_progress = torch.clamp(
        env.lift_hold_buffer[env_ids].float() / hold_steps, 0.0, 1.0
    )

    # ---------------- Lift success ----------------
    lift_success = (env.lift_hold_buffer[env_ids] >= hold_steps).float() * 50.0

    return (
        5.0 * height_reward
        + 2.0 * vel_reward
        + 10.0 * hold_progress
        + lift_success
    )

# ------------------ robot base / Object ------------------
def get_robot_root_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, env_ids: torch.Tensor = None) -> torch.Tensor:
    """로봇의 루트(Root/Base) 월드 좌표를 반환합니다."""
    # SceneEntityCfg에 정의된 이름("robot")으로 자산을 가져옴
    robot: Articulation = env.scene[asset_cfg.name]
    
    if env_ids is None:
        num_envs = robot.num_instances if hasattr(robot, "num_instances") else robot.num_envs
        env_ids = torch.arange(num_envs, device=robot.device)
        
    # 로봇의 루트 위치 (x, y, z) 반환
    return robot.data.root_pos_w[env_ids, :3]

# ------------------ Target / Object ------------------
def get_object_pos(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, env_ids: torch.Tensor = None) -> torch.Tensor:
    """대상 물체(Target)의 월드 좌표계 기준 위치를 반환합니다."""
    obj: RigidObject = env.scene[object_cfg.name]
    
    if env_ids is None:
        # RigidObject의 경우 root_pos_w의 첫 번째 차원을 통해 전체 환경 수를 파악
        num_envs = obj.data.root_pos_w.shape[0]
        env_ids = torch.arange(num_envs, device=obj.device)
        
    # 물체의 root 위치 (x, y, z) 반환
    return obj.data.root_pos_w[env_ids, :3]

# ------------------ EE / Robot ------------------
def get_ee_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, env_ids: torch.Tensor = None) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    
    # env_ids가 없을 경우, 전체 환경 개수를 env 객체에서 직접 가져옵니다.
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    # robot_cfg.body_ids는 resolve()를 통해 정수로 변환된 body 인덱스 리스트입니다.
    body_indices = robot_cfg.body_ids
    
    # body_pos_w shape: (num_envs, num_bodies_in_asset, 3)
    # env_ids와 body_indices를 이용해 필요한 위치만 추출합니다.
    # [env_ids[:, None], body_indices] -> (num_envs, num_selected_bodies, 3)
    body_poses = robot.data.body_pos_w[env_ids[:, None], body_indices, :]
    
    # 선택된 body들(예: 손가락 r2, l2)의 평균 위치 계산
    mid_pos_world = torch.mean(body_poses, dim=1)
    
    return mid_pos_world

def get_ee_quat(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, env_ids: torch.Tensor = None) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    
    # 기준이 되는 첫 번째 body의 쿼터니언을 가져옵니다.
    base_id = robot_cfg.body_ids[0]
    quat = robot.data.body_quat_w[env_ids, base_id, :]
    
    return quat

def get_ee_to_object_vec(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg, env_ids: torch.Tensor = None):
    obj: RigidObject = env.scene[object_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
        
    ee_pos = get_ee_pos(env, robot_cfg, env_ids=env_ids)
    
    # RigidObject의 경우 root_pos_w 사용
    obj_pos = obj.data.root_pos_w[env_ids, :3]
    
    return obj_pos - ee_pos

def get_joint_vel_rel(env, asset_cfg, env_ids=None):
    asset = env.scene[asset_cfg.name]
    
    if env_ids is None:
        # Isaac Lab v1.1+ 에서는 num_instances를 사용합니다.
        # 만약 구버전이라면 hasattr 체크를 통해 호환성을 유지할 수 있습니다.
        if hasattr(asset, "num_instances"):
            num_envs = asset.num_instances
        else:
            num_envs = asset.num_envs
        env_ids = torch.arange(num_envs, device=asset.device)
    
    # asset.data.joint_vel은 [num_envs, num_joints] 형태입니다.
    return asset.data.joint_vel[env_ids]


def get_contact_force_with_object(
    env: ManagerBasedRLEnv, 
    sensor_names: list[str], 
    object_cfg: SceneEntityCfg, 
    env_ids: torch.Tensor
) -> torch.Tensor:
    """
    ID 추적이 불가능한 상황에서 센서에 걸리는 전체 힘을 안전하게 반환합니다.
    (실제 물체 판정은 호출부의 거리 조건과 결합되어 필터링됩니다.)
    """
    total_force = torch.zeros(len(env_ids), device=env.device)
    
    for name in sensor_names:
        if name not in env.scene.sensors:
            continue
            
        sensor = env.scene.sensors[name]
        
        # net_forces_w: (num_envs, num_bodies_in_sensor, 3)
        # 센서(보통 손가락 끝)에 가해지는 모든 물리적인 힘의 합을 가져옵니다.
        raw_forces = sensor.data.net_forces_w[env_ids]
        
        # 3축 힘의 크기(Norm) 계산 후 합산
        # 관절(body)이 여러 개일 수 있으므로 마지막 두 차원을 기준으로 합산합니다.
        if raw_forces.dim() == 3:
            # (num_envs, num_bodies, 3) -> (num_envs,)
            net_force_norm = torch.norm(raw_forces, dim=-1).sum(dim=-1)
        else:
            # (num_envs, 3) -> (num_envs,)
            net_force_norm = torch.norm(raw_forces, dim=-1)
            
        total_force += net_force_norm
        
    return total_force

def get_total_contact_force(env: ManagerBasedRLEnv, sensor_name: list, env_ids: torch.Tensor = None) -> torch.Tensor:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    
    total_force = torch.zeros(len(env_ids), device=env.device)
    for name in sensor_name:
        sensor = env.scene.sensors[name]
        # 센서 데이터에서 요청된 env_ids만 추출
        forces = sensor.data.net_forces_w[env_ids]
        if forces.dim() == 3:
            total_force += torch.norm(forces, dim=-1).sum(dim=-1)
        else:
            total_force += torch.norm(forces, dim=-1)
    return total_force


def get_contact_binary_penalty(env: ManagerBasedRLEnv, sensor_name: list, threshold: float = 1.0) -> torch.Tensor:
    """임계값(threshold) 이상의 힘이 감지되면 1.0(페널티)을 반환합니다."""
    total_force = get_total_contact_force(env, sensor_name)
    
    # 힘이 threshold를 넘으면 1.0, 아니면 0.0 반환
    return (total_force > threshold).float()

def modify_reward_weight_by_iteration(env, term_name, schedule, iteration):
    weight = schedule[0][1]
    for it, w in schedule:
        if iteration >= it:
            weight = w
        else:
            break
    
    env.reward_manager.get_term_cfg(term_name).weight = weight

    print(f"[CURRICULUM][ITER={iteration}] {term_name} weight -> {weight}")

    return weight

# def modify_reward_weight(
#     env: ManagerBasedRLEnv,
#     env_ids: torch.Tensor,
#     term_name: str,
#     schedule: list[tuple[int, float]],
# ) -> float:
#     """
#     수정된 전문가용 가중치 업데이트 함수:
#     - Resume 여부와 상관없이 '실제 총 누적 스텝'을 추적하여 가중치 반영
#     """
#     # [핵심] env.common_step_counter는 에피소드 단위 혹은 세션 단위로 리셋될 수 있으므로,
#     # 학습 알고리즘(RSL_RL 등)이 관리하는 global_step이 있다면 그것을 사용하는 것이 베스트입니다.
#     # 만약 없다면, resume 시점에 우리가 수동으로 주입한 값을 더해줍니다.
    
#     # 기본적으로 common_step_counter를 쓰되, env 객체에 수동 오프셋이 있는지 확인
#     offset = getattr(env, "step_offset", 0)
#     current_total_step = env.common_step_counter + offset

#     # 스케줄링 로직
#     weight = schedule[0][1]
#     for step_threshold, w in schedule:
#         if current_total_step >= step_threshold:
#             weight = w
#         else:
#             break
    

#     # 가중치 강제 주입 (지난번 DEBUG로 확인한 _term_weights가 없으므로 가장 안전한 방식 사용)
#     if hasattr(env, "reward_manager"):
#         term_cfg = env.reward_manager.get_term_cfg(term_name)
#         if term_cfg is not None:
#             term_cfg.weight = weight
        
#         # 내부 연산용 텐서 업데이트 (dir() 결과에 없더라도 안전하게 접근 시도)
#         # getattr를 활용해 에러 없이 유연하게 대응
#         term_weights = getattr(env.reward_manager, "_term_weights", None)
#         if term_weights is not None:
#             term_idx = env.reward_manager._term_names.index(term_name)
#             term_weights[term_idx] = weight
                
#     return weight