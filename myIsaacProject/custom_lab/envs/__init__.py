import gymnasium as gym
from .task_cfg import E0509TaskCfg

# 태스크 등록
# gym.register(
#     id="Isaac-Reach-E0509-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "cfg_entry_point": E0509TaskCfg,
#     },
# )

# gym.register(
#     id="Isaac-E0509-Curriculum-v0",
#     entry_point=E0509CurriculumEnv,
#     disable_env_checker=True,
#     kwargs={"cfg_class": E0509TaskCfg}
# )


# gym.register(
#     id="Isaac-E0509-Curriculum-v0",
#     entry_point="custom_lab.envs.task_cfg:E0509CurriculumEnv", # 클래스가 정의된 위치
#     disable_env_checker=True,
#     kwargs={
#         "cfg_entry_point": "custom_lab.envs.task_cfg:E0509TaskCfg" # cfg_class 대신 cfg_entry_point 사용
#     }
# )


# IsaacLab Task + Curriculum 바로 등록
gym.register(
    id="Isaac-E0509-Curriculum-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",  # 기본 RL 환경 사용
    disable_env_checker=True,
    kwargs={
        "cfg_entry_point": E0509TaskCfg  # Curriculum 포함 TaskCfg 전달
    }
)