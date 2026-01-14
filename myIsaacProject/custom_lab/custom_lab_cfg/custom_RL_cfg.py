from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


@configclass
class CustomRunnerCfg(RslRlOnPolicyRunnerCfg):
    experiment_name = "Isaac-E0509-Curriculum-v0"
    run_name = "0113_2000_exp_start_stage123" 
    task = "Isaac-E0509-Curriculum-v0"
    
    num_envs = 1024             
    max_iterations = 1000        
    save_interval = 50
    #필수확인!
    resume_iter = 300
    
    num_steps_per_env = 128      
    logger = "tensorboard"

    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"],
    }

    policy = {
        "class_name": "ActorCritic",
        "activation": "elu",
        "actor_hidden_dims": [256, 256],
        "critic_hidden_dims": [256, 256],
    }

    algorithm = {
        "class_name": "PPO",
        "learning_rate": 3e-4,
        "num_learning_epochs": 5,
        "num_mini_batches": 16,
        "clip_param": 0.2,
        "gamma": 0.99,
        "lam": 0.95,
        "entropy_coef": 0.01,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
    }
