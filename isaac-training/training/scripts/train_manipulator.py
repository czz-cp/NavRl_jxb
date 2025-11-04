"""
Training script for UR10e Manipulator Navigation
UR10e机械臂导航训练脚本
"""
import os
import hydra
import datetime
import wandb
import torch
from omegaconf import DictConfig, OmegaConf
from omni.isaac.kit import SimulationApp
from ppo_manipulator import PPO
from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
from torchrl.envs.transforms import TransformedEnv, Compose
from utils import evaluate
from torchrl.envs.utils import ExplorationType\

# 这个环境使用了 Isaac Sim 的框架驱动模式，而不是传统的 step() 函数
# 训练通常通过 TorchRL 的训练器 启动
"""# 在收集器内部循环：
for _ in range(frames_per_batch):
    # 调用 transformed_env.step() 或等效方法
    # 这个step()会触发Isaac Sim的回调序列：
    # _pre_sim_step() → sim.step() → _post_sim_step() →
    #  _compute_state_and_obs() → _compute_reward_and_done()"""

FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")

@hydra.main(config_path=FILE_PATH, config_name="train_manipulator", version_base=None)
def main(cfg):
    # ========== Simulation App ==========
    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})
    
    # ========== Wandb ==========
    if cfg.wandb.run_id is None:
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
            id=wandb.util.generate_id(),
        )
    else:
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
            id=cfg.wandb.run_id,
            resume="must"
        )
    
    # ========== Environment ==========
    from manipulator_env import ManipulatorNavigationEnv
    env = ManipulatorNavigationEnv(cfg)
    
    # ========== Transformed Environment ==========
    transforms = []
    # 不需要VelController，直接输出末端速度
    # TransformedEnv 是一个环境包装器，它接受一个基础环境并通过一系列转换（Transforms） 来修改环境的输入/输出行为
    transformed_env = TransformedEnv(env, Compose(*transforms)).train()
    transformed_env.set_seed(cfg.seed)
    
    # ========== PPO Policy ==========
    policy = PPO(
        cfg.algo, 
        transformed_env.observation_spec, 
        transformed_env.action_spec, 
        cfg.device
    )
    
    # 可选：加载预训练检查点
    # checkpoint = "path/to/checkpoint.pt"
    # policy.load_state_dict(torch.load(checkpoint))
    
    # ========== Episode Stats ==========
    episode_stats_keys = [
        k for k in transformed_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(episode_stats_keys)
    
    # ========== Data Collector ==========
    collector = SyncDataCollector(
        transformed_env,
        policy=policy,
        frames_per_batch=cfg.env.num_envs * cfg.algo.training_frame_num,
        total_frames=cfg.max_frame_num,
        device=cfg.device,
        return_same_td=True,
        exploration_type=ExplorationType.RANDOM,
    )
    
    # ========== Training Loop ==========
    print("[Manipulator Training]: Starting training loop...")
    for i, data in enumerate(collector):
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
        
        # Train
        train_loss_stats = policy.train(data)
        info.update(train_loss_stats)
        
        # Episode stats
        episode_stats.add(data)
        if len(episode_stats) >= transformed_env.num_envs:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)
        
        # Evaluate
        if i % cfg.eval_interval == 0:
            print(f"[Manipulator Training]: Evaluating at step {i}...")
            env.enable_render(True)
            env.eval()
            eval_info = evaluate(
                env=transformed_env,
                policy=policy,
                seed=cfg.seed,
                cfg=cfg,
                exploration_type=ExplorationType.MEAN
            )
            env.enable_render(not cfg.headless)
            env.train()
            env.reset()
            info.update(eval_info)
            print("[Manipulator Training]: Evaluation done.\n")
        
        # Log
        run.log(info)
        
        # Save
        if i % cfg.save_interval == 0:
            ckpt_path = os.path.join(run.dir, f"checkpoint_{i}.pt")
            torch.save(policy.state_dict(), ckpt_path)
            print(f"[Manipulator Training]: Model saved at step {i}")
    
    # Final save
    ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
    torch.save(policy.state_dict(), ckpt_path)
    wandb.finish()
    sim_app.close()

if __name__ == "__main__":
    main()

