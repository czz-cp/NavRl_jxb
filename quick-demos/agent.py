import torch
import os
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from tensordict.tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from ppo import PPO


class Agent:
    def __init__(self, device):
        self.device = device
        self.policy = self.init_model()

    # PPO policy loader
    def init_model(self):
        observation_dim = 8
        num_dim_each_dyn_obs_state = 10
        observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.device), 
                    "lidar": UnboundedContinuousTensorSpec((1, 36, 4), device=self.device),
                    "direction": UnboundedContinuousTensorSpec((1, 3), device=self.device),
                    "dynamic_obstacle": UnboundedContinuousTensorSpec((1, 5, 10), device=self.device),
                }),
            }).expand(1)
        }, shape=[1], device=self.device)

        action_dim = 3
        action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": UnboundedContinuousTensorSpec((action_dim,), device=self.device), 
            })
        }).expand(1, action_dim).to(self.device)

        policy = PPO(observation_spec, action_spec, self.device)

        file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpts")
        checkpoint = "navrl_checkpoint.pt"

        policy.load_state_dict(torch.load(os.path.join(file_dir, checkpoint), map_location=self.device))
        return policy
    
    def plan(self, robot_state, static_obs_input, dyn_obs_input, target_dir):
        obs = TensorDict({
            "agents": TensorDict({
                "observation": TensorDict({
                    "state": robot_state,
                    "lidar": static_obs_input,
                    "direction": target_dir,
                    "dynamic_obstacle": dyn_obs_input,
                })
            })
        }, device=self.device)

        with set_exploration_type(ExplorationType.MEAN):
            output = self.policy(obs)
            velocity = output["agents", "action"][0][0].detach().cpu().numpy()[:2] 
        return velocity