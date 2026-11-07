"""AD model with a second input: the agent's own top-down map (common/pixel_map.py).

Observations are (6, 64, 64): channels 0-2 the game frame, 3-5 the rendered egocentric map (32x32 tiles at
2 px, the agent at the centre). The frame goes through the original encoder; the map through a smaller one
of the same design. Both feature vectors are concatenated before the layer that makes the latent, so
everything downstream (policy, value, AD's goal states and matching) is unchanged and sees both.
"""
from typing import Dict

import torch as th
from gym import spaces

from achievement_distillation.impala_cnn import ImpalaCNN
from achievement_distillation.model.ppo_ad import PPOADModel
from achievement_distillation.torch_util import FanInInitReLULayer


class PPOADMapModel(PPOADModel):
    def __init__(self, observation_space: spaces.Box, action_space: spaces.Discrete, hidsize: int,
                 impala_kwargs: Dict = {}, dense_init_norm_kwargs: Dict = {}, map_chans=(32, 64, 64), **kwargs):
        frame_space = spaces.Box(low=0, high=1, shape=(3, 64, 64))
        super().__init__(frame_space, action_space, hidsize, impala_kwargs=impala_kwargs,
                         dense_init_norm_kwargs=dense_init_norm_kwargs, **kwargs)
        self.observation_space = observation_space
        map_kwargs = {**impala_kwargs, "chans": list(map_chans)}
        self.map_enc = ImpalaCNN((3, 64, 64), dense_init_norm_kwargs=dense_init_norm_kwargs, **map_kwargs)
        outsize = impala_kwargs["outsize"]
        self.linear = FanInInitReLULayer(2 * outsize, hidsize, layer_type="linear", **dense_init_norm_kwargs)

    def encode(self, obs: th.Tensor) -> th.Tensor:
        x = th.cat([self.enc(obs[:, :3]), self.map_enc(obs[:, 3:])], dim=-1)
        return self.linear(x)
