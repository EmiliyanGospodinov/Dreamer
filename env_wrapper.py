import gym
import sys
import torch
import numpy as np
from PIL import Image
import cv2


class DeepMindControl:

    def __init__(self, name, seed, size=(64, 64), camera=None):

        domain, task = name.split('-', 1)
        if domain == 'cup':  # Only domain with multiple words.
          domain = 'ball_in_cup'
        if isinstance(domain, str):
          from dm_control import suite
          self._env = suite.load(domain, task, task_kwargs={'random':seed})
        else:
          assert task is None
          self._env = domain()
        self._size = size
        if camera is None:
          camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
          spaces[key] = gym.spaces.Box(
              -np.inf, np.inf, value.shape, dtype=np.float32)
        spaces['image'] = gym.spaces.Box(
            0, 255, (3,) + self._size , dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        time_step = self._env.step(action)
        obs = dict(time_step.observation)
        obs['image'] = self.render().transpose(2, 0, 1).copy()
        reward = time_step.reward or 0
        done = time_step.last()
        info = {'discount': np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs['image'] = self.render().transpose(2, 0, 1).copy()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
          raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)


class GymEnv:
    def __init__(
        self, env, symbolic, seed, terminate_when_unhealthy=None, bit_depth=5, obs_size=(64, 64),
    ):
        import gymnasium as gym

        self.symbolic = symbolic
        if self.symbolic:
            self._env = gym.make(env)
        elif terminate_when_unhealthy:
            self._env = gym.make(env, render_mode="rgb_array", terminate_when_unhealthy=terminate_when_unhealthy) # terminate_when_unhealthy=True
        else:
            self._env = gym.make(env, render_mode="rgb_array")
        self._seed = seed
        self._bit_depth = bit_depth
        self._obs_size = obs_size

    def reset(self):
        self.t = 0  # Reset internal timer
        # seed is passed over the reset method of the environment
        state, _ = self._env.reset(seed=self._seed)

        if self.symbolic:
            return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
        else:
            return self._make_observation(self.render())

    def step(self, action):
        state, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        if self.symbolic:
            observation = torch.tensor(state, dtype=torch.float32)
        else:
            observation = self._make_observation(
                self.render()
            )
        return observation, reward, done, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    @property
    def observation_space(self):
        spaces = {}
        spaces["obs_space"] = self._env.observation_space
        spaces['image'] = gym.spaces.Box(
            0, 255, (3,) + self._obs_size , dtype=np.uint8)
        return spaces

    @property
    def action_space(self):
        return self._env.action_space

    # Preprocesses an observation inplace (from float32 Tensor [0, 255] to [-0.5, 0.5])
    def _preprocess_observation(self, observation):
        observation.div_(2 ** (8 - self._bit_depth)).floor_().div_(2**self._bit_depth).sub_(
            0.5
        )  # Quantise to given bit depth and centre
        observation.add_(
            torch.rand_like(observation).div_(2**self._bit_depth)
        )  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)

        return observation

    # # Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
    # def _postprocess_observation(self, observation):
    #     obs = np.clip(
    #         np.floor((observation["image"] + 0.5) * 2**self._bit_depth) * 2 ** (8 - self._bit_depth),
    #         0,
    #         2**8 - 1,
    #     )
    #     obs = torch.from_numpy(obs, dtype=torch.uint8)

    #     return obs

    def _make_observation(self, obs):
        image = np.array(
            cv2.resize(obs, (64, 64), interpolation=cv2.INTER_AREA).transpose(2, 0, 1),
        ).astype(np.float32)  # Resize and put channel first
        # observation = self._preprocess_observation(
        #     images, self._bit_depth
        # )  # Quantise, centre and dequantise inplace
        # observation = {}
        # observation["image"] = obs
        #observation["raw"] = obs.transpose(2, 0, 1)
        return {"image": image}   


class TimeLimit:

    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        assert self._step is not None, 'Must reset environment.'
        obs, reward, done, info = self._env.step(action)
        self._step += 1
        if self._step >= self._duration:
          done = True
          if 'discount' not in info:
            info['discount'] = np.array(1.0).astype(np.float32)
          self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self._env.reset()


class ActionRepeat:

    def __init__(self, env, amount):
        self._env = env
        self._amount = amount

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self._amount and not done:
          obs, reward, done, info = self._env.step(action)
          total_reward += reward
          current_step += 1
        return obs, total_reward, done, info


class NormalizeActions:

    def __init__(self, env):
        self._env = env
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low),
            np.isfinite(env.action_space.high))
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self._env.step(original)


class ObsDict:

    def __init__(self, env, key='obs'):
        self._env = env
        self._key = key

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        spaces = {self._key: self._env.observation_space}
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return self._env.action_space

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = {self._key: np.array(obs)}
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs = {self._key: np.array(obs)}
        return obs


class OneHotAction:

    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        shape = (self._env.action_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.sample = self._sample_action
        return space

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
          raise ValueError(f'Invalid one-hot action:\n{action}')
        return self._env.step(index)

    def reset(self):
        return self._env.reset()

    def _sample_action(self):
        actions = self._env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class RewardObs:

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        spaces = self._env.observation_space.spaces
        assert 'reward' not in spaces
        spaces['reward'] = gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
        return gym.spaces.Dict(spaces)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs['reward'] = reward
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs['reward'] = 0.0
        return obs


class ResizeImage:

    def __init__(self, env, size=(64, 64)):
        self._env = env
        self._size = size
        self._keys = [
            k for k, v in env.obs_space.items()
            if len(v.shape) > 1 and v.shape[:2] != size]
        print(f'Resizing keys {",".join(self._keys)} to {self._size}.')
        if self._keys:
          from PIL import Image
          self._Image = Image

    def __getattr__(self, name):
        if name.startswith('__'):
          raise AttributeError(name)
        try:
          return getattr(self._env, name)
        except AttributeError:
          raise ValueError(name)

    @property
    def obs_space(self):
        spaces = self._env.obs_space
        for key in self._keys:
          shape = self._size + spaces[key].shape[2:]
          spaces[key] = gym.spaces.Box(0, 255, shape, np.uint8)
        return spaces

    def step(self, action):
        obs = self._env.step(action)
        for key in self._keys:
          obs[key] = self._resize(obs[key])
        return obs

    def reset(self):
        obs = self._env.reset()
        for key in self._keys:
          obs[key] = self._resize(obs[key])
        return obs

    def _resize(self, image):
        image = self._Image.fromarray(image)
        image = image.resize(self._size, self._Image.NEAREST)
        image = np.array(image)
        return image


class RenderImage:

    def __init__(self, env, key='image'):
        self._env = env
        self._key = key
        self._shape = self._env.render().shape

    def __getattr__(self, name):
        if name.startswith('__'):
          raise AttributeError(name)
        try:
          return getattr(self._env, name)
        except AttributeError:
          raise ValueError(name)

    @property
    def obs_space(self):
        spaces = self._env.obs_space
        spaces[self._key] = gym.spaces.Box(0, 255, self._shape, np.uint8)
        return spaces

    def step(self, action):
        obs = self._env.step(action)
        obs[self._key] = self._env.render('rgb_array')
        return obs

    def reset(self):
        obs = self._env.reset()
        obs[self._key] = self._env.render('rgb_array')
        return obs
