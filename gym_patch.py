import gym
from functools import wraps

# Store the original reset method
original_reset = gym.Env.reset

# Create a patched version that adapts the return value
@wraps(original_reset)
def patched_reset(self, **kwargs):
    result = original_reset(self, **kwargs)
    # If result is already a tuple with length 2, it's likely (obs, info)
    if isinstance(result, tuple) and len(result) == 2:
        return result
    # If result is a tuple with length 3, it's likely (obs, reward, done)
    elif isinstance(result, tuple) and len(result) == 3:
        obs, _, _ = result
        return obs, {}
    # If result is just the observation
    else:
        return result, {}

# Apply the patch
def apply_patch():
    gym.Env.reset = patched_reset