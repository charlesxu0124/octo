from functools import partial
import time
import cv2
import flax
import numpy as np
from serl_experiments import config
from serl_experiments.connector_insert.config import TrainConfig as ConnectorInsertTrainConfig
import tensorflow as tf
from octo.model.octo_model import OctoModel

import jax
import absl.app
import absl.flags

def rollout(
    env,
    agent,
    max_path_length=np.inf,
    o=None,
):
    observations = []
    actions = []
    path_length = 0
    o, _ = env.reset()
    rng=jax.random.PRNGKey(0)
    while path_length < max_path_length:
        # Get action from policy
        rng, sample_rng = jax.random.split(rng)
        a = agent(o, timestep=path_length, rng=sample_rng)
        # print(a[-1])
        next_o, rew, done, truncated, info = env.step(a)
        observations.append(o)
        actions.append(a)
        path_length += 1
        o = next_o
        if done:
            print("Reward: ", rew)
            break
    return dict(
        observations=observations,
        actions=actions,
    ), rew


def main(_):
    checkpoint_path = "/media/nvmep3p/octo_checkpoints/octo_finetune/experiment/connector_human_75"
    # Load example batch
    example_batch_path = tf.io.gfile.join(checkpoint_path, "example_batch.msgpack")
    with tf.io.gfile.GFile(example_batch_path, "rb") as f:
        example_batch = flax.serialization.msgpack_restore(f.read())
    # Load Processor & VLA
    model = OctoModel.load_pretrained(checkpoint_path)
    tasks = model.create_tasks(texts=["insert the VGA connector"])
    def policy(obs, timestep, rng):
        # Add batch dimension to each observation
        # Add "image" to observation keys that are not "state"
        obs = {
            "image_primary": obs['wrist_1'][None],
            'pad_mask_dict': {
                'image_primary': np.ones((1, 1), dtype=bool),
                'timestep': np.ones((1, 1), dtype=bool),
            },
            'timestep_pad_mask': np.ones((1, 1), dtype=bool),
            'task_completed': np.zeros((1, 1, 1), dtype=bool),
            'timestep': np.array(timestep, dtype=int).reshape(1, 1),
        }
        # batched_obs = {
        #     key: np.expand_dims(value, axis=0) 
        #     for key, value in obs.items()
        # }
        # batched_obs['timestep_pad_mask'] = np.zeros((1, 1), dtype=bool)
        actions = model.sample_actions(obs, tasks, rng=rng)
        return actions[0, 0, :6]

    config = ConnectorInsertTrainConfig()
    env = config.get_environment(
        fake_env=False,
        save_video=False,
        classifier=False,
    )

    # trigger jit
    rng = jax.random.PRNGKey(0)
    rng, rng1, rng2 = jax.random.split(rng, 3)
    policy(env.observation_space.sample(), 0, rng1)
    policy(env.observation_space.sample(), 0, rng2)

    success_count = 0
    cycle_times = []
    for n in range(20):
        start_time = time.time()
        _, rew = rollout(env, policy, max_path_length=200)
        finish_time = time.time()
        if rew:
            cycle_times.append(finish_time - start_time)
            success_count += 1
        print(f"Success Rate: {success_count} / {n+1}")
        print(f"Average Cycle Time: {np.mean(cycle_times)}")


if __name__ == "__main__":
    absl.app.run(main)
