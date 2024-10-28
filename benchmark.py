#!/usr/bin/env python

import mujoco
import numpy as np
import time

import jax
import jax.numpy as jnp
from mujoco import mjx
import evosax

from async_utils import run_benchmark, random_quat

from hydrax.algs import PredictiveSampling, CEM, Evosax
from hydrax.tasks.cube import CubeRotation
from hydrax.risk import WorstCase, ConditionalValueAtRisk

"""
Run an automated benchmark of the cube rotation task.
"""

# Set the seed (but note that asynchronous simulation is not deterministic)
np.random.seed(0)

# Set up the controller. Note that this needs to be a function for JAX and
# python multiprocessing to play well together.
def setup_controller():
    task = CubeRotation()
    ctrl = PredictiveSampling(
        task,
        num_samples=64,
        num_randomizations=8,
        noise_level=0.5,
        risk_strategy=ConditionalValueAtRisk(0.2),
    )
    return ctrl

# Define the model used for simulation
mj_model = mujoco.MjModel.from_xml_path("./models/scene.xml")

# Set the initial state
mj_data = mujoco.MjData(mj_model)
mj_data.qpos[:] = mj_model.qpos0
mj_data.qvel[:] = np.zeros(mj_model.nv)
mj_data.mocap_quat[0] = random_quat()

# Total simulation time (seconds)
run_time = 60.0

# Run the simulation
num_rotations, num_drops = run_benchmark(
    setup_controller, mj_model, mj_data, run_time)

print(f"Rotations: {num_rotations}, Drops: {num_drops}")
