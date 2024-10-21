#!/usr/bin/env python

import mujoco
import mujoco.viewer
import numpy as np
import time

import jax
import jax.numpy as jnp
from mujoco import mjx

from hydrax import ROOT
from hydrax.algs import PredictiveSampling
from hydrax.tasks.cube import CubeRotation

"""
Run an automated benchmark of the cube rotation task.
"""


######################## UTILITIES ########################
np.random.seed(0)

def random_quat():
    """Generate a random unit quaternion."""
    u, v, w = np.random.uniform(size=3)
    return np.array([
        np.sqrt(1 - u) * np.sin(2 * np.pi * v),
        np.sqrt(1 - u) * np.cos(2 * np.pi * v),
        np.sqrt(u) * np.sin(2 * np.pi * w),
        np.sqrt(u) * np.cos(2 * np.pi * w),
    ])


######################## CONTROLLER SETUP ########################

# Define the task (cost and dynamics)
task = CubeRotation()

# Set up the controller
ctrl = PredictiveSampling(
    task, num_samples=32, noise_level=0.2, num_randomizations=32
)

# Desired planning frequency (Hz)
frequency = 25


######################## SIMULATOR SETUP ########################

# Define the model used for simulation
mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/cube/scene.xml")
mj_model.opt.timestep = 0.005

# Set the initial state
mj_data = mujoco.MjData(mj_model)
mj_data.qpos[:] = mj_model.qpos0
mj_data.qvel[:] = np.zeros(mj_model.nv)
mj_data.mocap_quat[0] = random_quat()


######################## SIMULATION LOOP ########################

print(
    f"Planning with {ctrl.task.planning_horizon} steps "
    f"over a {ctrl.task.planning_horizon * ctrl.task.dt} "
    f"second horizon."
)

# Figure out how many sim steps to run before replanning
replan_period = 1.0 / frequency
sim_steps_per_replan = int(replan_period / mj_model.opt.timestep)
sim_steps_per_replan = max(sim_steps_per_replan, 1)
step_dt = sim_steps_per_replan * mj_model.opt.timestep
actual_frequency = 1.0 / step_dt
print(
    f"Planning at {actual_frequency} Hz, "
    f"simulating at {1.0/mj_model.opt.timestep} Hz"
)

# Initialize the controller
mjx_data = mjx.put_data(mj_model, mj_data)
mjx_data = mjx_data.replace(
    mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat
)
policy_params = ctrl.init_params()
jit_optimize = jax.jit(lambda d, p: ctrl.optimize(d,p)[0], donate_argnums=(1,))

# Warm-up the controller
st = time.time()
policy_params = jit_optimize(mjx_data, policy_params)
print(f"Time to jit: {time.time() - st:.3f} seconds")

# Start the simulation
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
        start_time = time.time()

        # Set the start state for the controller
        mjx_data = mjx_data.replace(
            qpos=jnp.array(mj_data.qpos),
            qvel=jnp.array(mj_data.qvel),
            mocap_pos=jnp.array(mj_data.mocap_pos),
            mocap_quat=jnp.array(mj_data.mocap_quat),
        )

        # Do a replanning step
        plan_start = time.time()
        policy_params = jit_optimize(mjx_data, policy_params)
        plan_time = time.time() - plan_start

        # Step the simulation
        for i in range(sim_steps_per_replan):
            t = i * mj_model.opt.timestep
            u = ctrl.get_action(policy_params, t)
            mj_data.ctrl[:] = np.array(u)
            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()

        # Check whether we're close to the target orientation
        err = task._get_cube_orientation_err(mj_data)
        if np.linalg.norm(err) < 0.4:
            mj_data.mocap_quat[0] = random_quat()
            print("Done!")

        # Check whether we dropped the cube
        pos = mj_data.site_xpos[mj_model.site("cube_center").id]
        if pos[2] < -0.08:
            mj_data.qpos[:] = mj_model.qpos0
            print("Dropped!")