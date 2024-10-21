#!/usr/bin/env python

import mujoco
import numpy as np

from hydrax import ROOT
from hydrax.algs import PredictiveSampling
from hydrax.mpc import run_interactive
from hydrax.tasks.cube import CubeRotation

"""
Run an interactive simulation of the cube rotation task.
"""

# Define the task (cost and dynamics)
task = CubeRotation()

# Set up the controller
ctrl = PredictiveSampling(
task, num_samples=32, noise_level=0.2, num_randomizations=32
)

# Define the model used for simulation
mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/cube/scene.xml")
start_state = np.concatenate([mj_model.qpos0, np.zeros(mj_model.nv)])

# Run the interactive simulation
run_interactive(
    mj_model,
    ctrl,
    start_state,
    frequency=25,
    fixed_camera_id=None,
    show_traces=True,
    max_traces=1,
    trace_color=[1.0, 1.0, 1.0, 1.0],
)
