from typing import Any, Callable
import jax
import time
from multiprocessing import Process, shared_memory, Lock, Event
import numpy as np
import jax.numpy as jnp

import mujoco
import mujoco.viewer
from mujoco import mjx

from hydrax.tasks.cube import CubeRotation
from hydrax.algs import PredictiveSampling


class SharedMemoryNumpyArray:
    """Helper class to store a numpy array in shared memory."""
    def __init__(self, arr: np.ndarray):
        """Create a shared memory numpy array.

        Args:
            arr: The numpy array to store in shared memory. Size and dtype must
                 be fixed.
        """
        self.shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        self.data = np.ndarray(arr.shape, dtype=arr.dtype, buffer=self.shm.buf)
        self.data[:] = arr[:]
        self.lock = Lock()

    def __getitem__(self, key):
        """Get an item from the shared array."""
        return self.data[key]
    
    def __setitem__(self, key, value):
        """Set an item in the shared array."""
        with self.lock:
            self.data[key] = value

    def __str__(self):
        """Return the string representation of the shared array."""
        return str(self.data)

    def __del__(self):
        """Clean up the shared memory on deletion."""
        self.shm.close()
        self.shm.unlink()

    @property
    def shape(self):
        """Return the shape of the shared array."""
        return self.shared_arr.shape


def simulator(
    qpos: SharedMemoryNumpyArray,
    qvel: SharedMemoryNumpyArray,
    mocap_pos: SharedMemoryNumpyArray,
    mocap_quat: SharedMemoryNumpyArray,
    ctrl: SharedMemoryNumpyArray,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    run_time: float,
    ready: Event,
    finished: Event,
):
    """Run a simulation loop.

    Args:
        qpos: Where we write the qpos data into shared memory.
        qvel: Where we write the qvel data into shared memory.
        mocap_pos: Where we write the mocap_pos data into shared memory.
        mocap_quat: Where we write the mocap_quat data into shared memory.
        ctrl: Where we read the control data from shared memory.
        mj_model: Mujoco model for the simulation.
        mj_data: Mujoco data specifying the initial state.
        run_time: Total simulation time in seconds.
        ready: Shared flag for starting the simulation.
        finished: Shared flag for stopping the simulation.
    """
    # Wait for the controller to be ready
    ready.wait()

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running() and mj_data.time < run_time:
            start_time = time.time()

            # Send the latest state to shared memory, where the controller can
            # read it
            qpos[:] = mj_data.qpos
            qvel[:] = mj_data.qvel
            mocap_pos[:] = mj_data.mocap_pos
            mocap_quat[:] = mj_data.mocap_quat

            # Read the lastest control values from shared memory
            mj_data.ctrl[:] = ctrl[:]

            # Step the simulation
            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()

            # Try to run in roughly real-time
            elapsed_time = time.time() - start_time
            if elapsed_time < mj_model.opt.timestep:
                time.sleep(mj_model.opt.timestep - elapsed_time)

    # Signal that the simulation is done
    finished.set()

def controller(
    qpos: SharedMemoryNumpyArray,
    qvel: SharedMemoryNumpyArray,
    mocap_pos: SharedMemoryNumpyArray,
    mocap_quat: SharedMemoryNumpyArray,
    ctrl: SharedMemoryNumpyArray,
    ready: Event,
    finished: Event,
):
    """Run the controller loop.

    Args:
        qpos: Where we read the qpos data from shared memory.
        qvel: Where we read the qvel data from shared memory.
        mocap_pos: Where we read the mocap_pos data from shared memory.
        mocap_quat: Where we read the mocap_quat data from shared memory.
        ctrl: Where we write the control data into shared memory.
        ready: Shared flag for starting the simulation.
        finished: Shared flag for stopping the simulation.
    """
    # N.B. We need to set up the task and controller here, otherwise jax
    # complains about being in a multithreaded setting
    task = CubeRotation()
    controller = PredictiveSampling(
        task,
        num_samples=128,
        num_randomizations=8,
        noise_level=0.5,
    )
    mjx_data = mjx.make_data(task.model)
    policy_params = controller.init_params()
    
    jit_optimize = jax.jit(lambda d, p: controller.optimize(d,p)[0], donate_argnums=(1,))

    print("Jitting controller...")
    st = time.time()
    policy_params = jit_optimize(mjx_data, policy_params)
    print(f"Time to jit: {time.time() - st}")
    print("")

    # Signal that the controller is ready to go
    ready.set()

    while not finished.is_set():
        # Set the start state for the controller, reading the lastest state info
        # from shared memory
        mjx_data = mjx_data.replace(
            qpos=jnp.array(qpos.data),
            qvel=jnp.array(qvel.data),
            mocap_pos=jnp.array(mocap_pos.data),
            mocap_quat=jnp.array(mocap_quat.data),
        )

        # Do a planning step
        st = time.time()
        policy_params = jit_optimize(mjx_data, policy_params)
        print(f"Plan time: {time.time() - st}")

        # Send the action to the simulator
        ctrl[:] = np.array(controller.get_action(policy_params, 0.0), dtype=np.float32)


if __name__=="__main__":
    run_time = 60.0  # Total sim time, in seconds

    # Set up the simulator model
    mj_model = mujoco.MjModel.from_xml_path("./models/scene.xml")
    mj_data = mujoco.MjData(mj_model)

    # Set the initial state
    mj_data.qpos[:] = mj_model.qpos0
    mj_data.qvel[:] = np.zeros(mj_model.nv)
    mj_data.mocap_quat[0] = np.array([1., 0., 0., 0.])

    # Create shared_memory data
    shm_qpos = SharedMemoryNumpyArray(
        np.asarray(mj_data.qpos, dtype=np.float32))
    shm_qvel = SharedMemoryNumpyArray(
        np.asarray(mj_data.qvel, dtype=np.float32))
    shm_mocap_pos = SharedMemoryNumpyArray(
        np.asarray(mj_data.mocap_pos, dtype=np.float32))
    shm_mocap_quat = SharedMemoryNumpyArray(
        np.asarray(mj_data.mocap_quat, dtype=np.float32))
    shm_ctrl = SharedMemoryNumpyArray(
        np.zeros(mj_model.nu, dtype=np.float32))
    ready = Event()
    finished = Event()
    
    # Set up the simulator and controller processes
    sim = Process(target=simulator, args=(
        shm_qpos, shm_qvel, shm_mocap_pos, shm_mocap_quat, shm_ctrl,
        mj_model, mj_data, run_time, ready, finished))
    control = Process(target=controller, args=(
        shm_qpos, shm_qvel, shm_mocap_pos, shm_mocap_quat, shm_ctrl, ready, finished))

    # Run the simulation and controller in parallel 
    sim.start()
    control.start()
    sim.join()
    control.join()

    # Clean up shared memory
    del shm_qpos, shm_qvel, shm_mocap_pos, shm_mocap_quat, shm_ctrl