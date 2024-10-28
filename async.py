from typing import Any, Callable, Tuple
import jax
import time
from multiprocessing import Process, shared_memory, Lock, Event, Queue
import numpy as np
import jax.numpy as jnp

import mujoco
import mujoco.viewer
from mujoco import mjx

from hydrax.alg_base import SamplingBasedController
from hydrax.tasks.cube import CubeRotation
from hydrax.algs import PredictiveSampling


"""
Utilities for asynchronous simulation via `multiprocessing`.
"""


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


class SharedMemoryMujocoData:
    """Helper class for passing mujoco data between concurrent processes."""

    def __init__(self, mj_data: mujoco.MjData):
        """Create shared memory objects for state and control data.

        Note that this does not copy the full mj_data object, only those fields
        that we want to share between the simulator and controller.

        Args:
            mj_data: The mujoco data object to store in shared memory.
        """
        # N.B. we use float32 to match JAX's default precision
        self.qpos = SharedMemoryNumpyArray(
            np.array(mj_data.qpos, dtype=np.float32))
        self.qvel = SharedMemoryNumpyArray(
            np.array(mj_data.qvel, dtype=np.float32))
        self.mocap_pos = SharedMemoryNumpyArray(
            np.array(mj_data.mocap_pos, dtype=np.float32))
        self.mocap_quat = SharedMemoryNumpyArray(
            np.array(mj_data.mocap_quat, dtype=np.float32))
        self.ctrl = SharedMemoryNumpyArray(
            np.zeros(mj_data.ctrl.shape, dtype=np.float32))


def random_quat() -> np.ndarray:
    """Generate a random unit quaternion."""
    u, v, w = np.random.uniform(size=3)
    return np.array([
        np.sqrt(1 - u) * np.sin(2 * np.pi * v),
        np.sqrt(1 - u) * np.cos(2 * np.pi * v),
        np.sqrt(u) * np.sin(2 * np.pi * w),
        np.sqrt(u) * np.cos(2 * np.pi * w),
    ])


def simulator(
    shm_data: SharedMemoryMujocoData,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    run_time: float,
    ready: Event,
    finished: Event,
    rotation_times: Queue,
    drop_times: Queue,
) -> None:
    """Run a simulation loop.

    Args:
        shm_data: Shared memory object for communicating with the controller.
        mj_model: Mujoco model for the simulation.
        mj_data: Mujoco data specifying the initial state.
        run_time: Total simulation time in seconds.
        ready: Shared flag for starting the simulation.
        finished: Shared flag for stopping the simulation.
        rotation_times: Queue for storing the time taken for each rotation.
        drop_times: Queue for storing the time taken for each drop.
    """
    # Wait for the controller to be ready
    ready.wait()

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running() and mj_data.time < run_time:
            start_time = time.time()

            # Send the latest state to shared memory, where the controller can
            # read it
            shm_data.qpos[:] = np.copy(mj_data.qpos)
            shm_data.qvel[:] = np.copy(mj_data.qvel)
            shm_data.mocap_pos[:] = np.copy(mj_data.mocap_pos)
            shm_data.mocap_quat[:] = np.copy(mj_data.mocap_quat)

            # Read the lastest control values from shared memory
            mj_data.ctrl[:] = np.copy(shm_data.ctrl[:])

            # Step the simulation
            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()

            # Check if we're close to the target orientation
            err = np.zeros(3)
            q1 = mj_data.mocap_quat[0]
            q2 = mj_data.qpos[-4:]
            mujoco.mju_subQuat(err, q1, q2)
            if np.linalg.norm(err) < 0.4:
                mj_data.mocap_quat[0] = random_quat()
                rotation_times.put(mj_data.time)

            # Check if the cube has fallen off the hand
            pos = mj_data.site_xpos[mj_model.site("cube_center").id]
            if pos[2] < -0.08:
                mj_data.mocap_quat[0] = random_quat()
                mj_data.qpos[:] = mj_model.qpos0
                drop_times.put(mj_data.time)

            # Try to run in roughly real-time
            elapsed_time = time.time() - start_time
            if elapsed_time < mj_model.opt.timestep:
                time.sleep(mj_model.opt.timestep - elapsed_time)

    # Signal that the simulation is done
    finished.set()


def controller(
    shm_data: SharedMemoryMujocoData,
    setup_fn: Callable[[], SamplingBasedController],
    ready: Event,
    finished: Event,
    rotation_times: Queue,
    drop_times: Queue,
) -> None:
    """Run the controller loop.

    Note that we need to create the controller within this process, otherwise
    JAX will complain about sharing data across processes.

    Args:
        shm_data: Shared memory object for communicating with the simulator.
        setup_fn: Function to set up the controller.
        ready: Shared flag for starting the simulation.
        finished: Shared flag for stopping the simulation.
        rotation_times: Queue for storing the time taken for each rotation.
        drop_times: Queue for storing the time taken for each drop.
    """
    # Set up the controller
    ctrl = setup_fn()
    mjx_data = mjx.make_data(ctrl.task.model)
    policy_params = ctrl.init_params()

    # Jit the optimizer step, then signal that we're ready to go
    print("Jitting controller...")
    st = time.time()
    jit_optimize = jax.jit(lambda d, p: ctrl.optimize(d, p)[
                           0], donate_argnums=(1,))
    get_action = jax.jit(ctrl.get_action)
    policy_params = jit_optimize(mjx_data, policy_params)
    print(f"Time to jit: {time.time() - st}")

    # Signal that we're ready to start
    ready.set()

    while not finished.is_set():
        st = time.time()

        # Set the start state for the controller, reading the lastest state info
        # from shared memory
        mjx_data = mjx_data.replace(
            qpos=jnp.array(shm_data.qpos.data),
            qvel=jnp.array(shm_data.qvel.data),
            mocap_pos=jnp.array(shm_data.mocap_pos.data),
            mocap_quat=jnp.array(shm_data.mocap_quat.data),
        )

        # Do a planning step
        policy_params = jit_optimize(mjx_data, policy_params)

        # Send the action to the simulator
        shm_data.ctrl[:] = np.array(get_action(
            policy_params, 0.0), dtype=np.float32)

        # Print the current planning frequency and other info
        num_rotations = rotation_times.qsize()
        num_drops = drop_times.qsize()
        print(
            f"{1/(time.time() - st):.2f} Hz, "
            f"{num_rotations} rotations, "
            f"{num_drops} drops", end="\r")


def run_benchmark(
        make_controller: Callable[[], SamplingBasedController],
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        run_time: float,
) -> Tuple[int, int]:
    """Run the cube rotation benchmark with asynchronous sim and control.

    Args:
        make_controller: Function that sets up the controller.
        mj_model: Mujoco model for the simulation.
        mj_data: Mujoco data specifying the initial state.
        run_time: Total simulation time in seconds.

    Returns:
        Number of successful rotations during the simulation.
        Number of drops during the simulation.
    """
    # Create shared_memory data
    shm_data = SharedMemoryMujocoData(mj_data)
    ready = Event()
    finished = Event()
    rotation_times = Queue()
    drop_times = Queue()

    # Set up the simulator and controller processes
    sim = Process(target=simulator, args=(
        shm_data, mj_model, mj_data, run_time, ready, finished, rotation_times, drop_times))
    control = Process(target=controller, args=(
        shm_data, make_controller, ready, finished, rotation_times, drop_times))

    # Run the simulation and controller in parallel
    sim.start()
    control.start()
    sim.join()
    control.join()

    print("\nSimulation finished.")
    num_rotations = rotation_times.qsize()
    num_drops = drop_times.qsize()
    return num_rotations, num_drops


if __name__ == "__main__":
    np.random.seed(0)
    run_time = 60.0  # Total sim time, in seconds

    # Set up the simulator model
    mj_model = mujoco.MjModel.from_xml_path("./models/scene.xml")
    mj_data = mujoco.MjData(mj_model)

    # Set the initial state
    mj_data.qpos[:] = mj_model.qpos0
    mj_data.qvel[:] = np.zeros(mj_model.nv)
    mj_data.mocap_quat[0] = random_quat()

    # Set up the controller
    def make_controller():
        task = CubeRotation()
        controller = PredictiveSampling(
            task,
            num_samples=64,
            num_randomizations=8,
            noise_level=0.5,
        )
        return controller

    # Run the benchmark
    num_rotations, num_drops = run_benchmark(
        make_controller, mj_model, mj_data, run_time)

    print(f"Rotations: {num_rotations}, Drops: {num_drops}")
