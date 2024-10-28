import time
from multiprocessing import Process, shared_memory, Lock, Event
import numpy as np

import mujoco
import mujoco.viewer


class SharedMemoryNumpyArray:
    """Helper class to store a numpy array in shared memory."""
    def __init__(self, arr: np.ndarray):
        """Create a shared memory numpy array.

        Args:
            arr: The numpy array to store in shared memory. Size and dtype must
                 be fixed.
        """
        self.shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        self.shared_arr = np.ndarray(
            arr.shape, dtype=arr.dtype, buffer=self.shm.buf)
        self.shared_arr[:] = arr[:]
        self.lock = Lock()

    def __getitem__(self, key):
        """Get an item from the shared array."""
        return self.shared_arr[key]
    
    def __setitem__(self, key, value):
        """Set an item in the shared array."""
        with self.lock:
            self.shared_arr[key] = value

    def __str__(self):
        """Return the string representation of the shared array."""
        return str(self.shared_arr)

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
        finished: Shared flag for stopping the simulation.
    """
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
    finished: Event,
):
    """Run the controller loop.

    Args:
        qpos: Where we read the qpos data from shared memory.
        qvel: Where we read the qvel data from shared memory.
        mocap_pos: Where we read the mocap_pos data from shared memory.
        mocap_quat: Where we read the mocap_quat data from shared memory.
        ctrl: Where we write the control data into shared memory.
        finished: Shared flag for stopping the simulation.
    """
    # TODO: controller setup, jitting, etc.
    while not finished.is_set():
        ctrl[:] = np.random.randn(ctrl.shape[0])
        time.sleep(1.0)


if __name__=="__main__":
    run_time = 10.0  # Total sim time, in seconds

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
    finished = Event()
    
    # Set up the simulator and controller processes
    sim = Process(target=simulator, args=(
        shm_qpos, shm_qvel, shm_mocap_pos, shm_mocap_quat, shm_ctrl,
        mj_model, mj_data, run_time, finished))
    control = Process(target=controller, args=(
        shm_qpos, shm_qvel, shm_mocap_pos, shm_mocap_quat, shm_ctrl,
        finished))

    # Run the simulation and controller in parallel 
    sim.start()
    control.start()
    sim.join()
    control.join()