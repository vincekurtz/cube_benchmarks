import time
from multiprocessing import Process, shared_memory
import numpy as np

class SharedMemoryNumpyArray:
    """Helper class to store a numpy array in shared memory."""
    def __init__(self, arr: np.ndarray):
        """Create a shared memory numpy array.

        Args:
            arr: The numpy array to store in shared memory. Size and dtype must
                 be fixed.

        Warning: The shared memory is writable by all processes that have
                 access to the shared memory object. Multiple processes should
                 not write to the shared memory at the same time.
        """
        self.shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        self.shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=self.shm.buf)
        self.shared_arr[:] = arr[:]

    def __del__(self):
        """Clean up the shared memory on deletion."""
        self.shm.close()
        self.shm.unlink()


def sender(info: SharedMemoryNumpyArray):
    # Attach to the shared memory
    arr = info.shared_arr

    # Write to the shared memory
    for i in range(10):
        arr[0] = i
        time.sleep(0.1)

def reciever(info: SharedMemoryNumpyArray):
    # Attach to the shared memory
    arr = info.shared_arr

    for i in range(100):
        print(arr)
        time.sleep(0.01)


if __name__=="__main__":
    # Create a shared memory numpy array
    arr = np.arange(10, dtype=np.float64)
    shm_arr = SharedMemoryNumpyArray(arr)

    # Sender and reciever processes
    p1 = Process(target=sender, args=(shm_arr,))
    p2 = Process(target=reciever, args=(shm_arr,))

    # Start the processes
    p1.start()
    p2.start()

    # Cleanup
    p1.join()
    p2.join()