import time
from multiprocessing import Process, shared_memory, Lock
import numpy as np

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


def sender(arr: SharedMemoryNumpyArray):
    for i in range(100):
        arr[0] = i
        time.sleep(0.01)

def reciever(arr: SharedMemoryNumpyArray):
    for i in range(10):
        print(arr)
        time.sleep(0.1)


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