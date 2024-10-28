import time
from multiprocessing import Process, shared_memory
import numpy as np
from dataclasses import dataclass

@dataclass
class SharedMemoryInfo:
    """Helper class to store a numpy array in shared memory."""
    name: str
    shape: tuple
    dtype: type


def sender(info: SharedMemoryInfo):
    # Attach to the shared memory
    shm = shared_memory.SharedMemory(name=info.name)
    arr = np.ndarray(info.shape, dtype=info.dtype, buffer=shm.buf)

    # Write to the shared memory
    for i in range(100):
        arr[0] = i
        time.sleep(0.01)

def reciever(info: SharedMemoryInfo):
    # Attach to the shared memory
    shm = shared_memory.SharedMemory(name=info.name)
    arr = np.ndarray(info.shape, dtype=info.dtype, buffer=shm.buf)

    for i in range(10):
        print(arr)
        time.sleep(0.1)


if __name__=="__main__":
    # Create a shared memory numpy array
    arr = np.arange(10, dtype=np.float64)
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    shared_arr[:] = arr[:]
    info = SharedMemoryInfo(shm.name, arr.shape, arr.dtype)

    # Sender and reciever processes
    p1 = Process(target=sender, args=(info,))
    p2 = Process(target=reciever, args=(info,))

    # Start the processes
    p1.start()
    p2.start()

    # Cleanup
    p1.join()
    p2.join()
    shm.close()
    shm.unlink()