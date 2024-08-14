import os
import sys
import time
from functools import partial
import numpy as np

currdir = os.path.dirname(__file__)
scriptdir = os.path.abspath(os.path.join(currdir, "../src"))

if scriptdir not in sys.path:
    sys.path.append(scriptdir)
    
import process
import manager
import share_data
    

class Test(process.Process):
    def __init__(self, index):
        self.index = index
        super().__init__()
    
    def _init_func(self):
        print(f"call init {self.index}")
        
    def _loop_func(self):
        print(f"call {self.index} in loop")
        time.sleep(1)
        
class TestWithSharedMemoryData(process.Process):
    def __init__(self, index, memory_info):
        super().__init__()
        self.memory_info = memory_info
        self.index = index
        self.memory = None
        
    def _init_func(self):
        print(f"call init {self.index}")
        self.memory = share_data.SharedMemoeryDataWithTime(self.memory_info[0], self.memory_info[1])
        
    def _loop_func(self):
        print(f"call {self.index} in loop")
        time.sleep(1)
        data = np.array((np.random.rand(),))
        self.memory.add_frame(data)
    
if __name__=="__main__":
    
    memory_info = share_data.make_info(10, (1,), np.float64)
    memory = share_data.SharedMemoeryDataWithTime(memory_info, None, True)
    
    processes = [
        Test(0),
        Test(1),
        TestWithSharedMemoryData(2, memory.memory_info)
    ]
    
    mgr = manager.SyncMainProcess()
    
    mgr.start(processes)
    time.sleep(11)
    mgr.join()
    
    data = memory.get_data()
    print(data)