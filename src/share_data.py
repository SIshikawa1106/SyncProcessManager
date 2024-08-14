import numpy as np
import time
from multiprocessing import shared_memory, Manager

def make_info(list_size, shape, dtype):
    return {
        'list_size': list_size,
        'shape': shape,
        'dtype': dtype
    }

class SharedMemoeryData:
    def __init__(self, memory_info: dict, create: bool=False):
        
        assert isinstance(memory_info, dict)
        assert 'shape' in memory_info
        assert 'list_size' in memory_info
        assert 'dtype' in memory_info
        
        self.memory_info = memory_info
        self.shape = memory_info['shape']
        self.dtype = memory_info['dtype']
        self.list_size = memory_info['list_size']
        
        if create:
            memory_size = int(np.prod(self.shape) * self.list_size * np.dtype(self.dtype).itemsize)
            self.shared_mem = shared_memory.SharedMemory(
                create=create, size=memory_size
            )
            self.name = self.shared_mem.name
            mgr = Manager()
            self.condition = mgr.Condition()
            self.current_frame_index = mgr.Value('i', 0)
            self.has_full_data = mgr.Value('b', False)
            self.memory_info['name'] = self.shared_mem.name
            self.memory_info['condition'] = self.condition
            self.memory_info['current_frame_index'] = self.current_frame_index
            self.memory_info['has_full_data'] = self.has_full_data
        else:
            assert 'name' in memory_info
            assert 'condition' in memory_info
            assert 'current_frame_index' in memory_info
            assert 'has_full_data' in memory_info
            self.name = memory_info['name']
            self.condition = memory_info['condition']
            self.current_frame_index = self.memory_info['current_frame_index']
            self.has_full_data = self.memory_info['has_full_data']
            self.shared_mem = shared_memory.SharedMemory(name=self.name)        
        
        self.array = np.ndarray((self.list_size, *self.shape), dtype=self.dtype, buffer=self.shared_mem.buf)
        
    def close(self):
        self.shared_mem.close()
        
    def unlink(self):
        self.shared_mem.unlink()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        self.unlink()
        
    def _add_frame(self, data: np.ndarray):
        assert data.shape == self.shape
        assert data.dtype == self.dtype
        self.array[self.current_frame_index.value] = data
        self.current_frame_index.value += 1
        if self.current_frame_index == self.list_size:
            self.has_full_data.value = True
            self.current_frame_index.value = 0
        
    def add_frame(self, data: np.ndarray):
        with self.condition:
            self._add_frame(data)
            
    def _get_frame(self, frame_index: int, start_index: int):
        if not self.has_full_data.value:
            if frame_index<start_index:
                return None
        index = (start_index + frame_index) % self.list_size
        data = self.array[index].copy()
        return data
    
    def get_frame(self, frame_index):
        # NOTE: Ensure that only the index being added is locked for exclusive access
        with self.condition:
            return self._get_frame(frame_index, self.current_frame_index.value)
    
class SharedMemoeryDataWithTime:
    def __init__(self, memory_info: dict, time_memory_info: dict=None, create: bool=False):
        self.memory = SharedMemoeryData(memory_info, create)
        
        if create:
            time_memory_info = memory_info.copy()
            time_memory_info['dtype'] = np.int64
            time_memory_info['shape'] = (1,)
            self.time_memory = SharedMemoeryData(time_memory_info, create)
        else:
            assert time_memory_info is not None
            self.time_memory = SharedMemoeryData(time_memory_info, create)
        
    def close(self):
        self.memory.close()
        
    def unlink(self):
        self.memory.unlink()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        self.unlink()
        
    
    def add_frame(self, data: np.ndarray):
        with self.memory.condition:
            self.memory._add_frame(data)
            time_data = np.array((time.perf_counter_ns(),))
            self.time_memory._add_frame(time_data)
            assert self.memory.current_frame_index.value == self.time_memory.current_frame_index.value
        
    def get_frame(self, frame_index):
        with self.memory.condition:
            start_index = self.memory.current_frame_index.value
            data = self.memory._get_frame(frame_index, start_index)
            time_data = self.memory._get_frame(frame_index, start_index)
            return data, time_data[0]
    
    def get_data(self):
        start = None
        
        if self.memory.has_full_data.value:
            size = self.memory.list_size
        else:
            size = self.memory.current_frame_index.value - 1;
            if size<0:
                return None, None
        
        raw_data_list = np.zeros((size, *self.memory.shape), dtype=self.memory.dtype)
        time_data_list = np.zeros((size, 1), dtype=self.time_memory.dtype)
        
        for index in range(size):
            with self.memory.condition:
                if index == 0:
                    if self.memory.has_full_data.value:
                        start = self.memory.current_frame_index.value
                    else:
                        start = 0
                raw_data_list[index] = self.memory._get_frame(index, start)
                time_data_list[index] = self.time_memory._get_frame(index, start)
                    
        return raw_data_list, time_data_list                    
        
    @property
    def memory_info(self):
        return self.memory.memory_info, self.time_memory.memory_info
        
        
    