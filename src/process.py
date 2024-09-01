import multiprocessing
import multiprocessing.managers
from multiprocessing.synchronize import Event
from typing import Callable

def _main_process(
    init_func: Callable,
    call_func: Callable,
    start_sync_event: Event,
    exit_event: Event,
    status_flags,
    process_index: int
    ):
    
    init_func()
    status_flags[process_index] = True
    start_sync_event.wait()
    print(f"Start {process_index} process")
    while not exit_event.is_set():
        status_flags[process_index] = False
        call_func()
        status_flags[process_index] = True
        

class Process(multiprocessing.Process):
    def __init__(self):
        super().__init__()
    
    def _init_func(self):
        raise NotImplementedError()
    
    def _loop_func(self):
        raise NotImplementedError()
    
    def _end_func(self):
        pass
    
    def start(self, 
              start_sync_event: Event,
              exit_event: Event,
              status_flags,
              process_index: int
        ):
        
        self.start_sync_event = start_sync_event
        self.exit_event = exit_event
        self.status_flags = status_flags
        self.process_index = process_index
        super().start()
    
    
    def run(self):
        self.status_flags[self.process_index] = False
        try:
            self._init_func()
        finally:
            self.status_flags[self.process_index] = True
        self.start_sync_event.wait()
        print(f"Start {self.process_index} process")
        
        try:
            while not self.exit_event.is_set():
                self._loop_func()
        finally:
            self._end_func()
            
        
        