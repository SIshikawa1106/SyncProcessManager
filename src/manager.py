import multiprocessing
import process
import time
    
class SyncMainProcess(process.Process):
    def __init__(self):
        super().__init__()
        self.exit_event = multiprocessing.Event()
        self.start_counter = multiprocessing.Value('i', -1)
    
    def _init_func(self):
        while not self.exit_event.is_set():
            try:
                if all(self.status_flags[:-1]):
                    # call function
                    self.start_counter.value = time.perf_counter_ns()
                    self.start_sync_event.set()
                    break
            except:
                pass
        print(f"call init {self.process_index}")
        
    def _loop_func(self):
        pass
    
    
    def start(self, processes: list):
        manager = multiprocessing.Manager()
        
        status_flags = manager.list([False]*(len(processes)+1))
        start_sync_event = multiprocessing.Event()
        
        for idx, p in enumerate(processes):
            assert isinstance(p, process.Process)
            p.start(start_sync_event, self.exit_event, status_flags, idx)
        super().start(start_sync_event, self.exit_event, status_flags, len(processes))
        self.processes = processes
        
    def stop(self):
        self.exit_event.set()
        if self.processes is not None:
            for p in self.processes:
                assert isinstance(p, process.Process)
                p.join()
    
    def join(self):
        self.stop()
        super().join()