import os
import sys
import time
import numpy as np
import cv2

currdir = os.path.dirname(__file__)
scriptdir = os.path.abspath(os.path.join(currdir, "../src"))

if scriptdir not in sys.path:
    sys.path.append(scriptdir)
    
import process
import manager
import share_data
import threading
    
    
class CameraProcess(process.Process):
    def __init__(self, camera_index: int, memory_info, verbose=False, log_dir=None):
        super().__init__()
        self.camera_index = camera_index
        self.memory_info = memory_info
        self.verbose = verbose
        if log_dir:
            self.log_dir = os.path.join(log_dir, f"{self.camera_index}")
        else:
            self.log_dir = None
        
    def _init_func(self):
        os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
        self.cap = cv2.VideoCapture(self.camera_index)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.memory_info[0]['shape'][0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.memory_info[0]['shape'][1])
                
        print(f"cap = {self.cap}")
        self.memory = share_data.SharedMemoryDataWithTime(self.memory_info[0], self.memory_info[1])

    def _loop_func(self):
        
        try:
            ret, frame = self.cap.read()
            if ret:
                counter = self.memory.add_frame(frame)
            
            if self.verbose:
                cv2.imshow(f'camera {self.camera_index}', frame)
                cv2.waitKey(1)
                
            if self.log_dir:
                if not os.path.isdir(self.log_dir):
                    os.makedirs(self.log_dir)
                
                filename = os.path.join(self.log_dir, f"{counter}.png")
                threading.Thread(target=lambda f=frame, fn=filename: cv2.imwrite(fn, f), daemon=True).start()
        except:
            pass
        

if __name__=="__main__":
    
    memory_info = share_data.make_info(100, (480,640,3), np.uint8)
    memory = share_data.SharedMemoryDataWithTime(memory_info, None, True)
    memory_info = memory.memory_info
    
    processes = [
        CameraProcess(0, memory_info, verbose=True)
    ]
    
    mgr = manager.SyncMainProcess()
    mgr.start(processes)
    
    while not mgr.start_sync_event.is_set():
        print("Wait ...")
        time.sleep(1)
    time.sleep(10)
    mgr.join()
    
    data = memory.get_data()
    print(data)