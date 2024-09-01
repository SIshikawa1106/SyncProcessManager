import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import numpy as np
import time

im_list = None
im_num = 100

cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
last_index = 0

def get_past_images(im_list, last_index, im_num):
    """
    過去100枚の画像データを順に格納された配列を取得する関数

    Args:
        im_list: 画像データを格納したNumPy配列
        last_index: 最新の画像が格納されているインデックス
        im_num: 全体の画像枚数

    Returns:
        過去100枚の画像データを順番に格納したNumPy配列
    """

    # インデックスの計算
    start_index = (last_index - im_num) % im_num
    end_index = last_index

    # 配列の形状を取得
    shape = im_list.shape[1:]

    # 出力配列を作成
    result = np.zeros((im_num, *shape), dtype=im_list.dtype)

    # データをコピー
    result[:end_index] = im_list[start_index:end_index]
    result[end_index:] = im_list[:start_index]

    return result


while True:
    
    ret1, im1 = cap1.read()
    ret2, im2 = cap2.read()
    
    if ret1 and ret2:

        if im_list is None:
            im_list = np.zeros((im_num, *im1.shape), dtype=np.uint8)

        start = time.time()
        if last_index==im_num:
            #im_list[:-1] = im_list[1:]
            #im_list[-1] = im1
            # 循環保存
            last_index = last_index%im_num
            im_list[last_index] = im1
        else:
            im_list[last_index] = im1
            last_index += 1
        end = time.time()
        print(f"elapsed time = {end-start}[sec],{last_index}")
        
        start = time.time()
        copy_im_list = np.roll(im_list, -last_index)
        end = time.time()
        print(f"roll elapsed time = {end-start}[sec],{last_index}")
            
        # 画像を水平方向に連結
        horizontal_img = np.hstack((im1, im2))

        cv2.imshow('Combined Image', horizontal_img)

        if cv2.waitKey(1) == ord('q'):
            break

# カメラの解放
cap1.release()
cap2.release()
cv2.destroyAllWindows()