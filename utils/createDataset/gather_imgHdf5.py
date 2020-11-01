import json
import cv2
import numpy as np
import os
import h5py

if __name__ =="__main__":
    name_num = 574
    num_view = 12
    view_size = 137
    hdf5_file = h5py.File("./Chairimg.hdf5", 'w')
    hdf5_file.create_dataset("pixels", [name_num,num_view,view_size,view_size], np.uint8, compression=9)
    with open("/home/RS/CJH/chenjinghuan/3DFutureProject/data/3DFuture/ChairVoxelization/3DFutureChairfileName.txt", "r") as f:
        tableFileName = f.readlines()
    imagePath = "/home/RS/CJH/chenjinghuan/3DFutureProject/utils/chairImages"
    imageList = os.listdir(imagePath)
    for idx, imgidx in enumerate(tableFileName):
        t = 0
        for j in imageList:
            if -1 != j.find(imgidx[0:7]):
                img = cv2.imread(os.path.join(imagePath, j), cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img, (view_size, view_size))
                imgo = img[:,:,:3]
                imgo = cv2.cvtColor(imgo, cv2.COLOR_BGR2GRAY)
                # imga = (img[:,:,3])/255.0
                # img = imgo*imga + 255*(1-imga)
                img = np.round(imgo).astype(np.uint8)
                hdf5_file["pixels"][idx,t,:,:] = img
                t = t + 1
                print("%s, t = %d"%(j, t))
    hdf5_file.close()
        


    # for idx in range(name_num):
    #     print(idx)



