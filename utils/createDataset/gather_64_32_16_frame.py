'''
    improve the sample algorithom
    Randomly select boundary points to alleviate the problem of unselectable boundary points concentration.
'''

import numpy as np
# import cv2
import os
import h5py
from scipy.io import loadmat
import random
import json
from multiprocessing import Process, Queue, Lock
import queue
import time

dim = 64

vox_size_1 = 16
vox_size_2 = 32
vox_size_3 = 64

batch_size_1 = 16*16*16
batch_size_2 = 16*16*16
batch_size_3 = 16*16*16*4


global FutureVoxel64
global frame_sample_rate  # the sproportion of the sampling points in object frame
frame_sample_rate = 0.9


'''
#do not use progressive sampling (center2x2x2 -> 4x4x4 -> 6x6x6 ->...)
#if sample non-center points only for inner(1)-voxels,
#the reconstructed model will have railing patterns.
#since all zero-points are centered at cells,
#the model will expand one-points to a one-planes.
'''
def sample_point_in_cube(block,target_value,halfie):
	halfie2 = max(1, halfie*2)
	
	for i in range(100):
		x = np.random.randint(halfie2)
		y = np.random.randint(halfie2)
		z = np.random.randint(halfie2)
		if block[x,y,z]==target_value:
			return x,y,z
	
	if block[halfie,halfie,halfie]==target_value:
		return halfie,halfie,halfie
	
	i=1
	ind = np.unravel_index(np.argmax(block[halfie-i:halfie+i,halfie-i:halfie+i,halfie-i:halfie+i], axis=None), (i*2,i*2,i*2))
	if block[ind[0]+halfie-i,ind[1]+halfie-i,ind[2]+halfie-i]==target_value:
		return ind[0]+halfie-i,ind[1]+halfie-i,ind[2]+halfie-i
	
	for i in range(2,halfie+1):
		six = [(halfie-i,halfie,halfie),(halfie+i-1,halfie,halfie),(halfie,halfie,halfie-i),(halfie,halfie,halfie+i-1),(halfie,halfie-i,halfie),(halfie,halfie+i-1,halfie)]
		for j in range(6):
			if block[six[j]]==target_value:
				return six[j]
		ind = np.unravel_index(np.argmax(block[halfie-i:halfie+i,halfie-i:halfie+i,halfie-i:halfie+i], axis=None), (i*2,i*2,i*2))
		if block[ind[0]+halfie-i,ind[1]+halfie-i,ind[2]+halfie-i]==target_value:
			return ind[0]+halfie-i,ind[1]+halfie-i,ind[2]+halfie-i
	print('hey, error in your code!')
	exit(0)


def get_points_from_vox(q, name_list, lock): 
    name_num = len(name_list)
    for idx in range(name_num):
                  
        #compress model 64 -> 64
        dim_voxel = 64
        with lock:
            voxel_model_64 = FutureVoxel64[name_list[idx]]
            voxel_model_64 = np.flip(np.transpose(voxel_model_64, (0,2,1)),2)
            voxel_model_64 = np.flip(np.transpose(voxel_model_64, (1,0,2)),2)
            voxel_model_64 = np.flip(np.transpose(voxel_model_64, (2,1,0)),2)

        
        multiplier = int(64/dim_voxel)
        halfie = int(multiplier/2)
        #write voxel
        sample_voxels = np.reshape(voxel_model_64, (dim_voxel,dim_voxel,dim_voxel,1))
        
        #sample points near surface
        batch_size = batch_size_3
        
        sample_points = np.zeros([batch_size,3],np.uint8)
        sample_values = np.zeros([batch_size,1],np.uint8)
        batch_size_counter = 0
        voxel_model_temp_flag = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
        sample_idx = [(i, j, k) for i in range(1, dim_voxel - 1) for j in range(1, dim_voxel - 1) for k in range(1, dim_voxel - 1) if np.max(voxel_model_64[i-1:i+2,j-1:j+2,k-1:k+2])!=np.min(voxel_model_64[i-1:i+2,j-1:j+2,k-1:k+2])]
        np.random.shuffle(sample_idx)
        temp_max_range = int(min(len(sample_idx), batch_size * frame_sample_rate))
        for id_ in range(temp_max_range):
            i, j, k = sample_idx[id_]
            si,sj,sk = sample_point_in_cube(voxel_model_64[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_64[i,j,k],halfie)
            sample_points[batch_size_counter,0] = si+i*multiplier
            sample_points[batch_size_counter,1] = sj+j*multiplier
            sample_points[batch_size_counter,2] = sk+k*multiplier
            sample_values[batch_size_counter,0] = voxel_model_64[i,j,k]
            voxel_model_temp_flag[i,j,k] = 1
            batch_size_counter +=1
        # temp_range = list(range(1,dim_voxel-1,4))+list(range(2,dim_voxel-1,4))+list(range(3,dim_voxel-1,4))+list(range(4,dim_voxel-1,4))
        # for j in temp_range:
        #     if (batch_size_counter>=batch_size): break
        #     for i in temp_range:
        #         if (batch_size_counter>=batch_size): break
        #         for k in temp_range:
        #             if (batch_size_counter>=batch_size): break
        #             if (np.max(voxel_model_64[i-1:i+2,j-1:j+2,k-1:k+2])!=np.min(voxel_model_64[i-1:i+2,j-1:j+2,k-1:k+2])):
        #                 si,sj,sk = sample_point_in_cube(voxel_model_64[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_64[i,j,k],halfie)
        #                 sample_points[batch_size_counter,0] = si+i*multiplier
        #                 sample_points[batch_size_counter,1] = sj+j*multiplier
        #                 sample_points[batch_size_counter,2] = sk+k*multiplier
        #                 sample_values[batch_size_counter,0] = voxel_model_64[i,j,k]
        #                 voxel_model_temp_flag[i,j,k] = 1
        #                 batch_size_counter +=1
        if (batch_size_counter>=batch_size):
            print("64-- batch_size exceeded!")
            exceed_64_flag = 1
        else:
            exceed_64_flag = 0
            #fill other slots with random points
            print("64-- batch_size64 unexceeded!")
            while (batch_size_counter<batch_size):
                while True:
                    i = random.randint(0,dim_voxel-1)
                    j = random.randint(0,dim_voxel-1)
                    k = random.randint(0,dim_voxel-1)
                    if voxel_model_temp_flag[i,j,k] != 1: break
                si,sj,sk = sample_point_in_cube(voxel_model_64[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_64[i,j,k],halfie)
                sample_points[batch_size_counter,0] = si+i*multiplier
                sample_points[batch_size_counter,1] = sj+j*multiplier
                sample_points[batch_size_counter,2] = sk+k*multiplier
                sample_values[batch_size_counter,0] = voxel_model_64[i,j,k]
                voxel_model_temp_flag[i,j,k] = 1
                batch_size_counter +=1
            
        sample_points_64 = sample_points
        sample_values_64 = sample_values

        print("64-- batch_size64 unexceeded333!")
        #compress model 64 -> 32
        dim_voxel = 32
        voxel_model_temp = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
        multiplier = int(64/dim_voxel)
        halfie = int(multiplier/2)
        for i in range(dim_voxel):
            for j in range(dim_voxel):
                for k in range(dim_voxel):
                    voxel_model_temp[i,j,k] = np.max(voxel_model_64[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier])
        
        #sample points near surface
        batch_size = batch_size_2
        
        sample_points = np.zeros([batch_size,3],np.uint8)
        sample_values = np.zeros([batch_size,1],np.uint8)
        batch_size_counter = 0
        voxel_model_temp_flag = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
        sample_idx = [(i, j, k) for i in range(1, dim_voxel - 1) for j in range(1, dim_voxel - 1) for k in range(1, dim_voxel - 1) if np.max(voxel_model_64[i-1:i+2,j-1:j+2,k-1:k+2])!=np.min(voxel_model_64[i-1:i+2,j-1:j+2,k-1:k+2])]
        np.random.shuffle(sample_idx)
        temp_max_range = int(min(len(sample_idx), batch_size * frame_sample_rate))
        for id_ in range(temp_max_range):
            i, j, k = sample_idx[id_]
            si,sj,sk = sample_point_in_cube(voxel_model_64[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
            sample_points[batch_size_counter,0] = si+i*multiplier
            sample_points[batch_size_counter,1] = sj+j*multiplier
            sample_points[batch_size_counter,2] = sk+k*multiplier
            sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
            voxel_model_temp_flag[i,j,k] = 1
            batch_size_counter +=1

        # temp_range = list(range(1,dim_voxel-1,4))+list(range(2,dim_voxel-1,4))+list(range(3,dim_voxel-1,4))+list(range(4,dim_voxel-1,4))
        # for j in temp_range:
        #     if (batch_size_counter>=batch_size): break
        #     for i in temp_range:
        #         if (batch_size_counter>=batch_size): break
        #         for k in temp_range:
        #             if (batch_size_counter>=batch_size): break
        #             if (np.max(voxel_model_temp[i-1:i+2,j-1:j+2,k-1:k+2])!=np.min(voxel_model_temp[i-1:i+2,j-1:j+2,k-1:k+2])):
        #                 si,sj,sk = sample_point_in_cube(voxel_model_64[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
        #                 sample_points[batch_size_counter,0] = si+i*multiplier
        #                 sample_points[batch_size_counter,1] = sj+j*multiplier
        #                 sample_points[batch_size_counter,2] = sk+k*multiplier
        #                 sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
        #                 voxel_model_temp_flag[i,j,k] = 1
        #                 batch_size_counter +=1
        if (batch_size_counter>=batch_size):
            print("32-- batch_size exceeded!")
            exceed_32_flag = 1
        else:
            exceed_32_flag = 0
            #fill other slots with random points
            while (batch_size_counter<batch_size):
                while True:
                    i = random.randint(0,dim_voxel-1)
                    j = random.randint(0,dim_voxel-1)
                    k = random.randint(0,dim_voxel-1)
                    if voxel_model_temp_flag[i,j,k] != 1: break
                si,sj,sk = sample_point_in_cube(voxel_model_64[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
                sample_points[batch_size_counter,0] = si+i*multiplier
                sample_points[batch_size_counter,1] = sj+j*multiplier
                sample_points[batch_size_counter,2] = sk+k*multiplier
                sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
                voxel_model_temp_flag[i,j,k] = 1
                batch_size_counter +=1
        
        sample_points_32 = sample_points
        sample_values_32 = sample_values
        
        
        #compress model 64 -> 16
        dim_voxel = 16
        voxel_model_temp = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
        multiplier = int(64/dim_voxel)
        halfie = int(multiplier/2)
        for i in range(dim_voxel):
            for j in range(dim_voxel):
                for k in range(dim_voxel):
                    voxel_model_temp[i,j,k] = np.max(voxel_model_64[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier])
        
        #sample points near surface
        batch_size = batch_size_1
        
        sample_points = np.zeros([batch_size,3],np.uint8)
        sample_values = np.zeros([batch_size,1],np.uint8)
        batch_size_counter = 0
        for i in range(dim_voxel):
            for j in range(dim_voxel):
                for k in range(dim_voxel):
                    si,sj,sk = sample_point_in_cube(voxel_model_64[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
                    sample_points[batch_size_counter,0] = si+i*multiplier
                    sample_points[batch_size_counter,1] = sj+j*multiplier
                    sample_points[batch_size_counter,2] = sk+k*multiplier
                    sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
                    batch_size_counter +=1
        if (batch_size_counter!=batch_size):
            print("batch_size_counter!=batch_size")
        
        sample_points_16 = sample_points
        sample_values_16 = sample_values
        
        q.put([name_list[idx],exceed_64_flag,exceed_32_flag,sample_points_64,sample_values_64,sample_points_32,sample_values_32,sample_points_16,sample_values_16,sample_voxels])



if __name__ == "__main__":


    
    try:
        voxelhdf = h5py.File('/home/RS/CJH/chenjinghuan/3DFutureProject/data/3DFuture/models/ChairVoxelization/3DFutureChair64.h5', 'r')
    except:
        raise Exception("无法打开该H5文件！")
    FutureVoxel64 = voxelhdf["tensor"]

    class_name = "chair64_00000002"
    print(class_name)
    if not os.path.exists(class_name):
        os.makedirs(class_name)
    hdf5_path = class_name+'/'+class_name[:8]+'_vox64.hdf5'

    #obj_list
    fout = open(class_name+'/'+class_name[:8]+'_vox64.txt','w',newline='')

    #record statistics
    fstatistics = open(class_name+'/statistics.txt','w',newline='')
    exceed_32 = 0
    exceed_64 = 0


    name_num = FutureVoxel64.shape[0]
    num_of_process = 12
    list_of_list_of_names = []
    for i in range(num_of_process):
        list_of_names = []
        for j in range(i,name_num,num_of_process):
            list_of_names.append(j)
        list_of_list_of_names.append(list_of_names)

    #map processes
    q = Queue()
    lock = Lock()
    workers = [Process(target=get_points_from_vox, args = (q, list_of_names, lock)) for list_of_names in list_of_list_of_names]

    for p in workers:
        p.start()

    #reduce process
    hdf5_file = h5py.File(hdf5_path, 'w')
    hdf5_file.create_dataset("voxels", [name_num,dim,dim,dim,1], np.uint8)
    hdf5_file.create_dataset("points_16", [name_num,batch_size_1,3], np.uint8)
    hdf5_file.create_dataset("values_16", [name_num,batch_size_1,1], np.uint8)
    hdf5_file.create_dataset("points_32", [name_num,batch_size_2,3], np.uint8)
    hdf5_file.create_dataset("values_32", [name_num,batch_size_2,1], np.uint8)
    hdf5_file.create_dataset("points_64", [name_num,batch_size_3,3], np.uint8)
    hdf5_file.create_dataset("values_64", [name_num,batch_size_3,1], np.uint8)



    while True:
        item_flag = True
        try:
            idx,exceed_64_flag,exceed_32_flag,sample_points_64,sample_values_64,sample_points_32,sample_values_32,sample_points_16,sample_values_16,sample_voxels = q.get(True, 1.0)
        except queue.Empty:
            item_flag = False
        
        if item_flag:
            #process result
            exceed_32+=exceed_32_flag
            exceed_64+=exceed_64_flag
            hdf5_file["points_64"][idx,:,:] = sample_points_64
            hdf5_file["values_64"][idx,:,:] = sample_values_64
            hdf5_file["points_32"][idx,:,:] = sample_points_32
            hdf5_file["values_32"][idx,:,:] = sample_values_32
            hdf5_file["points_16"][idx,:,:] = sample_points_16
            hdf5_file["values_16"][idx,:,:] = sample_values_16
            hdf5_file["voxels"][idx,:,:,:,:] = sample_voxels
        
        allExited = True
        for p in workers:
            if p.exitcode is None:
                allExited = False
                break
        if allExited and q.empty():
            break




    fstatistics.write("total: "+str(name_num)+"\n")
    fstatistics.write("exceed_32: "+str(exceed_32)+"\n")
    fstatistics.write("exceed_32_ratio: "+str(float(exceed_32)/name_num)+"\n")
    fstatistics.write("exceed_64: "+str(exceed_64)+"\n")
    fstatistics.write("exceed_64_ratio: "+str(float(exceed_64)/name_num)+"\n")

    fstatistics.close()
    hdf5_file.close()
    print("finished")

