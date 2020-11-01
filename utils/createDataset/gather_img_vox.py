import h5py
import numpy as np


if __name__ == "__main__":

    num_view = 12
    view_size = 137
    vox_size = 64
    vox_size_1 = 16
    vox_size_2 = 32
    vox_size_3 = 64
    batch_size_1 = 16*16*16
    batch_size_2 = 16*16*16
    batch_size_3 = 16*16*16*4
    num_shapes = 574
    
    
    

    hdf5_file = h5py.File("3DFUtureChair_img_vox.hdf5", 'w')

    hdf5_file.create_dataset("pixels", [num_shapes,num_view,view_size,view_size], np.uint8, compression=9)
    hdf5_file.create_dataset("voxels", [num_shapes,vox_size,vox_size,vox_size,1], np.uint8, compression=9)
    hdf5_file.create_dataset("points_16", [num_shapes,batch_size_1,3], np.uint8, compression=9)
    hdf5_file.create_dataset("values_16", [num_shapes,batch_size_1,1], np.uint8, compression=9)
    hdf5_file.create_dataset("points_32", [num_shapes,batch_size_2,3], np.uint8, compression=9)
    hdf5_file.create_dataset("values_32", [num_shapes,batch_size_2,1], np.uint8, compression=9)
    hdf5_file.create_dataset("points_64", [num_shapes,batch_size_3,3], np.uint8, compression=9)
    hdf5_file.create_dataset("values_64", [num_shapes,batch_size_3,1], np.uint8, compression=9)


    voxel_hdf5_dir1 = "/home/RS/CJH/chenjinghuan/3DFutureProject/Chair256_00000000/Chari256_vox256.hdf5"
    voxel_hdf5_file1 = h5py.File(voxel_hdf5_dir1, 'r')
    voxel_hdf5_voxels = voxel_hdf5_file1['voxels'][:]
    voxel_hdf5_points_16 = voxel_hdf5_file1['points_16'][:]
    voxel_hdf5_values_16 = voxel_hdf5_file1['values_16'][:]
    voxel_hdf5_points_32 = voxel_hdf5_file1['points_32'][:]
    voxel_hdf5_values_32 = voxel_hdf5_file1['values_32'][:]
    voxel_hdf5_points_64 = voxel_hdf5_file1['points_64'][:]
    voxel_hdf5_values_64 = voxel_hdf5_file1['values_64'][:]
    voxel_hdf5_file1.close()

    image_hdf5_dir = "/home/RS/CJH/chenjinghuan/3DFutureProject/Chairimg.hdf5"
    image_hdf5_file = h5py.File(image_hdf5_dir, 'r')
    image_hdf5_pixels = image_hdf5_file['pixels'][:]
    image_hdf5_file.close()

    

    hdf5_file["pixels"][0:num_shapes] = image_hdf5_pixels[0:num_shapes]
    hdf5_file["voxels"][0:num_shapes] = voxel_hdf5_voxels[0:num_shapes]
    hdf5_file["points_16"][0:num_shapes] = voxel_hdf5_points_16[0:num_shapes]
    hdf5_file["values_16"][0:num_shapes] = voxel_hdf5_values_16[0:num_shapes]
    hdf5_file["points_32"][0:num_shapes] = voxel_hdf5_points_32[0:num_shapes]
    hdf5_file["values_32"][0:num_shapes] = voxel_hdf5_values_32[0:num_shapes]
    hdf5_file["points_64"][0:num_shapes] = voxel_hdf5_points_64[0:num_shapes]
    hdf5_file["values_64"][0:num_shapes] = voxel_hdf5_values_64[0:num_shapes]

    hdf5_file.close()