from os import confstr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch.autograd import Variable

import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import h5py
import numpy as np
import time
import mcubes
from utils.plyUtils import *
from utils.config import Config
from models.IMBASE import *




class BAE_Network(nn.Module):
    def __init__(self, ef_dim, z_dim, point_dim, gf_dim, gf_split):
        super(BAE_Network, self).__init__()
        self.ef_dim = ef_dim
        self.z_dim = z_dim
        self.point_dim = point_dim
        self.gf_dim = gf_dim
        self.gf_split = gf_split
        self.encoder = encoder(self.ef_dim, self.z_dim)
        self.generator = generator(self.z_dim, self.point_dim, self.gf_dim, self.gf_split)

    def forward(self, inputs, z_vector, point_coord, is_training=False):
        part_, net_ = None, None
        if is_training:
            z_vector = self.encoder(inputs, is_training=is_training)
            part_, net_ = self.generator(point_coord, z_vector, is_training=is_training)
        else:
            if inputs is not None:
                z_vector = self.encoder(inputs, is_training=is_training)
            if z_vector is not None and point_coord is not None:
                part_, net_ = self.generator(point_coord, z_vector, is_training=is_training)
        return z_vector, part_, net_




class IMSEG(IM_BASE):
    def __init__(self, config):
        super(IMSEG, self).__init__(config)

        self.data_dir = config.data_dir

        if config.supervised:
            #TODO
            pass
        else:
            self.gf_split = 8

        self.test_size = config.test_size
        self.batch_size = self.test_size * self.test_size * self.test_size #do not change

    def _load_dataset(self):

        # allset_name = self.dataset_name[:8] + "_vox"
        allset_name = self.config.dataset_name + "_test.hdf5"
        if self.config.is_training:
            allset_name = self.config.dataset_name + "_train.hdf5"
        data_hdf5_name = self.config.data_dir + '/' + allset_name
        if os.path.exists(data_hdf5_name):
            self.data_dict = h5py.File(data_hdf5_name, 'r')
            data_points_int = self.data_dict['points_' + str(self.config.frame_grid_size)][:].astype(np.float32)
            self.data_points = (data_points_int + 0.5) / self.config.oringal_voxel_size - 0.5   # importance
            self.data_values = self.data_dict['values_' + str(self.config.frame_grid_size)][:].astype(np.float32)
            self.data_voxels = self.data_dict['voxels'][:]
            self.data_voxels = np.reshape(self.data_voxels, [-1, 1, self.config.input_size, self.config.input_size, self.config.input_size])
            # if self.points_per_shape != self.data_points.shape[1]:
            #     print("error: points_per_shape!=data_points.shape")
            #     exit(0)
            # if self.input_size != self.data_voxels.shape[1]:
            #     print("error: input_size!=data_voxels.shape")
            #     exit(0)
        else:
            print("error: cannot load " + data_hdf5_name)
            exit(0)

    def get_coords_for_test(self):

        dimc = self.config.cell_grid_size
        dimf = self.frame_grid_size
        self.cell_x = np.zeros([dimc, dimc, dimc], np.int32)
        self.cell_y = np.zeros([dimc, dimc, dimc], np.int32)
        self.cell_z = np.zeros([dimc, dimc, dimc], np.int32)
        self.cell_coords = np.zeros([dimf, dimf, dimf, dimc, dimc, dimc, 3], np.float32)
        self.frame_coords = np.zeros([dimf, dimf, dimf, 3], np.float32)
        self.frame_x = np.zeros([dimf, dimf, dimf], np.int32)
        self.frame_y = np.zeros([dimf, dimf, dimf], np.int32)
        self.frame_z = np.zeros([dimf, dimf, dimf], np.int32)
        for i in range(dimc):
            for j in range(dimc):
                for k in range(dimc):
                    self.cell_x[i, j, k] = i
                    self.cell_y[i, j, k] = j
                    self.cell_z[i, j, k] = k
        for i in range(dimf):
            for j in range(dimf):
                for k in range(dimf):
                    self.cell_coords[i, j, k, :, :, :, 0] = self.cell_x + i * dimc
                    self.cell_coords[i, j, k, :, :, :, 1] = self.cell_y + j * dimc
                    self.cell_coords[i, j, k, :, :, :, 2] = self.cell_z + k * dimc
                    self.frame_coords[i, j, k, 0] = i
                    self.frame_coords[i, j, k, 1] = j
                    self.frame_coords[i, j, k, 2] = k
                    self.frame_x[i, j, k] = i
                    self.frame_y[i, j, k] = j
                    self.frame_z[i, j, k] = k
        self.cell_coords = (self.cell_coords.astype(np.float32) + 0.5) / self.real_size - 0.5
        self.cell_coords = np.reshape(self.cell_coords, [dimf, dimf, dimf, dimc * dimc * dimc, 3])
        self.cell_x = np.reshape(self.cell_x, [dimc * dimc * dimc])
        self.cell_y = np.reshape(self.cell_y, [dimc * dimc * dimc])
        self.cell_z = np.reshape(self.cell_z, [dimc * dimc * dimc])
        self.frame_x = np.reshape(self.frame_x, [dimf * dimf * dimf])
        self.frame_y = np.reshape(self.frame_y, [dimf * dimf * dimf])
        self.frame_z = np.reshape(self.frame_z, [dimf * dimf * dimf])
        self.frame_coords = (self.frame_coords.astype(np.float32) + 0.5) / dimf - 0.5
        self.frame_coords = np.reshape(self.frame_coords, [dimf * dimf * dimf, 3])
        # self.coords = self.coords.to(self.device)

    def network_loss(self, G, point_Value):
        if self.config.L1reg:
            # return 0.000001 * torch.sum(torch.abs(self.bae_model.generator.part_layer_weights - 1))
            return torch.mean((G-point_Value) **2) + 0.000001 * torch.sum(torch.abs(self.bae_model.generator.part_layer_weights - 1))
        return torch.mean((G-point_Value) **2)

    def build_model(self):
        # a = BAE_Network(**self.config)
        self.bae_model = BAE_Network(self.config.ef_dim, self.config.z_dim, self.config.point_dim, self.config.gf_dim, self.config.gf_split)
        self.bae_model.to(self.device)
        self.optimizer = torch.optim.Adam(self.bae_model.parameters(), lr=self.config.learning_rate, betas=(self.config.beta1, 0.999))

    # def loadCheckpoint(self):
    #     checkpoint_txt = os.path.join(self.checkpoint_dir, "checkpoint")
    #     if os.path.exists(checkpoint_txt):
    #         fin = open(checkpoint_txt)
    #         model_dir = fin.readline().strip()
    #         fin.close()
    #         self.bae_model.load_state_dict(torch.load(model_dir))
    #         print("[*] Load SUCCESS")
    #     else:
    #         print("[!] Load FAILED...")



    def get_Z_vector(self, config):

        self._load_dataset()
        self.get_devices()
        self.build_model()
        self.loadCheckpoint()
        self.bae_model.eval()

        hdf5_path = config.checkpoint + '/' + self.model_dir + '_train_z.hdf5'
        shape_num = len(self.data_voxels)
        hdf5_file = h5py.File(hdf5_path, mode='w')
        hdf5_file.create_dataset("zs", [shape_num, config.z_dim], np.float32)

        self.bae_model.eval()
        print(shape_num)
        for t in range(shape_num):
            batch_voxels = self.data_voxels[t:t+1].astype(np.float32)
            batch_voxels = torch.from_numpy(batch_voxels)
            batch_voxels = batch_voxels.to(self.device)
            out_z, _, _ = self.bae_model(batch_voxels, None, None, is_training=False)
            hdf5_file["zs"][t:t+1,:] = out_z.detach().cpu().numpy()
            print("[z]:",t)
        hdf5_file.close()


    def train(self, config):

        sys.stdout = open(os.path.join("./", self.config.dataset_name + str(self.config.frame_grid_size) + "log.txt"), 'w')

        self._load_dataset()
        self.get_devices()
        self.get_coords_for_training()
        self.build_model()
        self.loadCheckpoint()


        print("PID = %d" % os.getpid())

        shape_num = len(self.data_voxels)
        batch_index_list = np.arange(shape_num)

        print("\n\n----------net summary----------")
        print("training samples   ", shape_num)
        print("-------------------------------\n\n")
        sys.stdout.flush()
        start_time = time.time()
        assert config.epoch == 0 or config.iteration == 0

        batch_idxs = len(self.data_points)

        training_epoch = config.epoch + int(config.iteration / batch_idxs)

        print("batch_idxs = {0}".format(batch_idxs))
        train_batchnum = int(batch_idxs / self.config.batch_size_train)

        # point_batch_num = int(self.load_point_batch_size / self.point_batch_size)
        sche_lambda = lambda epoch: max(0.95 ** epoch, 0.01)
        scheduler_1ambda = StepLR(self.optimizer, step_size=40, gamma=0.98)
        for epoch in range(0, training_epoch + 1):
            self.bae_model.train()
            np.random.shuffle(batch_index_list)
            avg_loss_sp = 0
            avg_num = 0
            # self.test_ae(config, "train_" + str(self.frame_grid_size) + "_" + str(epoch))
            for idx in range(train_batchnum):
                dxb = batch_index_list[idx * self.config.batch_size_train : (idx+1) * self.config.batch_size_train]


                batch_voxels = self.data_voxels[dxb].astype(np.float32)
                point_coord = self.data_points[dxb]
                point_value = self.data_values[dxb]

                batch_voxels = torch.from_numpy(batch_voxels)
                point_coord = torch.from_numpy(point_coord)
                point_value = torch.from_numpy(point_value)

                batch_voxels = batch_voxels.to(self.device)
                point_coord = point_coord.to(self.device)
                point_value = point_value.to(self.device)

                self.bae_model.zero_grad()
                _, part_out, net_out = self.bae_model(batch_voxels, None, point_coord, is_training=True)
                # part_out, point_out = net_out # part_out [n, 8], point_out [n, 1]
                # temp = point_out[0]
                errSP = self.network_loss(net_out, point_value)

                # errSP = errSP.requires_grad(True)
                errSP = errSP.requires_grad_()
                errSP.backward()
                self.optimizer.step()
                avg_loss_sp += errSP.item()
                avg_num += 1
                # print(str(self.frame_grid_size)+" lr: %.6f Epoch: [%2d/%2d] time: %4.4f, loss_sp: %.6f" % (self.optimizer.param_groups[0]['lr'], epoch, training_epoch, time.time() - start_time, avg_loss_sp/avg_num))

            scheduler_1ambda.step()
            print(str(self.config.frame_grid_size)+" lr: %.6f Epoch: [%2d/%2d] time: %4.4f, loss_sp: %.6f" % (self.optimizer.param_groups[0]['lr'], epoch, training_epoch, time.time() - start_time, avg_loss_sp/avg_num))
            if epoch % 10 == 9:
                self.test_ae(config, "train_" + str(self.config.frame_grid_size) + "_" + str(epoch))
            if epoch % 20 == 9:
                save_dir = os.path.join(self.config.checkpoint, self.config.checkpoint_name + str(self.config.frame_grid_size) + "-" + str(epoch) + ".pth")
                self.save(save_dir)
            sys.stdout.flush()

    def test_ae_all(self):

        self._load_dataset()
        self.get_devices()
        self.get_coords_for_training()
        self.build_model()
        self.loadCheckpoint()
        self.bae_model.eval()

        shape_num = len(self.data_voxels)
        print("testing samples   ", shape_num)
        multiplier = int(self.config.frame_grid_size / self.config.test_size)
        multiplier2 = multiplier * multiplier

        for t in range(shape_num):
            model_float = np.zeros([self.config.frame_grid_size + 2, self.config.frame_grid_size + 2, self.config.frame_grid_size + 2], np.float32)
            batch_voxels = self.data_voxels[t:t + 1].astype(np.float32)
            sq_batch_voxel = np.squeeze(batch_voxels)
            vertices_gt, triangles_gt = mcubes.marching_cubes(sq_batch_voxel, self.config.sampling_threshold)
            write_ply_triangle(self.config.sample_dir + "/" + str(t) + "gt_vox.ply", vertices_gt, triangles_gt)
            batch_voxels = torch.from_numpy(batch_voxels)
            batch_voxels = batch_voxels.to(self.device)
            z_vector, _, _ = self.bae_model(batch_voxels, None, None, is_training=False)
            for i in range(multiplier):
                for j in range(multiplier):
                    for k in range(multiplier):
                        minib = i * multiplier2 + j * multiplier + k
                        point_coord = self.coords[minib:minib + 1]
                        _, _, net_out = self.bae_model(None, z_vector, point_coord, is_training=False)
                        model_float[self.aux_x + i + 1, self.aux_y + j + 1, self.aux_z + k + 1] = np.reshape(
                            net_out.detach().cpu().numpy(), [self.test_size, self.test_size, self.test_size])
            vertices, triangles = mcubes.marching_cubes(model_float, self.config.sampling_threshold)
            vertices = (vertices.astype(np.float32) - 0.5) / self.config.frame_grid_size - 0.5
            # output ply sum
            write_ply_triangle(self.config.sample_dir+"/"+str(t)+"_vox.ply", vertices, triangles)
            print("[sample]")

    def test_ae_withcolor(self):

        self._load_dataset()
        self.get_devices()
        self.get_coords_for_training()
        self.build_model()
        self.loadCheckpoint()

        shape_num = len(self.data_voxels)
        print("testing samples   ", shape_num)
        multiplier = int(self.config.frame_grid_size / self.test_size)
        multiplier2 = multiplier * multiplier
        self.bae_model.eval()

        color_list = ["255 0 0","0 255 0","0 0 255","255 255 0","255 0 255","0 255 255","180 180 180", "100 100 100", "255 128 128","128 255 128","128 128 255","255 255 128","255 128 255","128 255 255"]

        for t in range(min(shape_num, 12)):
            model_float = np.zeros([self.config.frame_grid_size + 2, self.config.frame_grid_size + 2, self.config.frame_grid_size + 2, self.gf_split], np.float32)
            batch_voxels = self.data_voxels[t:t + 1].astype(np.float32)
            sq_batch_voxel = np.squeeze(batch_voxels)
            vertices_gt, triangles_gt = mcubes.marching_cubes(sq_batch_voxel, self.config.sampling_threshold)
            write_ply_triangle(self.config.sample_dir + "/" + str(t) + "gt_vox.ply", vertices_gt, triangles_gt)
            batch_voxels = torch.from_numpy(batch_voxels)
            batch_voxels = batch_voxels.to(self.device)
            z_vector, _, _ = self.bae_model(batch_voxels, None, None, is_training=False)
            for i in range(multiplier):
                for j in range(multiplier):
                    for k in range(multiplier):
                        minib = i * multiplier2 + j * multiplier + k
                        point_coord = self.coords[minib:minib + 1]
                        _, part_out, net_out = self.bae_model(None, z_vector, point_coord, is_training=False)
                        model_float[self.aux_x + i + 1, self.aux_y + j + 1, self.aux_z + k + 1, :] = np.reshape(
                            part_out.detach().cpu().numpy(), [self.test_size, self.test_size, self.test_size, self.gf_split])

            thres = 0.4
            vertices_num = 0
            triangles_num = 0
            vertices_list = []
            triangles_list = []
            vertices_num_list = [0]
            for split in range(self.gf_split):
                vertices, triangles = mcubes.marching_cubes(model_float[:,:,:,split], thres)
                vertices_num += len(vertices)
                triangles_num += len(triangles)
                vertices_list.append(vertices)
                triangles_list.append(triangles)
                vertices_num_list.append(vertices_num)

            #output ply
            fout = open(self.config.sample_dir+"/"+str(t)+"_vox.ply", 'w')
            fout.write("ply\n")
            fout.write("format ascii 1.0\n")
            fout.write("element vertex "+str(vertices_num)+"\n")
            fout.write("property float x\n")
            fout.write("property float y\n")
            fout.write("property float z\n")
            fout.write("property uchar red\n")
            fout.write("property uchar green\n")
            fout.write("property uchar blue\n")
            fout.write("element face "+str(triangles_num)+"\n")
            fout.write("property uchar red\n")
            fout.write("property uchar green\n")
            fout.write("property uchar blue\n")
            fout.write("property list uchar int vertex_index\n")
            fout.write("end_header\n")

            for split in range(self.gf_split):
                vertices = (vertices_list[split])/self.frame_grid_size-0.5
                for i in range(len(vertices)):
                    color = color_list[split]
                    fout.write(str(vertices[i,0])+" "+str(vertices[i,1])+" "+str(vertices[i,2])+" "+color+"\n")

            for split in range(self.gf_split):
                triangles = triangles_list[split] + vertices_num_list[split]
                for i in range(len(triangles)):
                    color = color_list[split]
                    fout.write(color+" 3 "+str(triangles[i,0])+" "+str(triangles[i,1])+" "+str(triangles[i,2])+"\n")
            print("output!")


    def test_ae(self, config, name):
        multiplier = int(config.frame_grid_size / self.test_size)
        multiplier2 = multiplier * multiplier
        self.bae_model.eval()
        t = np.random.randint(len(self.data_voxels))
        model_float = np.zeros([config.frame_grid_size + 2, config.frame_grid_size + 2, config.frame_grid_size + 2], np.float32)
        batch_voxels = self.data_voxels[t:t + 1].astype(np.float32)
        batch_voxels = torch.from_numpy(batch_voxels)
        batch_voxels = batch_voxels.to(self.device)
        z_vector, _, _ = self.bae_model(batch_voxels, None, None, is_training=False)
        for i in range(multiplier):
            for j in range(multiplier):
                for k in range(multiplier):
                    minib = i * multiplier2 + j * multiplier + k
                    point_coord = self.coords[minib:minib + 1]
                    _, _, net_out = self.bae_model(None, z_vector, point_coord, is_training=False)
                    model_float[self.aux_x + i + 1, self.aux_y + j + 1, self.aux_z + k + 1] = np.reshape(
                        net_out.detach().cpu().numpy(), [self.test_size, self.test_size, self.test_size])


        vertices, triangles = mcubes.marching_cubes(model_float, self.config.sampling_threshold)
        vertices = (vertices.astype(np.float32) - 0.5) / config.frame_grid_size - 0.5
        # output ply sum
        write_ply_triangle(config.sample_dir + "/" + name + ".ply", vertices, triangles)
        print("[sample]")






if __name__ =="__main__":
    cfg = Config("/home/RS/CJH/chenjinghuan/implict3Dreconstruct/configs/BAE/BAE_AE_cfg.py")
    if not os.path.exists(cfg.config.model.sample_dir):
    	os.makedirs(cfg.config.model.sample_dir)
    BAE_NET_TRAIN = IMSEG(cfg.config.model)
    BAE_NET_TRAIN.train(cfg.config.model)
    # BAE_NET_TRAIN.test_ae_all(cfg.config.model)
    # BAE_NET_TRAIN.test_ae_withcolor(cfg.config.model)
    # BAE_NET_TRAIN.get_Z_vector(cfg.config.model)

