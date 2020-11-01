from BAE_AE import IM_BASE, generator
from models.blackbones.resnet import img_encoder
import torch
import torch.nn as nn
from utils.config import Config
import os
import sys
from torch.optim.lr_scheduler import LambdaLR, StepLR
import numpy as np
import h5py
import time
import torch.nn.functional as F
import mcubes
from utils.plyUtils import *

class IMSVR_Network(nn.Module):
    def __init__(self, cfg):
        super(IMSVR_Network, self).__init__()
        self.cfg = cfg
        self.img_encoder = img_encoder(cfg.img_ef_dim, cfg.z_dim)
        self.generator = generator(cfg.z_dim, cfg.point_dim, cfg.gf_dim, cfg.gf_split)


    def forward(self, inputs, z_vector, point_coord, is_training=False):
        part_, net_ = None, None
        if is_training:
            z_vector = self.img_encoder(inputs, is_training=is_training)
        else:
            if inputs is not None:
                z_vector = self.img_encoder(inputs, is_training=is_training)
            if z_vector is not None and point_coord is not None:
                 part_, net_ = self.generator(point_coord, z_vector, is_training=False)
        return z_vector, part_, net_


class IM_SVR(IM_BASE):
    def __init__(self, config):
        super(IM_SVR, self).__init__(config)

    def build_model(self):
        self.im_network = IMSVR_Network(self.config)
        self.im_network.to(self.device)

    def _load_dataset(self):

        allset_name = self.config.dataset_name + "_test.hdf5"
        if self.config.is_training:
            allset_name = self.config.dataset_name + "_train.hdf5"
        data_hdf5_name = self.config.data_dir + '/' + allset_name
        if os.path.exists(data_hdf5_name):
            data_dict = h5py.File(data_hdf5_name, 'r')
            print(data_dict.keys())
            self.crop_edge = self.config.imput_image_size - self.config.crop_size
            offset_x = int(self.crop_edge/2)
            offset_y = int(self.crop_edge/2)
            #reshape to NCHW
            self.data_pixels = np.reshape(data_dict['pixels'][:,:,offset_y:offset_y + self.config.crop_size, offset_x : offset_x+self.config.crop_size], 
                                            [-1, self.config.view_num, 1,self.config.crop_size, self.config.crop_size])
        else:
            print("error: cannot load " + data_hdf5_name)
            exit(0)

        if self.config.is_training:
            # dataz_hdf5_name = self.config.checkpoint + '/' + self.model_dir + '_train_z.hdf5'
            dataz_hdf5_name = self.config.z_vector_dataPath
            if os.path.exists(dataz_hdf5_name):
                dataz_dict = h5py.File(dataz_hdf5_name, 'r')
                self.data_zs = dataz_dict['zs'][:]
            else:
                print("error: cannot load "+dataz_hdf5_name)
                exit(0)
            if len(self.data_zs) != len(self.data_pixels):
                print("error: len(self.data_zs) != len(self.data_pixels)")
                print(len(self.data_zs), len(self.data_pixels))
                exit(0)

    def before_training(self, cfg):

        self.optimizer = torch.optim.Adam(self.im_network.parameters(), lr=self.config.learning_rate, betas=(self.config.beta1, 0.999))
        sche_lambda = lambda epoch: max(0.95 ** epoch, 0.01)
        self.scheduler_1ambda = StepLR(self.optimizer, step_size=50, gamma=0.98)
        # self.im_network.train()

    def network_loss(self, pred_z, gt_z):
        return torch.mean((pred_z - gt_z)**2)



    def test_SVR(self, config, name):

        self.get_coords_for_training()


        multiplier = int(config.frame_grid_size / config.test_size)
        multiplier2 = multiplier * multiplier
        self.im_network.eval()
        t = np.random.randint(len(self.data_pixels))
        model_float = np.zeros([config.frame_grid_size + 2, config.frame_grid_size + 2, config.frame_grid_size + 2], np.float32)
        test_idx = np.random.randint(self.config.view_num)
        batch_view = self.data_pixels[t:t+1, test_idx].astype(np.float32)/255.0
        batch_view = torch.from_numpy(batch_view)
        batch_view = batch_view.to(self.device)
        z_vector, _, _ = self.im_network(batch_view, None, None, is_training=False)
        for i in range(multiplier):
            for j in range(multiplier):
                for k in range(multiplier):
                    minib = i * multiplier2 + j * multiplier + k
                    point_coord = self.coords[minib:minib + 1]
                    _, _, net_out = self.im_network(None, z_vector, point_coord, is_training=False)
                    model_float[self.aux_x + i + 1, self.aux_y + j + 1, self.aux_z + k + 1] = np.reshape(
                                                    net_out.detach().cpu().numpy(), [self.test_size, self.test_size, self.test_size])


        vertices, triangles = mcubes.marching_cubes(model_float, self.config.sampling_threshold)
        vertices = (vertices.astype(np.float32) - 0.5) / config.frame_grid_size - 0.5
        # output ply sum
        write_ply_triangle(config.sample_dir + "/" + name + ".ply", vertices, triangles)
        print("[sample]")


    def trian(self, config):
        self._load_dataset()
        self.get_devices()
        self.build_model()
        self.before_training(config)
        self.loadCheckpoint()

        sys.stdout = open(os.path.join("./", config.dataset_name + "IMSVR_log.txt"), 'w')
        print("PID = %d" % os.getpid())

        shape_num = len(self.data_pixels)
        batch_index_list = np.arange(shape_num)

        print("\n\n----------net summary----------")
        print("training samples   ", shape_num)
        print("-------------------------------\n\n")

        start_time = time.time()
        assert config.epoch==0 or config.iteration==0
        training_epoch = config.epoch + int(config.iteration/shape_num)
        batch_num = int(shape_num/config.image_batch_size)
        print("batch_num = {0}".format(batch_num))
        for epoch in range(0, training_epoch):
            self.im_network.train()
            np.random.shuffle(batch_index_list)
            avg_loss, avg_num = 0, 0

            for idx in range(batch_num):
                dxb = batch_index_list[idx*config.batch_size_train:(idx+1)*config.batch_size_train]

                which_view = np.random.randint(config.view_num)
                # which_view = 7
                batch_view = self.data_pixels[dxb,which_view].astype(np.float32)/255.0
                batch_zs = self.data_zs[dxb]

                batch_view = torch.from_numpy(batch_view)
                batch_zs = torch.from_numpy(batch_zs)

                batch_view = batch_view.to(self.device)
                batch_zs = batch_zs.to(self.device)

                self.im_network.zero_grad()
                z_vector, _, _  = self.im_network(batch_view, None, None, is_training=True)
                err = self.network_loss(z_vector, batch_zs)

                err.backward()
                self.optimizer.step()

                avg_loss += err
                avg_num += 1
            self.scheduler_1ambda.step()
            print(" lr: %.6f Epoch: [%2d/%2d] time: %4.4f, loss_sp: %.6f" % (self.optimizer.param_groups[0]['lr'], epoch, training_epoch, time.time() - start_time, avg_loss/avg_num))
            if epoch%10==9:
                self.test_SVR(config,"train_"+str(epoch))
            if epoch%100==99:
                save_dir = os.path.join(self.config.checkpoint, self.config.checkpoint_name+"-"+str(epoch)+".pth")
                self.save(save_dir)
            sys.stdout.flush()



if __name__ == "__main__":

    cfg = Config("/home/RS/CJH/chenjinghuan/implict3Dreconstruct/configs/IM_SVR/IM_SVR_cfg.py")
    if not os.path.exists(cfg.config.model.sample_dir):
        os.makedirs(cfg.config.model.sample_dir)
    # if not os.path.exists(cfg.config.model.sample_dir):
    # 	os.makedirs(cfg.config.model.sample_dir)
    SVR_MODEL = IM_SVR(cfg.config.model)
    SVR_MODEL.trian(cfg.config.model)
    # SVR_MODEL.build_model()


