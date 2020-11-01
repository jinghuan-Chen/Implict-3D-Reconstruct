from abc import ABCMeta, abstractclassmethod
import torch
import os
import time
import numpy as np


class IM_BASE(metaclass=ABCMeta):
    def __init__(self, config):
        self.config = config
        self.checkpoint_manager_list = [None] * int(config.max_to_keep)
        self.checkpoint_manager_pointer = 0

    @abstractclassmethod
    def build_model(self):
        pass

    @abstractclassmethod
    def _load_dataset(self):
        pass

    @property
    def model_dir(self):
        return "{}_ae_{}".format(self.config.dataset_name, self.config.input_size)

    def get_devices(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

    def save(self, save_dir):
        if not os.path.exists(self.config.checkpoint):
            os.makedirs(self.config.checkpoint)

        self.checkpoint_manager_pointer = (self.checkpoint_manager_pointer + 1) % self.config.max_to_keep
        # delete checkpoint
        if self.checkpoint_manager_list[self.checkpoint_manager_pointer] is not None:
            if os.path.exists(self.checkpoint_manager_list[self.checkpoint_manager_pointer]):
                os.remove(self.checkpoint_manager_list[self.checkpoint_manager_pointer])
        # save checkpoint
        torch.save(self.im_network.state_dict(), save_dir)
        # update checkpoint manager
        self.checkpoint_manager_list[self.checkpoint_manager_pointer] = save_dir
        # write file
        checkpoint_txt = os.path.join(self.config.checkpoint, "checkpoint")
        fout = open(checkpoint_txt, 'w')
        for i in range(self.config.max_to_keep):
            pointer = (self.checkpoint_manager_pointer + self.config.max_to_keep - i) % self.config.max_to_keep
            if self.checkpoint_manager_list[pointer] is not None:
                fout.write(self.checkpoint_manager_list[pointer] + "\n")
        fout.close()


    def loadCheckpoint(self):
        checkpoint_txt = os.path.join(self.config.AE_checkpoint, "checkpoint")
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            fin.close()
            self.bae_model.load_state_dict(torch.load(model_dir))
            print("[*] Load SUCCESS")
        else:
            print("[!] Load FAILED...")


    def get_coords_for_training(self):

        self.test_size = self.config.test_size
        self.batch_size = self.test_size * self.test_size * self.test_size #do not change

        dima = self.test_size
        dim = self.config.frame_grid_size
        self.aux_x = np.zeros([dima, dima, dima], np.uint8)
        self.aux_y = np.zeros([dima, dima, dima], np.uint8)
        self.aux_z = np.zeros([dima, dima, dima], np.uint8)
        multiplier = int(dim / dima)
        multiplier2 = multiplier * multiplier
        multiplier3 = multiplier * multiplier * multiplier
        for i in range(dima):
            for j in range(dima):
                for k in range(dima):
                    self.aux_x[i, j, k] = i * multiplier
                    self.aux_y[i, j, k] = j * multiplier
                    self.aux_z[i, j, k] = k * multiplier
        self.coords = np.zeros([multiplier3, dima, dima, dima, 3], np.float32)
        for i in range(multiplier):
            for j in range(multiplier):
                for k in range(multiplier):
                    self.coords[i * multiplier2 + j * multiplier + k, :, :, :, 0] = self.aux_x + i
                    self.coords[i * multiplier2 + j * multiplier + k, :, :, :, 1] = self.aux_y + j
                    self.coords[i * multiplier2 + j * multiplier + k, :, :, :, 2] = self.aux_z + k
        self.coords = (self.coords + 0.5) / dim - 0.5
        self.coords = np.reshape(self.coords, [multiplier3, self.batch_size, 3])
        self.coords = torch.from_numpy(self.coords)
        self.coords = self.coords.to(self.device)