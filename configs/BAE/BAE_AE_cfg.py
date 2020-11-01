from addict import Dict

model = Dict(
    real_size = 64, # point-value voxel grid size in training [64]"
    points_per_shape = 8192, # "num of points per shape [32768]"
    input_size = 64, # input voxel grid size
    z_dim = 128, # encoder output feature shape [n, 128]
    L1reg = True,  # "True for adding L1 regularization at layer 3 [False]"
    supervised = False,  # "True for supervised training, False for unsupervised [False]
    is_training = True,
    test_size = 32, #related to testing batch_size, adjust according to gpu memory size

    oringal_voxel_size = 256, # normal the input dataset
    frame_grid_size = 64,
    sampling_threshold = 0.5,
    checkpoint_name='IM_AE.model',
    cell_grid_size = 4,

    #encoder_cfg
    ef_dim = 32,  #encoder base featrue shape

    #  generator_cfg
    point_dim = 3, # coordinate of n point [n, 3]
    gf_dim = 256,
    gf_split = 8,

    #train_cfg
    batch_size_train = 12,
    sample_dir = "./samples/03001627_chair/AE",
    epoch = 2000, #Iteration to train. Either epoch or iteration need to be zero [0]
    iteration = 0,
    learning_rate = 0.000015, # default 0.00005
    beta1 = 0.5, # Momentum term of adam [0.5]
    dataset_name = "03001627_vox256_img",
    AE_checkpoint = "checkpoint/03001627_chair",
    data_dir = "./data/03001627_chair",
    max_to_keep = 5,
)



