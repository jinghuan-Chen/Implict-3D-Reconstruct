from addict import Dict

model = Dict(
    img_ef_dim = 64, #image encoder base featrue shape
    gf_dim = 256,
    z_dim = 128, # shape of image feature vector
    point_dim = 3,
    gf_split = 8,
    frame_grid_size = 64,
    
    is_training = True,

    sampling_threshold = 0.5,
    image_batch_size = 16,
    imput_image_size = 137,
    crop_size = 128,
    view_num = 24, # the image number of a same shape

    batch_size_train = 12,
    sample_dir = "./samples/03001627_chair/SVR",
    epoch = 2000, #Iteration to train. Either epoch or iteration need to be zero [0]
    iteration = 0,
    learning_rate = 0.000015, # default 0.00005
    beta1 = 0.5, # Momentum term of adam [0.5]
    dataset_name = "03001627_vox256_img",
    data_dir = "./data/03001627_chair",
    test_size = 32,
    z_vector_dataPath = "/home/RS/CJH/chenjinghuan/implict3Dreconstruct/checkpoint/03001627_chair/03001627_vox256_img_ae_64_train_z.hdf5",
    checkpoint_name='IM_SVR.model',
    checkpoint = "checkpoint/03001627_chair_SVR",
    max_to_keep = 5,

)