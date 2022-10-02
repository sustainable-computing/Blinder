#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[13]:


#!/home/xyang18/miniconda3/envs/pytorch/bin/ python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import sys
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import WeightedRandomSampler
# from tensorboardX import SummaryWriter

from options import *
from update import *
from models import *
from utils import *
from dataset_loader import *

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from collections import Counter
from matplotlib import pyplot as plt

# class Args:
#     def __init__(self):
#         self.verbose = 1
#         self.gpu_id = 3
#         self.seed = 30 # random seed
#         self.z_dim = 25 # size of latent representation
#         self.frac = 0.4 # fraction of users selected in the training per epoch
#         self.epochs = 20 # number of training epoches
#         self.inner_update_step = 1  # task-level inner update steps
#         self.aux_ep = 5  # training steps of the discriminator
#         self.TRAIN_MODEL = True
#         self.k_spt = 16 # size of the support set
#         self.k_qry = 48 # size of the query set
#         self.k_finetune = 16 # samples used for finetuning
#         self.batch_size = 256 # batchsize for test only
#         self.lr = 1e-3 # learning rate
#         self.maml_lr = 1e-4 # meta learning rate
#         self.alpha = 0.2 # hyper parameter for auxloss
#         self.finetune_step = 10
#         self.train_sampler = 'smote' # 'none' no sampler used in training set; 'smote': smote sampler
#         self.dataset = 'mobi' # dataset, 'mobi' for MobiAct; 'motion' for MotionSense
#         self.anonymize = 'random' # 'determ': deterministic anonymization; 'random': stochastic anonymization
#         self.num_users = 36 # total number of users in the dataset
#         self.num_public_attr = 4 # total number of public attributes
#         self.num_private_attr = 2  # total number of private attributes
#         self.private = 'gender' # 'weight', 'gender'
#         self.smote_num = 4000
# args = Args()


args = args_parser()


if args.dataset=='mobi':
    args.sample_size=768
    args.num_users = 36
    args.num_public_attr=4
    args.smote_num = 4000
elif args.dataset=='motion':
    args.sample_size=256
    args.num_users = 24
    args.num_public_attr=4
    args.smote_num = 600
    
if args.private == 'weight':
    args.num_private_attr = 3
elif args.private == 'gender':
    args.num_private_attr = 2
        

activities = np.arange(args.num_public_attr).astype(int)

comment = ''
model_folder = '../models/Blinder_' + str(args.num_users) + '_' + args.dataset + '_' + args.private + '_ep' + str(args.epochs) + '_localep' + str(
    args.inner_update_step) +'_spt' + str(args.k_spt) + '_qry' + str(args.k_qry) + '_seed' + str(args.seed) + '_trainsamp_' + args.train_sampler + comment + '/'

# print(model_folder)

# Create folder to save models
try:
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
except OSError as err:
    print(err)

# Save a copy of all the arguments
save_exp_details(args, model_folder)


# In[ ]:


path_project = os.path.abspath('..')
# logger = SummaryWriter('../logs')

# torch.multiprocessing.set_sharing_strategy('file_system')

if args.gpu_id>=0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    cuda_id = "cuda:" + str(0)  # cuda:2

device = torch.device(cuda_id if torch.cuda.is_available() else "cpu")
print("Device:", device)
if (torch.cuda.is_available()):
    torch.cuda.set_device(cuda_id)
    print("Current GPU ID:", torch.cuda.current_device())


# In[ ]:


# Uncomment to dump output to txt file
# sys.stdout = open(model_folder+'output.txt','w')


# In[ ]:


# load dataset
if args.dataset=='mobi':
    x_train, x_test, activity_train_label, activity_test_label, gender_train_label, gender_test_label, weight_train_label, weight_test_label, user_groups, user_groups_test, id_train, id_test = load_mobiact(args)
elif args.dataset=='motion':
    x_train, x_test, activity_train_label, activity_test_label, gender_train_label, gender_test_label, user_groups, user_groups_test, id_train, id_test = load_motionsense()

if args.private=='gender':
    private_train_label = gender_train_label
    private_test_label = gender_test_label
elif args.private == 'weight':
    private_train_label = weight_train_label
    private_test_label = weight_test_label


# In[ ]:


# Prepare datasets for every client, balance public attribtues using SMOTE
def get_train_loaders(args, x_train, private_train_label, public_train_label):
    user_train_loaders = {}
    if args.dataset=='mobi':
        x_train = np.reshape(x_train, [x_train.shape[0], args.sample_size])

    for user_id in range(args.num_users):
        # generate weights for samples        
        if args.train_sampler == 'smote':
            user_filter = [True if x in user_groups[user_id] else False for x in id_train]
            raw_user_tensor_x = x_train[user_filter]  # transform to torch tensor
            raw_user_tensor_y = private_train_label[user_filter]  # onehot
            user_act_onehot = public_train_label[user_filter]
            user_act = np.argmax(user_act_onehot, axis=1)
            raw_act_counter=Counter(user_act)
            print('User', user_id, ': Original dataset %s' % raw_act_counter)
            smote_strategy = {}
            for i in range(args.num_public_attr):
                smote_strategy[i] = max(args.smote_num, raw_act_counter[i])
            sm = SMOTE(sampling_strategy=smote_strategy, k_neighbors=5)
            X_res, y_res = sm.fit_resample(raw_user_tensor_x, user_act)
            down_strategy = {}
            for i in range(args.num_public_attr):
                down_strategy[i] = args.smote_num
            rus = RandomUnderSampler(random_state=args.seed, sampling_strategy=down_strategy)
            X_res, y_res = rus.fit_resample(X_res, y_res)
            print('User', user_id, ': Resampled dataset %s' % Counter(y_res))
            user_tensor_x = torch.as_tensor(X_res.astype('float32'))
            y_res = to_categorical(y_res, num_classes=args.num_public_attr)
            user_tensor_act = torch.as_tensor(y_res.astype('float32'))
            user_gen = np.argmax(raw_user_tensor_y[0]) # user gender, digit, not one-hot
            new_gens = np.ones(len(X_res)) * user_gen # fill new gender array with user's gender
            user_tensor_y = torch.from_numpy(to_categorical(new_gens, num_classes=args.num_private_attr).astype('float32'))
            user_dataset = TensorDataset(user_tensor_x, user_tensor_y, user_tensor_act)
            user_train_loader = list(torch.utils.data.DataLoader(user_dataset, batch_size=args.k_spt + args.k_qry, shuffle=True, pin_memory=True))               
        elif args.train_sampler == 'none':
            user_filter = [True if x in user_groups[user_id] else False for x in id_train]
            user_tensor_x = torch.from_numpy(x_train[user_filter].astype('float32'))  # transform to torch tensor
            user_tensor_y = torch.from_numpy(private_train_label[user_filter].astype('float32'))  # onehot
            user_tensor_act = torch.from_numpy(public_train_label[user_filter].astype('float32'))
            user_dataset = TensorDataset(user_tensor_x, user_tensor_y, user_tensor_act)
            user_train_loader = list(torch.utils.data.DataLoader(user_dataset, batch_size=args.k_spt + args.k_qry, shuffle=True, pin_memory=True))
        user_train_loaders[user_id] = user_train_loader
    return user_train_loaders

if args.TRAIN_MODEL:
    user_train_loaders = get_train_loaders(args, x_train, private_train_label, activity_train_label)
    dataset_generators = {}
    for uid in range(args.num_users):
        dataset_generators[uid] = iter(user_train_loaders[uid])


# In[ ]:


if args.TRAIN_MODEL:
    train_loss, train_aux_loss, train_accuracy = [], [], []
    
    global_encoder = Encoder(args.z_dim, args.sample_size)
    global_decoder = Decoder(args.z_dim, args.num_private_attr, args.num_public_attr, args.sample_size)
    global_aux = AUX(args.z_dim, numLabels=args.num_private_attr)

    global_encoder.to(device)
    global_decoder.to(device)
    global_aux.to(device)

    global_encoder.train()
    global_decoder.train()
    global_aux.train()

    optimizer_encoder = torch.optim.Adam(global_encoder.parameters(), lr=args.lr)
    optimizer_decoder = torch.optim.Adam(global_decoder.parameters(), lr=args.lr)
    optimizer_aux = torch.optim.Adam(global_aux.parameters(), lr=args.lr)
    
    print("########################################################")

    # get smallest number of local batches, should be the same if data is rebalanced
    min_dataset_batch = sys.maxsize
    for i in range(args.num_users):
        min_dataset_batch = min(min_dataset_batch, len(user_train_loaders[i]))
    print('Min Dataset batch:', min_dataset_batch)
    
    # number of selected users
    m = max(int(args.frac * args.num_users), 1)
        
    for epoch in tqdm(range(args.epochs)):
        print(f'\n | Global Training Round : {epoch + 1} |')

        # randomly select m users from total users
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print("Participating Users:", idxs_users)
        
        for comm_round in range(min_dataset_batch):
            sum_qry_loss = 0
            sum_qry_aux_loss = 0
            sum_qry_recon_loss = 0
            sum_qry_kld_loss = 0
            
            local_encoder_grads = []
            local_decoder_grads = []
            local_aux_grads = []

            # local training steps:
            for idx in idxs_users:
                try:
                    train_x, train_y, train_act = next(dataset_generators[idx])
                except StopIteration:
                    # restart the generator if the previous generator is exhausted.
                    dataset_generators[idx] = iter(user_train_loaders[idx])
                    train_x, train_y, train_act = next(dataset_generators[idx])
                
                train_act = train_act.type(torch.LongTensor)

                local_model = Client(args=args, device=device)
                encoder_grads, decoder_grads, aux_grads, qry_loss, qry_recons_loss, qry_kld_loss, qry_auxLoss = local_model.update(
                    data_train_x=train_x, data_train_y=train_y, data_train_act=train_act,
                    encoder_model=copy.deepcopy(global_encoder),
                    decoder_model=copy.deepcopy(global_decoder),
                    aux_model=copy.deepcopy(global_aux))

                local_encoder_grads.append(copy.deepcopy(encoder_grads))
                local_decoder_grads.append(copy.deepcopy(decoder_grads))
                local_aux_grads.append(copy.deepcopy(aux_grads))

                sum_qry_loss += qry_loss
                sum_qry_recon_loss += qry_recons_loss
                sum_qry_kld_loss += qry_kld_loss
                sum_qry_aux_loss += qry_auxLoss

            avg_qry_loss = sum_qry_loss/m
            avg_qry_recon_loss = sum_qry_recon_loss/m
            avg_qry_kld_loss = sum_qry_kld_loss/m
            avg_qry_aux_loss = sum_qry_aux_loss/m

            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            optimizer_aux.zero_grad()

            # Average local gradients
            avg_encoder_grads = average_gradients(local_encoder_grads)
            avg_decoder_grads = average_gradients(local_decoder_grads)
            avg_aux_grads = average_gradients(local_aux_grads)
            
            # Update global models
            global_encoder = update_global_grads(global_encoder, avg_encoder_grads)
            global_decoder = update_global_grads(global_decoder, avg_decoder_grads)
            global_aux = update_global_grads(global_aux, avg_aux_grads)

            # Backpropagation
            optimizer_encoder.step()
            optimizer_decoder.step()
            optimizer_aux.step()

            train_loss.append(avg_qry_loss)
            train_aux_loss.append(avg_qry_aux_loss)
            
            if (args.verbose and comm_round % 50 == 0):
                print("Global EP %d, Local Comm Round %d: Total loss: %f, AUX loss: %f, MSE: %f, KLD loss: %f" % 
                      (epoch+1, comm_round, avg_qry_loss, avg_qry_aux_loss, avg_qry_recon_loss, avg_qry_kld_loss))
                
        # Save model at the end of every epoch in case of losing progress caused by numerical instability
        path_encoder, path_decoder, path_aux = get_paths_encoder_decoder_aux(model_folder=model_folder, z_dim=args.z_dim, model=args.model)
        torch.save(global_encoder.state_dict(), path_encoder)
        torch.save(global_decoder.state_dict(), path_decoder)
        torch.save(global_aux.state_dict(), path_aux)

    # Save loss plots
    fig1, ax1 = plt.subplots()
    lines1, = ax1.plot(np.arange(len(train_loss)), train_loss)
    fig1.savefig(model_folder + 'train_loss.png', dpi=300)
    plt.show()
    plt.close(fig1)

    fig2, ax2 = plt.subplots()
    lines2, = ax2.plot(np.arange(len(train_aux_loss)), train_aux_loss)
    fig2.savefig(model_folder + 'train_aux_loss.png', dpi=300)
    plt.show()
    plt.close(fig2)


# ## Evaluation

# In[ ]:


# Load desired inference model & private attribute inference model
def load_eval_models(args):
    path = "../eval_models/"
    if args.dataset=='mobi':
        eval_act_model = load_model(path + "Inference models MobiAct/activity_model_DC.hdf5")
        if args.private=='gender':
            eval_private_model = load_model(path + "Inference models MobiAct/gender_model_DC.hdf5")
        elif args.private=='weight':
            eval_private_model = load_model(path + "Inference models MobiAct/weight_model_DC.hdf5")
    elif args.dataset=='motion':
        eval_act_model = load_model(path + "Inference models MotionSense/my_activity_model_mlp.hdf5")
        eval_private_model = load_model(path + "Inference models MotionSense/my_gender_model_mlp.hdf5")
    return eval_act_model, eval_private_model

eval_act_model, eval_private_model = load_eval_models(args)


# In[ ]:


if args.dataset=='mobi':
    public_txt_labels = ["wlk","std", "jog", "ups"]
elif args.dataset =='motion':
    public_txt_labels = ["dws","ups", "wlk", "jog"]
    
if args.private=='gender':
    private_txt_labels = ["m", "f"]
elif args.private=='weight':
    private_txt_labels = ["<=70", "70-90", ">90"]

enc_model, dec_model, aux_model = get_models_encoder_decoder_aux(model_folder, args)
enc_model, dec_model, aux_model = enc_model.to(device), dec_model.to(device), aux_model.to(device)


# In[ ]:


Y_act_true, Y_act_pred, Y_gen_true, Y_gen_pred = inference(args, device, enc_model, dec_model, 
                                                           eval_act_model, eval_private_model, 
                                                           x_test, activity_test_label, private_test_label, 
                                                           public_txt_labels, private_txt_labels)

print("EVALUATION ON ALL TEST SET:")
print("Public Attribute Identification:")
print_accu_confmat_f1score(Y_act_true, Y_act_pred, public_txt_labels)
print("Private Attribute Identification:")
print_accu_confmat_f1score(Y_gen_true, Y_gen_pred, private_txt_labels)

