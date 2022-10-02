#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import time
import numpy as np

from utils import *
from models import *

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.utils import to_categorical

class Client(object):
    def __init__(self, args, device):
        self.args = args
        self.device = device

    def update(self, data_train_x, data_train_y, data_train_act, encoder_model, decoder_model, aux_model):

        encoder_model = encoder_model.to(self.device)
        decoder_model = decoder_model.to(self.device)
        aux_model = aux_model.to(self.device)

        # learn2learn lib wrapper
        meta_encoder = MAML(encoder_model, lr=self.args.maml_lr)
        meta_decoder = MAML(decoder_model, lr=self.args.maml_lr)
        meta_aux = MAML(aux_model, lr=self.args.maml_lr)
        
        encoder_learner = meta_encoder.clone()
        decoder_learner = meta_decoder.clone()
        aux_learner = meta_aux.clone()

        epoch_loss = []
        epoch_aux_loss = []
        
        # Split support set & query set
        train_x, train_y, train_act = data_train_x[:self.args.k_spt, :], data_train_y[:self.args.k_spt, :], data_train_act[:self.args.k_spt]
        train_x, train_y, train_act = train_x.to(self.device), train_y.to(self.device), train_act.to(self.device)

        train_qry_x, train_qry_y, train_qry_act = data_train_x[self.args.k_spt:, :], data_train_y[self.args.k_spt:, :], data_train_act[self.args.k_spt:]
        train_qry_x, train_qry_y, train_qry_act = train_qry_x.to(self.device), train_qry_y.to(self.device), train_qry_act.to(self.device)

        # Loop local steps on the support set
        for local_step in range(self.args.inner_update_step):
            batch_loss = 0
            batch_aux_loss = 0
            qry_batch_loss = 0
            qry_batch_aux_loss = 0

            # Generate shadow samples
            with torch.no_grad():
                fake_z_samples, mu_raw, log_var_raw = encoder_learner(train_x.detach())

                # Generate fake private attribute
                rest_private = np.arange(self.args.num_private_attr)
                rest_private = np.delete(rest_private, np.argmax(train_y[0].detach().cpu().numpy()))
                fake_y = np.random.choice(rest_private, len(train_x))                
                fake_y = to_categorical(fake_y, num_classes=self.args.num_private_attr)
                fake_y = torch.from_numpy(fake_y)
                fake_y = fake_y.to(self.device)
                
                # Fake public attributes remain the same as the original ones
                fake_pub = train_act.to(self.device)
                
                # Concatenate fake latent representations
                cat_fake = torch.cat((fake_z_samples, fake_y, fake_pub), dim=1)
                
                # Generate fake samples
                train_fake = decoder_learner(cat_fake)
                
                # Concatenate fake samples with the raw support set
                train_x_fake = torch.cat((train_x, train_fake))
                train_y_fake = torch.cat((train_y, fake_y))
                train_act_fake = torch.cat((train_act, fake_pub))
                
                # Shuffle new support set
                shuffle_idx=torch.randperm(train_x_fake.size()[0])
                train_x_fake = train_x_fake[shuffle_idx]
                train_y_fake = train_y_fake[shuffle_idx]
                train_act_fake = train_act_fake[shuffle_idx]
                
                # Input to train discriminator
                train_z_fake, mu_fake, log_var_fake = encoder_learner(train_x_fake)
                
            # Use raw support set to update the VAE
            train_z, mu, log_var = encoder_learner(train_x)
            cat_z = torch.cat((train_z, train_y, train_act), dim=1)
            train_xr = decoder_learner(cat_z)
            
            for aux_loop in range(self.args.aux_ep):
                # Train the aux net to predict y from z
                auxY_fake = aux_learner(train_z_fake.detach()) # detach: to ONLY update AUX
                auxLoss_fake = F.binary_cross_entropy_with_logits(auxY_fake, train_y_fake) # correct order  #predY is a Nx2 use 2nd col.
                aux_learner.adapt(auxLoss_fake)
                batch_aux_loss += auxLoss_fake.detach().cpu().numpy()
                batch_aux_loss_counter += 1

            # Train the encoder to NOT predict y from z
            auxK = aux_learner(train_z)
            auxEncLoss = F.binary_cross_entropy_with_logits(auxK, train_y)
            recons_loss = F.mse_loss(train_xr, train_x) * 512
            kld_loss = torch.mean(-2 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim = 0)
            loss = (recons_loss + kld_loss)/150 - self.args.alpha * auxEncLoss
            
            encoder_learner.adapt(loss)
            decoder_learner.adapt(loss)

            
        #------- Compute Meta Loss on Query Set -------
        train_qry_z, qry_mu, qry_log_var = encoder_learner(train_qry_x)
        qry_cat_z = torch.cat((train_qry_z, train_qry_y, train_qry_act), dim=1)
        train_qry_xr = decoder_learner(qry_cat_z)

        qry_auxY = aux_learner(train_qry_z.detach()) # detach: to ONLY update AUX
        qry_auxLoss = F.binary_cross_entropy_with_logits(qry_auxY, train_qry_y)
        qry_auxLoss.backward(retain_graph=True)
        
        aux_grads = get_parameters_gradients(meta_aux)
        qry_batch_aux_loss += qry_auxLoss.detach()
        
        qry_auxK = aux_learner(train_qry_z)  # not detached update the encoder
        qry_auxEncLoss = F.binary_cross_entropy_with_logits(qry_auxK, train_qry_y)
        qry_recons_loss = F.mse_loss(train_qry_xr, train_qry_x) * 512
        qry_kld_loss = torch.mean(-2 * torch.sum(1 + qry_log_var - qry_mu.pow(2) - qry_log_var.exp(), dim=1), dim = 0)
        qry_loss = (qry_recons_loss + qry_kld_loss)/150  - self.args.alpha * qry_auxEncLoss      
        
        qry_loss.backward()

        encoder_grads = get_parameters_gradients(meta_encoder)
        decoder_grads = get_parameters_gradients(meta_decoder)
        
        return encoder_grads, decoder_grads, aux_grads, qry_loss.detach().cpu().numpy(), qry_recons_loss.detach().cpu().numpy(), qry_kld_loss.detach().cpu().numpy(), qry_auxLoss.detach().cpu().numpy()


def inference(args, device, encoder_model, decoder_model, eval_act_model, eval_gen_model, x_test, activity_test_label, gender_test_label, act_labels, gen_labels):
    test_data = x_test
    X = x_test

    print("Raw Activity Identification:")
    Y_act = eval_act_model.predict(X, verbose=args.verbose)
    Y_act_labels = np.argmax(Y_act, axis=1)  # generate predicted vector of labels
    pred_act = to_categorical(Y_act_labels, num_classes=args.num_public_attr)
    print_accu_score(Y_true=activity_test_label, Y_pred=pred_act)
    print_accu_confmat_f1score(Y_true=np.argmax(activity_test_label, axis=1), Y_pred=Y_act_labels, txt_labels=act_labels)
    
    if args.private == 'gender':
        print("Raw Gender Identification:")
        Y_gen = eval_gen_model.predict(X, verbose=args.verbose)
        Y_gen_labels = np.where(Y_gen > 0.5, 1, 0)

    elif args.private == 'weight':
        print("Raw Weight Identification:")
        Y_gen = eval_gen_model.predict(X, verbose=args.verbose)
        Y_gen_labels = np.argmax(Y_gen, axis=1)

    pred_gen = to_categorical(Y_gen_labels, num_classes=args.num_private_attr)        
    print_accu_score(Y_true=gender_test_label, Y_pred=pred_gen)
    print_accu_confmat_f1score(Y_true=np.argmax(gender_test_label, axis=1), Y_pred=Y_gen_labels, txt_labels=gen_labels)
    
    # Prepare anonymzied dataset and evaluate test accuracy for anonymized test set
    hat_test_data = np.empty((0, args.sample_size), float)
    
    # Generate dataset X
    X_inside = test_data
    X_inside = np.reshape(X_inside, [X_inside.shape[0], args.sample_size])
    tensor_X = torch.from_numpy(X_inside)  # transform to torch tensor

    # Generate gender dataset Y, predicted gender for inside activity
    y_dataset = np.copy(pred_gen)
    tensor_Y = torch.from_numpy(y_dataset)
    test_dataset = TensorDataset(tensor_X.float(), tensor_Y.float())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    z = np.empty((0, args.z_dim), float)

    for batch_idx, (test_x, test_y) in enumerate(test_loader):
        test_x, test_y = test_x.to(device), test_y.to(device)
        z_batch = encoder_model(test_x)[0]
        z = np.append(z, z_batch.data.cpu(), axis=0)

    z_train = z.copy()
    tensor_z = torch.from_numpy(z_train)

    # Anonymize latent representations:
    y_dataset = np.copy(pred_gen)  # get a copy of the predicted gender    
    if args.anonymize == 'random':
        obscured_labels = np.random.randint(args.num_private_attr, size=len(y_dataset)) # 2 private attributes, uniformly genearte random integers in [0,2)
        y_dataset = to_categorical(obscured_labels, num_classes=args.num_private_attr)
    elif args.anonymize == 'determ':
        argmax_ys = np.argmax(y_dataset, axis=1)
        determ_y_dataset = np.mod(np.add(1, argmax_ys), args.num_private_attr)  # reverse gender label to obscure
        y_dataset = to_categorical(determ_y_dataset, num_classes=args.num_private_attr)         

    tensor_y = torch.from_numpy(y_dataset)
    tensor_act = torch.from_numpy(pred_act)

    z_dataset = TensorDataset(tensor_z.float(), tensor_y.float(), tensor_act.float())
    z_loader = torch.utils.data.DataLoader(z_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    for batch_idx, (z, y, act) in enumerate(z_loader):
        z, y, act = z.to(device), y.to(device), act.to(device)
        z_cat = torch.cat((z, y, act), dim=1)
        x_hat = decoder_model(z_cat.float())
        hat_test_data = np.append(hat_test_data, x_hat.data.cpu(), axis=0)

    print()

    if args.dataset=='mobi':
        X = np.reshape(hat_test_data, [test_data.shape[0], test_data.shape[1], test_data.shape[2],test_data.shape[3]])
    elif args.dataset=='motion':
        X = np.reshape(hat_test_data, [test_data.shape[0], test_data.shape[1]])
        
    print("Testing Anonymzied Activity Identification:")

    Y_act_test_pred = eval_act_model.predict(X, verbose=args.verbose)
    Y_act_test_labels = np.argmax(Y_act_test_pred, axis=1)  # generate predicted labels
    pred_act_onehot = to_categorical(Y_act_test_labels, num_classes=args.num_public_attr)
    argmax_act_test_labels = np.argmax(activity_test_label, axis=-1)
    print_accu_score(Y_true=argmax_act_test_labels, Y_pred=Y_act_test_labels)

    Y_act_true = argmax_act_test_labels
    Y_act_pred = Y_act_test_labels

    if args.private == 'gender':
        print("Testing Anonymized Gender Identification:")
        Y_gen_test_pred = eval_gen_model.predict(X, verbose=args.verbose)
        Y_gen_test_label = np.where(Y_gen_test_pred > 0.5, 1, 0)
    elif args.private == 'weight':
        print("Testing Anonymized Weight Identification:")
        Y_gen_test_pred = eval_gen_model.predict(X, verbose=args.verbose)
        Y_gen_test_label = np.argmax(Y_gen_test_pred, axis=1)

    argmax_gen_test_label = np.argmax(gender_test_label, axis=1)
    print_accu_score(Y_true=argmax_gen_test_label, Y_pred=Y_gen_test_label)
    Y_gen_true = argmax_gen_test_label
    Y_gen_pred = Y_gen_test_label      

    return Y_act_true, Y_act_pred, Y_gen_true, Y_gen_pred
