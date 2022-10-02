#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
from models import *
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def average_gradients(g):
    """
    Returns the average of the gradients.
    """
    g_avg = g[0]
    for j in range(len(g_avg)):
        for i in range(1, len(g)):
            g_avg[j] += g[i][j]
        g_avg[j] = torch.divide(g_avg[j], len(g))
    return g_avg

def get_parameters_gradients(model):
    grads = []
    for param in model.parameters():
        grads.append(param.grad.detach())
    return grads


def update_global_weights(state_dict, grads, lr):
    """
    Update global weights using SGD.
    """
    idx = 0
    for key in state_dict.keys():
        state_dict[key] -= lr * grads[idx]
        idx += 1
    return state_dict

def update_global_grads(model, grads):
    counter = 0
    for p in model.parameters():
        p.grad = grads[counter]
        counter += 1
    return model

def save_exp_details(args, model_folder):
    config_filename = 'config.txt'
    print("\nExperimental details:")

    with open(model_folder + config_filename, 'w') as f:
        f.writelines("Experimental details:\n")
        for attr, value in args.__dict__.items():
            print(attr, '=', value)
            f.writelines(str(attr) + '=' + str(value) + '\n')
    return

def print_accu_score(Y_true, Y_pred):
    accu = accuracy_score(y_true=Y_true, y_pred=Y_pred)
    print("    " + str(accu))
    print("    ***[EVALUATION RESULT]*** Accuracy: " + str(round(accu, 4) * 100) + "%\n")
    
def eval_accu(model, X, Y):
    result = model.evaluate(X, Y, verbose=1)
    print(result)
    print("***[EVALUATION RESULT]*** Accuracy: " + str(round(result[1], 4) * 100) + "%")


def print_accu_confmat_f1score(Y_true, Y_pred, txt_labels):
    act_accu = accuracy_score(y_true=Y_true, y_pred=Y_pred)
    print("***[MY EVALUATION RESULT]*** Accuracy: " + str(round(act_accu, 4) * 100) + "%\n")

    conf_mat = confusion_matrix(y_true=Y_true, y_pred=Y_pred)
    conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    print("***[RESULT]***  Confusion Matrix:")
    print(" | ".join(txt_labels))
    print(np.array(conf_mat).round(3) * 100)
    print()

    f1act = f1_score(y_true=Y_true, y_pred=Y_pred, average=None).mean()
    print("***[RESULT]*** Averaged F-1 Score: " + str(f1act * 100) + "\n")
    
def get_paths_encoder_decoder_aux(model_folder, z_dim, model):
    path_encoder = model_folder + 'ml_obs_mobi_g_encoder_alpha_02_beta_2_' + str(z_dim) + model
    path_decoder = model_folder + 'ml_obs_mobi_g_decoder_alpha_02_beta_2_' + str(z_dim) + model
    path_aux = model_folder + 'ml_obs_mobi_g_aux_alpha_02_beta_2_' + str(z_dim) + model
    return path_encoder, path_decoder, path_aux


def get_models_encoder_decoder_aux(model_folder, args):
    path_encoder, path_decoder, path_aux = get_paths_encoder_decoder_aux(model_folder, z_dim=args.z_dim)
    
    encoder_model = Encoder(args.z_dim, args.sample_size)
    decoder_model = Decoder(args.z_dim, args.num_private_attr, args.num_public_attr, args.sample_size)    
        
    encoder_model.load_state_dict(torch.load(path_encoder, map_location=torch.device('cpu')))
    decoder_model.load_state_dict(torch.load(path_decoder, map_location=torch.device('cpu')))

    aux_model = AUX(args.z_dim, numLabels=args.num_private_attr)
    aux_model.load_state_dict(torch.load(path_aux, map_location=torch.device('cpu')))

    return encoder_model, decoder_model, aux_model
