#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # important parameters:
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id, set -1 for cpu')
    parser.add_argument('--dataset', type=str, default='mobi', help="name of dataset")
    parser.add_argument('--private', type=str, default='gender', help='private attribute: gender or weight')
    parser.add_argument('--anonymize', type=str, default='determ', help='anonymization strategy: determ or random')
    parser.add_argument('--seed', type=int, default=20, help='random seed')
    parser.add_argument('--verbose', type=int, default=1, help='verbose mode: 1 or 0')
    
    # model training:
    parser.add_argument('--TRAIN_MODEL', action='store_false', help='execute model training')
    parser.add_argument('--epochs', type=int, default=30, help="number of rounds of global training")
    parser.add_argument('--inner_update_step', type=int, default=1, help="the number of local updates")
    parser.add_argument('--aux_ep', type=int, default=5, help="the number of discriminator updates")                       
    parser.add_argument('--k_spt', type=int, default=1, help="local support set size")
    parser.add_argument('--k_qry', type=int, default=15, help="local query set size")
    
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--maml_lr', type=float, default=1e-4, help='meta learning rate')
    parser.add_argument('--alpha', type=float, default=0.2, help='hyper parameter')
    
    parser.add_argument('--frac', type=float, default=0.4, help='the fraction of clients selected in the model training')
    parser.add_argument('--z_dim', type=int, default=25, help='dimension of latent feature representation')
    parser.add_argument('--train_sampler', type=str, default='smote', help='sampler for training set: smote or none')
    parser.add_argument('--smote_num', type=int, default=4000, help='target data samples for each public attribute class')
    
    # automatically adjusted:
    parser.add_argument('--num_users', type=int, default=36, help="number of users in the dataset")
    parser.add_argument('--num_private_attr', type=int, default=2, help='number of private attribute classes')
    parser.add_argument('--num_public_attr', type=int, default=4, help='number of public attribute classes')
    parser.add_argument('--sample_size', type=int, default=768, help='dimension of data sample')
    
    # optional:
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for inference')
    
    # finetuning parameters:
    parser.add_argument('--k_finetune', type=int, default=32, help="number of data samples for finetuning")
    parser.add_argument('--finetune_step', type=int, default=10, help="number of finetuning steps")        

    args = parser.parse_args()
    return args