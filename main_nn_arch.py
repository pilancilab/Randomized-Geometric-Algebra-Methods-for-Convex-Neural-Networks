import os
import re
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn

import argparse
import utils

import time
from sklearn import linear_model
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from utils import set_seed, NNClassifier, evaluate, train, initialize_model, scnn_inner, accuracy, eval_model, cvx_solver_mosek, cvx_solver_evaluate
import warnings
warnings.filterwarnings("ignore")

def get_parser():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument("--data_name", type=str, default='IMDB', choices=['IMDB', 'Amazon', 'cvx-forum', 'cola', 'qqp',
        'ECG-signal','ECG-report','ECG-sr','mnist','cifar10','ECG-signal-mfcc','ECG-sr-mfcc'])
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--data_path", type=str, default = './data/')
    parser.add_argument("--seed", type=int, default = 1)
    parser.add_argument("--add_skip", action='store_true')
    parser.add_argument("--train_method", type=str, default='cvx', choices=['cvx', 'noncvx', 'lasso', 'lasso_unit'])
    parser.add_argument("--Epochs", type=int, default=10)
    parser.add_argument("--train_choice", type=str, default='std', choices=['std','f1'])
    parser.add_argument("--num_trial", type=int, default=5)
    parser.add_argument("--embed", type=str, default = 'OpenAI', choices=['OpenAI','Bert'])
    parser.add_argument("--train_num", type=str, default='std', choices=['std','f1','f2','f3','f4'])
    parser.add_argument("--Hidden", type=int, default=10)
    parser.add_argument("--shuffle", action='store_true')
    parser.add_argument("--solver", type=str, default='std', choices=['std','cvxpy'])
    parser.add_argument("--cpsolver", type=str, default='mosek', choices=['mosek','scs'])
    parser.add_argument("--add_eps", action='store_true')
    parser.add_argument("--eps", type=float, default = 1e-8)
    parser.add_argument("--aug_sym", action='store_true')
    parser.add_argument("--polish", action='store_true')
    parser.add_argument("--polish_freq", type=int, default=5)
    parser.add_argument("--sdim", type=int, default=100)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=20)
    return parser

def load_data_and_embeddings(dataset_name, data_path='data/'):
    # Base data path

    # Check the dataset name and set appropriate paths
    if dataset_name == "IMDB-OpenAI-2K":
        data_embeddings_path = data_path + "IMDB-OpenAI-2K-Embeddings.csv"
    elif dataset_name == "IMDB-Bert-2K":
        data_embeddings_path = data_path + "IMDB-Bert-2K-Embeddings.csv"
    elif dataset_name == "IMDB-OpenAI-full":
        data_embeddings_path = data_path + "IMDB-OpenAI-full-Embedding.csv"
    elif dataset_name == "IMDB-Bert-full":
        data_embeddings_path = data_path + "IMDB-Bert-full-Embeddings.csv"
    elif dataset_name == "Amazon-OpenAI-30K":
        data_embeddings_path = data_path + "AmazonPolarity-OpenAI-30K-Embedding.csv"
    elif dataset_name == "Amazon-Bert-30K":
        data_embeddings_path = data_path + "AmazonPolarity-Bert-30K-Embeddings.csv"
    elif dataset_name == "cvx-forum-OpenAI-full":
        data_embeddings_path = data_path + "cvx-forum-OpenAI-full-Embedding.csv"
    elif dataset_name == "cvx-forum-Bert-full":
        data_embeddings_path = data_path + "cvx-forum-Bert-full-Embeddings.csv"
    elif dataset_name == "glue-cola-OpenAI-full":
        data_embeddings_path = data_path + dataset_name+"-Embedding.csv"
    elif dataset_name == "glue-cola-Bert-full":
        data_embeddings_path = data_path + dataset_name+"-Embeddings.csv"
    elif dataset_name == "glue-qqp-OpenAI-30K":
        data_embeddings_path = data_path + "glue-qqp-OpenAI-30K-Embeddings.csv"
    elif dataset_name == "glue-qqp-Bert-30K":
        data_embeddings_path = data_path + "glue-qqp-Bert-30K-Embeddings.csv"
    elif dataset_name == "glue-qqp-OpenAI-50K":
        data_embeddings_path = data_path + "glue-qqp-OpenAI-50K-Embeddings.csv"
    elif dataset_name == "glue-qqp-Bert-50K":
        data_embeddings_path = data_path + "glue-qqp-Bert-50K-Embeddings.csv"
    elif dataset_name == 'ECG-report':
        data_embeddings_path = data_path + "ECG_newreports.csv"
    elif dataset_name == 'ECG-signal':
        data_embeddings_path = data_path + "cnn_emb_v2.csv"
    else:
        raise ValueError("Invalid dataset name.")

    # Load embeddings and convert to tensors
    def load_embeddings(file_path):
        in_df = pd.read_csv(file_path)
        embeddings = (in_df.iloc[:, :-1].values)
        labels = in_df.iloc[:, -1].values
        embeddings = torch.tensor(embeddings).float()
        return embeddings, labels

    def load_embeddings_ECG_signal(file_path):
        in_df = pd.read_csv(file_path)
        embeddings = (in_df.iloc[:, 1:].values)
        labels = None
        embeddings = torch.tensor(embeddings).float()
        return embeddings, labels

    if dataset_name == 'ECG-signal':
        data_embeddings, data_labels = load_embeddings_ECG_signal(data_embeddings_path)
    else:
        data_embeddings, data_labels = load_embeddings(data_embeddings_path)

    return data_embeddings, data_labels

def main():

    parser = get_parser()
    args = parser.parse_args()

    # create folders
    os.makedirs('./results', exist_ok = True)

    data_path = args.data_path 
    data_name = args.data_name 

    if data_name == 'IMDB':
        Bert_embeddings, Bert_labels = load_data_and_embeddings("IMDB-Bert-full", data_path=data_path)
        OpenAI_embeddings, OpenAI_labels = load_data_and_embeddings("IMDB-OpenAI-full", data_path=data_path)
    elif data_name == 'Amazon':
        Bert_embeddings, Bert_labels = load_data_and_embeddings("Amazon-Bert-30K", data_path=data_path)
        OpenAI_embeddings, OpenAI_labels = load_data_and_embeddings("Amazon-OpenAI-30K", data_path=data_path)
    elif data_name == 'cvx-forum':
        Bert_embeddings, Bert_labels = load_data_and_embeddings("cvx-forum-Bert-full", data_path=data_path)
        OpenAI_embeddings, OpenAI_labels = load_data_and_embeddings("cvx-forum-OpenAI-full", data_path=data_path)
    elif data_name == 'cola':
        Bert_embeddings, Bert_labels = load_data_and_embeddings("glue-cola-Bert-full", data_path=data_path)
        OpenAI_embeddings, OpenAI_labels = load_data_and_embeddings("glue-cola-OpenAI-full", data_path=data_path)
    elif data_name == 'qqp':
        Bert_embeddings, Bert_labels = load_data_and_embeddings("glue-qqp-Bert-50K", data_path=data_path)
        OpenAI_embeddings, OpenAI_labels = load_data_and_embeddings("glue-qqp-OpenAI-50K", data_path=data_path)
    elif data_name == 'ECG-signal':
        _, labels = load_data_and_embeddings("ECG-report", data_path=data_path)
        embeddings, _ = load_data_and_embeddings("ECG-signal", data_path=data_path)
        embeddings = torch.clamp(embeddings,max=1)*0.15
        Bert_embeddings, Bert_labels = embeddings.clone(), labels.copy()
        OpenAI_embeddings, OpenAI_labels = embeddings.clone(), labels.copy()
    elif data_name == 'ECG-signal-mfcc':
        _, labels = load_data_and_embeddings("ECG-report", data_path=data_path)
        embeddings = np.load('{}signal_mfcc.npy'.format(data_path))
        embeddings = torch.tensor(embeddings).float()
        Bert_embeddings, Bert_labels = embeddings.clone(), labels.copy()
        OpenAI_embeddings, OpenAI_labels = embeddings.clone(), labels.copy()
    elif data_name == 'ECG-report':
        embeddings, labels = load_data_and_embeddings("ECG-report", data_path=data_path)
        embeddings = torch.nan_to_num(embeddings)
        Bert_embeddings, Bert_labels = embeddings.clone(), labels.copy()
        OpenAI_embeddings, OpenAI_labels = embeddings.clone(), labels.copy()
    elif data_name == 'ECG-sr':
        embeddings_signal, _ = load_data_and_embeddings("ECG-signal", data_path=data_path)
        embeddings_report, labels = load_data_and_embeddings("ECG-report", data_path=data_path)
        embeddings_signal = torch.clamp(embeddings_signal,max=1)*0.15
        embeddings_report = torch.nan_to_num(embeddings_report)
        embeddings = torch.cat([embeddings_signal,embeddings_report],dim=1)
        Bert_embeddings, Bert_labels = embeddings.clone(), labels.copy()
        OpenAI_embeddings, OpenAI_labels = embeddings.clone(), labels.copy()
    elif data_name == 'ECG-sr-mfcc':
        embeddings = np.load('{}signal_mfcc.npy'.format(data_path))
        embeddings_signal = torch.tensor(embeddings).float()
        embeddings_report, labels = load_data_and_embeddings("ECG-report", data_path=data_path)
        embeddings_report = torch.nan_to_num(embeddings_report)
        embeddings = torch.cat([embeddings_signal,embeddings_report],dim=1)
        Bert_embeddings, Bert_labels = embeddings.clone(), labels.copy()
        OpenAI_embeddings, OpenAI_labels = embeddings.clone(), labels.copy()
    elif data_name == 'mnist':
        container = np.load('{}mnist_transformed.npz'.format(data_path))
        training_data_np = container['train_data'].reshape([60000,-1])
        training_labels_np = container['train_label']
        test_data_np = container['test_data'].reshape([10000,-1])
        test_labels_np = container['test_label']
        embeddings = np.concatenate([training_data_np, test_data_np],axis=0)
        labels = np.concatenate([training_labels_np, test_labels_np])
        index1 = np.where(labels==0)
        index2 = np.where(labels==1)
        index = np.concatenate([index1[0],index2[0]])
        embeddings = embeddings[index,:]
        labels = labels[index]
        embeddings = torch.tensor(embeddings).float()
        Bert_embeddings, Bert_labels = embeddings.clone(), labels.copy()
        OpenAI_embeddings, OpenAI_labels = embeddings.clone(), labels.copy()
    elif data_name == 'cifar10':
        container = np.load('{}cifar10_transformed.npz'.format(data_path))
        training_data_np = container['train_data'].reshape([50000,-1])
        training_labels_np = container['train_label']
        test_data_np = container['test_data'].reshape([10000,-1])
        test_labels_np = container['test_label']
        embeddings = np.concatenate([training_data_np, test_data_np],axis=0)
        labels = np.concatenate([training_labels_np, test_labels_np])
        index1 = np.where(labels==0)
        index2 = np.where(labels==1)
        index = np.concatenate([index1[0],index2[0]])
        embeddings = embeddings[index,:]
        labels = labels[index]
        embeddings = torch.tensor(embeddings).float()
        Bert_embeddings, Bert_labels = embeddings.clone(), labels.copy()
        OpenAI_embeddings, OpenAI_labels = embeddings.clone(), labels.copy()


    print(Bert_embeddings.shape)

    # ensure that labels from Bert embedding and OpenAI embedding matches
    assert np.linalg.norm(np.array(Bert_labels-OpenAI_labels))<1e-8, 'datasets are not matched'
    num_trial = args.num_trial

    # train and test split
    n = Bert_embeddings.shape[0]
    num_train = n//10*9

    train_str = ''
    if args.train_choice == 'f1':
        n = Bert_embeddings.shape[0]
        num_train = n//2
        train_str = '_T_f1'

    if args.debug:
        n = 4000
        num_train = 2000
        num_trial = 1

    if args.add_skip:
        skip_str = '_skip'
    else:
        skip_str = ''
        
    index = np.arange(n)
    np.random.seed(2)
    np.random.shuffle(index)
    Bert_train = Bert_embeddings[index[:num_train]].detach().numpy()
    label_train = Bert_labels[index[:num_train]]
    OpenAI_train = OpenAI_embeddings[index[:num_train]].detach().numpy()

    Bert_test = Bert_embeddings[index[num_train:n]].detach().numpy()
    label_test = Bert_labels[index[num_train:n]]
    OpenAI_test = OpenAI_embeddings[index[num_train:n]].detach().numpy()

    label_train = np.array(label_train)*2-1
    label_test = np.array(label_test)*2-1

    seed = args.seed 
    methods = ['Gaussian', 'Geometric_Algebra']
    beta_list = [1e-3,1e-4,1e-5,1e-6]
    lr_list = [1e-1,1e-2,1e-3,1e-4]
    input_num_list = [100,200,500,1000,2000,5000,10000,20000]
    if data_name == 'Amazon' or 'ECG' in data_name:
        input_num_list = [100,200,500,1000,2000,5000,10000]
    elif data_name == 'cola':
        input_num_list = [100,200,500,1000,2000]
    if args.debug:
        methods = ['Gaussian']
        beta_list = [1e-3]
        lr_list = [1e-1]
        input_num_list = [100]

    train_num_str = ''
    if args.train_num == 'f1':
        input_num_list = 200*np.arange(1,11)
        train_num_str = '_TN_f1'
    elif args.train_num == 'f2':
        input_num_list = 50*np.arange(4,11)
        train_num_str = '_TN_f2'
    elif args.train_num == 'f3':
        input_num_list = [5000]
        train_num_str = '_TN_f3'
    elif args.train_num == 'f4':
        input_num_list = [100,200,500,1000]
        train_num_str = '_TN_f4'

    # print(input_num_list)

    # np.random.seed(seed)
    set_seed(seed)

    Hidden = args.Hidden
    sdim = args.sdim
    tol = 1e-6
    solver = args.solver
    beta = 1e-3

    solver_str = ''
    if args.train_method == 'cvx' and solver == 'cvxpy':
        solver_str = '_cvxpy'
    if args.train_method == 'lasso' and args.aug_sym:
        solver_str = '_aug'

    shuffle_str = ''
    if args.shuffle:
        shuffle_str = '_shuffle'

    eps_str = ''
    if args.train_method == 'cvx' and args.add_eps:
        eps_str = '_eps{:.0e}'.format(args.eps)
    if args.train_method == 'lasso' and args.add_eps:
        eps_str = '_eps{:.0e}'.format(args.eps)

    polish_str = ''
    if args.train_method == 'noncvx' and args.polish:
        polish_str = '_polish{}'.format(args.polish_freq)
    sdim_str = ''
    if args.sdim != 100:
        sdim_str = '_sdim{}'.format(sdim)

    if data_name == 'ECG-signal':
         tol = 1e-7

    if args.embed == 'OpenAI':
        training_data_np = OpenAI_train.copy()
        training_labels_np = label_train.copy()
        test_data_np = OpenAI_test.copy()
        test_labels_np = label_test.copy()
    elif args.embed == 'Bert':
        training_data_np = Bert_train.copy()
        training_labels_np = label_train.copy()
        test_data_np = Bert_test.copy()
        test_labels_np = label_test.copy()

    # print(f"{np.max(test_labels_np)} {np.min(test_labels_np)}")

    Epochs = args.Epochs
    D_in = training_data_np.shape[1]
    batch_size = args.batch_size
    str_bundle = sdim_str + shuffle_str+ skip_str+ solver_str + eps_str + polish_str

    save_name = './results/deepNN_{}_{}_Hidden{}_L{}_{}_NT{}_seed{}.npz'.format(data_name+train_str+train_num_str,args.embed, args.Hidden, args.num_layers, str_bundle, num_trial, args.seed)
    
    result_dict = {}
    for input_num in input_num_list:

        # Create the DataLoader for our validation set
        val_data = TensorDataset(torch.tensor(test_data_np).float(), torch.tensor(test_labels_np).float())
        val_sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size, drop_last=False)

        for lr in tqdm(lr_list):
            for i in tqdm(range(num_trial)):
                index = np.arange(num_train)
                if args.shuffle:
                    np.random.shuffle(index)
                training_data_np_sub = training_data_np[index[:input_num]]
                training_labels_np_sub = training_labels_np[index[:input_num]]
                print(training_data_np_sub.shape)
                # Create the DataLoader for our training set
                train_data = TensorDataset(torch.tensor(training_data_np_sub).float(), torch.tensor(training_labels_np_sub).float())
                train_sampler = RandomSampler(train_data)
                train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, drop_last=False)
                model = utils.deepNN(D_in, args.Hidden, num_layers=args.num_layers, output_dim=1, add_skip = args.add_skip)

                # Tell PyTorch to run the model on GPU
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)


                # Create the optimizer
                optimizer = AdamW(model.parameters(),
                                lr=lr,    # Default learning rate
                                eps=1e-8,    # Default epsilon value
                                weight_decay = beta #weight decay
                                )

                # Total number of training steps
                total_steps = len(train_dataloader) * args.Epochs

                # Set up the learning rate scheduler
                scheduler = get_linear_schedule_with_warmup(optimizer,
                                                            num_warmup_steps=0, # Default value
                                                            num_training_steps=total_steps)
                cum_time, train_loss, test_loss, train_acc, test_acc = train(model, optimizer, scheduler, train_dataloader, val_dataloader, epochs=Epochs, 
                    evaluation=True, freq_batch=input_num//batch_size, polish=args.polish, polish_freq = args.polish_freq, sdim=sdim)

                key = '{}_{}_{}'.format(input_num,lr,i)
                result_dict[key] ={'cum_time': cum_time, 'train_loss': train_loss, 'test_loss': test_loss, 'train_acc': train_acc, 'test_acc': test_acc}
    np.savez(save_name, result_dict=result_dict)



if __name__=='__main__':
    main()


