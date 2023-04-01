import gc
import time
import random
import string
import os
import re
import platform
import itertools
import collections
import pkg_resources  # pip install py-rouge
from io import open

import copy
from copy import deepcopy

import wandb

import argparse
import ast

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from konlpy.tag import Mecab

## Pytorch Import
import torch 
import torch.nn as nn

from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

## Transforemr Import
from transformers import AutoTokenizer, AdamW, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM

# Accelerate
from accelerate import Accelerator

# About tqdm: https://github.com/tqdm/tqdm/#ipython-jupyter-integration
from tqdm.auto import tqdm, trange
# from tqdm.notebook import trange

# import Pytorch Lightning 2.0 
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.fabric import Fabric
from lightning.pytorch.callbacks import ModelCheckpoint

# huggingFace/peft 
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from utils import *
from model import *


    
############### config = define() #########################
def define():
    p = argparse.ArgumentParser()

    p.add_argument('--data_path', type = str, default = "./data/", help="Data Folder Path")
    p.add_argument('--model_save', type = str, default = "./models/", help="Trained Model Save Path")
    p.add_argument('--sub_path', type = str, default = "./submission/", help="Data Folder Path")
    
    true_false_list = ['true', 'yes', "1", 't','y']
    p.add_argument("--is_sample", type= lambda s : s.lower() in true_false_list, required=False, default=True, 
                   help="Sample or Not : True or False (e.g true,y, 1 | false, n, 0)")
    p.add_argument('--sample', type = int, default = 1000, help="Number of Rows of train.csv")
    p.add_argument('--ratio', type = float, default = 0.8, help="Percentage of data to train")
    p.add_argument('--try_title', type = str, default = "test", help="Experimental Information")
    
    p.add_argument('--model', type = str, default = "eenzeenee/t5-small-korean-summarization", help="HuggingFace Pretrained Model")    

    # p.add_argument('--is_lora', type = str, default = 'True', help = "LoRA Applied?")
    p.add_argument("--is_lora", type= lambda s : s.lower() in true_false_list, required=False, default=True, 
                   help="LoRA or Not : True or False (e.g true,y, 1 | false, n, 0)")
    p.add_argument('--lora_r', type = int, default = 4, help="Max Length")
    p.add_argument('--lora_alpha', type = int, default = 32, help="Max Length")
    p.add_argument('--lora_target_modules', type = str, default = "['q', 'v']", help="List of Nodes")
    p.add_argument('--lora_dropout_p', type = float, default = 0.05, help="Min LR")
    
    p.add_argument("--is_compiled", type= lambda s : s.lower() in true_false_list, required=False, default=True, 
                   help="torch.compile() or NOT : True or False (e.g true,y, 1 | false, n, 0)")
      
    p.add_argument('--compiled_mode', type = str, default = "default", help="torch.compile() MODE")
    
    p.add_argument('--seed', type = int, default = 2023, help="Seed")
    p.add_argument('--n_epochs', type = int, default = 3, help="Epochs")
    
    p.add_argument('--num_sentences', type = int, default = 4, help="Number of Senternces for infer during training")
    
    p.add_argument('--train_batch_size', type = int, default = 16, help="Train Batch Size")
    p.add_argument('--valid_batch_size', type = int, default = 16, help="Valid Batch Size")
    
    p.add_argument('--max_length', type = int, default = 512, help="Max Length")
    p.add_argument('--target_max_length', type = int, default = 65, help="Target Max Length")
    
    p.add_argument('--T_max', type = int, default = 500, help="T_max")
    p.add_argument('--learning_rate', type = float, default = 1e-5, help="lr")
    p.add_argument('--min_lr', type = float, default = 1e-6, help="Min LR")
    p.add_argument('--weight_decay', type = float, default = 1e-6, help="Weight Decay")
    
    p.add_argument('--n_accumulate', type = int, default = 1, help="n_accumulate")
    # p.add_argument('--device', type = str, default = "cuda", help="CUDA or MPS or CPU?") ## auto by Fabric

    config = p.parse_args()
    return config
  

############# main ##################
def main(config):

  
    #################### Data #############################
    if config.is_sample:
        train = pd.read_csv(config.data_path + "train.csv")
        train = train[:config.sample].reset_index(drop= True)
    else:
        train = pd.read_csv(config.data_path + "train.csv")
    print(train.shape)
    print(train.head(3))
    
    ##################### Set Seed ###########################
    L.seed_everything(config.seed)
    print("Seed Fixed!")
        
    
    ###################### Tokenizer ################################
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    print("Tokenizer Downloaded")
        
        
    ########################## Device #################################
    fabric = Fabric()
    device = fabric.device
    device

    print("Device", device)
    
    
    ############### Model's ###############
    model_kwargs = {'model_name' : config.model, 
                    'is_lora': config.is_lora, 
                    'lora_r': config.lora_r,
                    'lora_alpha': config.lora_alpha,
                    'lora_target_modules': ast.literal_eval(config.lora_target_modules),
                    'lora_dropout_p': config.lora_dropout_p,
                    'num_sentences': config.num_sentences,
                    'lr': config.learning_rate,
                    'wd': config.weight_decay
                    }
    print("MODEL's  kwargs: ")
    print(model_kwargs)
    
    
    ############### get_lightning_train_model ###############
    compiled_model = get_lightning_train_model(model_kwargs = model_kwargs,
                                              is_compiled = config.is_compiled, 
                                              mode = config.compiled_mode)
    print("GOT MODEL")
            
        
    ################# Collate_fn #################
    collate_fn= DataCollatorForSeq2Seq(tokenizer, model = compiled_model.model)
    print("Collate_function Defined")
    
    ############## prepare_loader 실행 ########################
    train_loader, valid_loader = prepare_loader(train = train, 
                                                ratio = config.ratio, 
                                                tokenizer = tokenizer, 
                                                max_length = config.max_length, 
                                                target_max_length = config.target_max_length,
                                                train_bs = config.train_batch_size,
                                                valid_bs = config.valid_batch_size,
                                                collate_fn = collate_fn
                                               )
    
    ################### Batch Size 확인 ####################
    data = next(iter(valid_loader))
    print(data['input_ids'].shape, data['labels'].shape)
    
    
    ################### wandb ######################
    wandb_logger = WandbLogger(project='Korean_Summarization', 
                              config=config,
                              job_type='Train',
                              name= f"Lightning_2.0_"+ config.try_title,
                              anonymous='must')
    
    ################### MODEL CHECKPOINT ##########################
    checkpoint_callback = ModelCheckpoint(monitor= "eval/R1",
                                        # filename=f"{config.model.replace('/', '_')}_fold_{fold}", # 이런식으로 지정하면 Epoch 넘버와 Valid Loss까지 찍혀서 모델이름으로 저장됩니다. 
                                        filename = "best_model",
                                        dirpath = config.model_save + 'output/',
                                        mode= 'max',
                                        save_top_k = 1,
                                        # save_top_k = -1,
                                        # every_n_epochs = 1,
                                        save_on_train_epoch_end = True,
                                        save_weights_only=True
                                        )
    
    
    ############################# Trainer ###################################
    trainer = L.Trainer(accelerator = "auto", 
                        devices = -1, 
                        max_epochs= config.n_epochs,
                        logger = wandb_logger,
                        strategy = config.strategy,
                        callbacks=[checkpoint_callback])
    
    
    ############# Let's Train! ##################
    trainer.fit(compiled_model, 
                train_dataloaders = train_loader, 
                val_dataloaders = valid_loader,)
    
    
    ############ wandb.finish() #################
    wandb.finish()
    
    torch.cuda.empty_cache()
    _ = gc.collect()

    print("Train Completed")

if __name__ == '__main__':
    config = define()
    main(config)
  
  
  
