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

## Pytorch Import
import torch 
import torch.nn as nn

from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

## Transforemr Import
from transformers import AutoTokenizer, AdamW, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# Accelerate
from accelerate import Accelerator

# About tqdm: https://github.com/tqdm/tqdm/#ipython-jupyter-integration
from tqdm.auto import tqdm, trange
# from tqdm.notebook import trange

# huggingFace/peft 
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from utils import *
    
############### config = define() #########################
def define():
    p = argparse.ArgumentParser()

    # p.add_argument('--base_path', type = str, default = "./data/", help="Base Path")    
    p.add_argument('--data_path', type = str, default = "./data/", help="Data Folder Path")
    p.add_argument('--model_save', type = str, default = "./models/", help="Trained Model Save Path")
    # p.add_argument('--peft_save', type = str, default = "./models/", help="Trained Peft Model Save  Path")
    p.add_argument('--sub_path', type = str, default = "./submission/", help="Data Folder Path")
   
    p.add_argument('--sample', type = int, default = 1000, help="Number of Rows of train.csv")
    p.add_argument('--ratio', type = float, default = 0.8, help="Percentage of data to train")
    p.add_argument('--try_title', type = str, default = "test", help="Experimental Information")
    
    p.add_argument('--model', type = str, default = "eenzeenee/t5-small-korean-summarization", help="HuggingFace Pretrained Model")    
    p.add_argument('--model_type', type = str, default = "t5", help="HuggingFace Bart or T5")
    p.add_argument('--is_lora', type = bool, default = True, help = "LoRA Applied?")
    p.add_argument('--lora_r', type = int, default = 4, help="Max Length")
    p.add_argument('--lora_alpha', type = int, default = 32, help="Max Length")
    p.add_argument('--lora_target_modules', type = str, default = "['q', 'v']", help="List of Nodes")
    p.add_argument('--lora_dropout_p', type = float, default = 0.05, help="Min LR")
    
    p.add_argument('--seed', type = int, default = 2023, help="Seed")
    p.add_argument('--n_epochs', type = int, default = 3, help="Epochs")
    
    p.add_argument('--train_batch_size', type = int, default = 16, help="Train Batch Size")
    p.add_argument('--valid_batch_size', type = int, default = 16, help="Valid Batch Size")
    
    p.add_argument('--max_length', type = int, default = 512, help="Max Length")
    p.add_argument('--target_max_length', type = int, default = 65, help="Target Max Length")
    
    p.add_argument('--T_max', type = int, default = 500, help="T_max")
    p.add_argument('--learning_rate', type = float, default = 1e-5, help="lr")
    p.add_argument('--min_lr', type = float, default = 1e-6, help="Min LR")
    p.add_argument('--weight_decay', type = float, default = 1e-6, help="Weight Decay")

    p.add_argument('--n_accumulate', type = int, default = 1, help="n_accumulate")
    p.add_argument('--max_grad_norm', type = int, default = 1000, help="max_grad_norm")
    p.add_argument('--grad_clipping', type = bool, default = False, help="Gradient Clipping")
    p.add_argument('--device', type = str, default = "cuda", help="CUDA or MPS or CPU?")

    config = p.parse_args()
    return config
  

############# main ##################
def main(config):
    
    #################### Data #############################
    if config.sample:
        train = pd.read_csv(config.data_path + "train.csv")
        train = train[:config.sample].reset_index(drop= True)
    else:
        train = pd.read_csv(config.data_path + "train.csv")
    print(train.shape)
    print(train.head(3))
    
    ##################### Set Seed ###########################
    set_seed(config.seed)
    print("Seed Fixed!")
    
    ###################### Tokenizer ################################
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    print("Tokenizer Downloaded")

    ########################## Device #################################
    if config.device == "mps":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        
    elif config.device == "cuda":
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        
    else:
        device = torch.device("cpu")

    print("Device", device)
    
    ######################### prepare_ds #################################
    index_num = int(train.shape[0] * config.ratio)
    print(index_num)

    train_df = train[: index_num].reset_index(drop = True)
    valid_df = train[index_num: ].reset_index(drop = True)

    #################### train, valid -> Dataset ###########################
    train_ds = MyDataset(train_df, 
                        tokenizer = tokenizer ,
                        max_length =  config.max_length,
                        target_max_length = config.target_max_length,
                        mode = "train")

    valid_ds = MyDataset(valid_df, 
                        tokenizer = tokenizer ,
                        max_length =  config.max_length,
                        target_max_length = config.target_max_length,
                        mode = "train")
    print("Dataset Loaded")
    
    
    ############### Accelerator ###############
    # from accelerate import Accelerator
    accelerator = Accelerator()
    
    ############### Define Model ###############
    if config.model_type == "t5":
        
        ################# T5 Base #########################
        model = AutoModelForSeq2SeqLM.from_pretrained(config.model).to(device)
        
        if config.is_lora:
            ################### LoRA ###############################
            lora_config = LoraConfig(r= config.lora_r,
                                    lora_alpha= config.lora_alpha,
                                    target_modules= ast.literal_eval(config.lora_target_modules),
                                    lora_dropout=config.lora_dropout_p,
                                    bias="none", 
                                    task_type=TaskType.SEQ_2_SEQ_LM)
            
            print("Int 8 model for training")
            model = prepare_model_for_int8_training(model)
            
            print("LoRA Adaptor Added")
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
        else:
            ################### Pure T5 ############################
            model, optimizer = accelerator.prepare(model, optimizer)
            print("Accelerator applied")
            
    else:
        ################# Bart Base #########################
        model = AutoModelForSeq2SeqLM.from_pretrained(config.model).to(device)
        
        if config.is_lora:
            ################### LoRA ###############################
            lora_config = LoraConfig(r= config.lora_r,
                                    lora_alpha= config.lora_alpha,
                                    target_modules=ast.literal_eval(config.lora_target_modules),
                                    lora_dropout=config.lora_dropout_p,
                                    bias="none", 
                                    task_type=TaskType.SEQ_2_SEQ_LM)
            
            print("Int 8 model for training")
            model = prepare_model_for_int8_training(model)
            
            print("LoRA Adaptor Added")
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
        else:
            ################### Pure T5 ############################
            model, optimizer = accelerator.prepare(model, optimizer)
            print("Accelerator applied")
            
        
    ################# Collate_fn #################
    collate_fn= DataCollatorForSeq2Seq(tokenizer, model = model)
    print("Collate_function Defined")
    
    
    ################## Define Opimizer and Scheduler ##########################
    optimizer = AdamW(model.parameters(), 
                      lr = config.learning_rate, 
                      weight_decay = config.weight_decay
                     )
    print("Optimizer Defined")
    
    
    ############### scheduler = fetch_scheduler(optimizer) #####################
    lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, 
                                         warmup=100,  
                                         max_iters=2000
                                        )
    print("LR Scheduler Loaded")
    
    
    ############### Trained Model Path ############
    output_path = config.model_save + f'output_/'
    print(output_path)
    
    ################# training_args #################
    training_args = Seq2SeqTrainingArguments(
                        output_dir = output_path,
                        evaluation_strategy = 'epoch', 
                        per_device_train_batch_size = config.train_batch_size, 
                        per_device_eval_batch_size = config.valid_batch_size,
                        num_train_epochs= config.n_epochs, 
                        learning_rate = config.learning_rate,
                        weight_decay = config.weight_decay, 
                        gradient_accumulation_steps = config.n_accumulate,
                        max_grad_norm = config.max_grad_norm,
                        seed= config.seed,
                        predict_with_generate=True,
                        # group_by_length = True,
                        metric_for_best_model = 'R1', 
                        load_best_model_at_end = False,  # https://discuss.huggingface.co/t/save-only-best-model-in-trainer/8442/4      
                        greater_is_better = True, # https://huggingface.co/transformers/v3.5.1/main_classes/trainer.html
                        save_strategy="epoch",
                        save_total_limit = 1, 
                        report_to="wandb",
                        )

    ################# Trainer #################
    trainer = Seq2SeqTrainer(model,
                            training_args,
                            train_dataset = train_ds,
                            eval_dataset = valid_ds,
                            data_collator = collate_fn,
                            tokenizer = tokenizer,
                            optimizers = (optimizer, lr_scheduler), 
                            compute_metrics = compute_metrics)
    
    ################### wandb ######################
    run = wandb.init(project='Korean_Summarization', 
                      config=config,
                      job_type='Train',
                      # group=group_name,
                      # tags=[config['model'], f'{HASH_NAME}'],
                      # name=f'{HASH_NAME}-fold-{fold}',
                       name = config.try_title + f"_hf_Trainer",
                       anonymous='must')
    
    ############# Let's Train! ##################
    trainer.train()
    
    ############ Save Best Model ################
    # trainer.save_model()
    
    ############ Save Best PEFT LORA Model ################
    if config.peft_save:
        peft_path = config.model_save +  f'output_peft_dir'
        trainer.model.save_pretrained(peft_path)
        print("Peft LoRA Model Saved")
        tokenizer.save_pretrained(peft_path)
        print("Tokenizer Saved")
    else:
        print(output_path)
        trainer.save_model(output_path)
        print("Peft Model Saved")
        tokenizer.save_pretrained(output_path)
        print("Tokenizer Saved")

    ############ wandb.finish() #################
    run.finish()
    
    torch.cuda.empty_cache()
    _ = gc.collect()

    print("Train Completed")

if __name__ == '__main__':
    config = define()
    main(config)
    
