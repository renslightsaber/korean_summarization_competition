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
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments

from accelerate import Accelerator

# About tqdm: https://github.com/tqdm/tqdm/#ipython-jupyter-integration
from tqdm.auto import tqdm, trange
# from tqdm.notebook import trange

# HuggingFace peft 라이브러리
# from peft import get_peft_model, PeftModel, TaskType, LoraConfig

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from utils import *
    
############### config = define() #########################
def define():
    p = argparse.ArgumentParser()

    p.add_argument('--base_path', type = str, default = "./data/", help="Base Path")    
    p.add_argument('--data_path', type = str, default = "./data/", help="Data Folder Path")
    p.add_argument('--model_save', type = str, default = "./models/", help="Data Folder Path")
    p.add_argument('--sub_path', type = str, default = "./submission/", help="Data Folder Path")
   
    p.add_argument('--ratio', type = float, default = 0.95, help="Percentage of data to train")
    p.add_argument('--try', type = str, default = "T17", help="Experimental Information")
    
    p.add_argument('--model', type = str, default = "kakaobrain/kogpt", help="HuggingFace Pretrained Model")    
    # p.add_argument('--model_type', type = str, default = "AutoModelForSequenceClassification", help="HuggingFace Pretrained Model")
    
    p.add_argument('--seed', type = int, default = 2023, help="Seed")
    p.add_argument('--n_epochs', type = int, default = 8, help="Epochs")
    
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
    
    ## Data
    train = pd.read_csv(base_path = config.data_path + "train.csv")
    print(train.shape)
    print(train.head(2))
    
    ## Set Seed
    set_seed(config.seed)
    
    ## Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    print("Tokenizer Downloaded")

    # Device
    if config.device == "mps":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        
    elif config.device == "cuda":
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        
    else:
        device = torch.device("cpu")

    print("Device", device)
    
    ## prepare_ds
    index_num = int(train.shape[0] * config.ratio)
    print(index_num)

    train_df = train[: index_num].reset_index(drop = True)
    valid_df = train[index_num: ].reset_index(drop = True)

    ## train, valid -> Dataset
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
    
#     # Define Model 
#     if config.model_type == "t5":
#         model = AutoModelForSequenceClassification.from_pretrained(config.model, 
#                                                                    num_labels = 3).to(device)
#     else:
#         model = AutoModelForSeq2SeqLM.from_pretrained(config.model).to(device)
        
    ## Define Model
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model).to(device)
        
    ## Collate_fn
    collate_fn= DataCollatorForSeq2Seq(tokenizer, model = model)
    
    # Define Opimizer and Scheduler
    optimizer = AdamW(model.parameters(), 
                      lr = config['learning_rate'], 
                      weight_decay = config['weight_decay'])
    print("Optimizer Defined")

    # scheduler = fetch_scheduler(optimizer)
    lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=100,  max_iters=2000)
    print("LR Scheduler Loaded")
    
    ## Accelerator
    # from accelerate import Accelerator
    accelerator = Accelerator()
    model, optimizer = accelerator.prepare(model, optimizer)
    print("Accelerator applied")
    

    ################# compute metrics for huggingface #####################      
    def compute_metrics(eval_pred):

        predictions, labels = eval_pred

        # Rouge Metric instance
        metric = Rouge(max_n=2, metrics = ["rouge-n", "rouge-l"] )

        # Decode generated summaries into text
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # Decode reference summaries into text
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # ROUGE expects a newline after each sentence
        # decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
        # decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]

        # Compute ROUGE scores
        rouges = metric.get_scores(decoded_preds, decoded_labels)

        return {"R1": rouges['rouge-1']['f'], "R2": rouges['rouge-2']['f'], "RL": rouges['rouge-l']['f']}

        # Extract the median scores
        # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        # return {k: round(v, 4) for k, v in result.items()}  

    ## training_args
    training_args = Seq2SeqTrainingArguments(
                        output_dir = config.model_save + f'output_/', # config['base_path'] + f'output_{fold}/', 
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

      
    ## Trainer
    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset = train_ds,
        eval_dataset = valid_ds,
        data_collator = collate_fn,
        tokenizer = tokenizer,
        optimizers = (optimizer, lr_scheduler), 
        compute_metrics = compute_metrics)
    
    run = wandb.init(project='Korean_Summarization', 
                      config=config,
                      job_type='Train',
                      # group=group_name,
                      # tags=[config['model'], f'{HASH_NAME}'],
                      # name=f'{HASH_NAME}-fold-{fold}',
                       name = config['try'] + f"_hf_Trainer",
                       anonymous='must')
    
    ## Let's Train!
    trainer.train()
    
    ## Save Best Model
    trainer.save_model()

    ## wandb.finish()
    run.finish()

    torch.cuda.empty_cache()
    _ = gc.collect()

    print("Train Completed")

if __name__ == '__main__':
    config = define()
    main(config)
    
