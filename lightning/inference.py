import gc
import os

import argparse
import ast

import numpy as np
import pandas as pd

## Pytorch Import
import torch 
import torch.nn as nn

from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

## Transforemr Import
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM

# Accelerate
from accelerate import Accelerator

# import Pytorch Lightning 2.0 
import lightning as L
from lightning.fabric import Fabric

# huggingFace/peft 
from peft import PeftModel

# About tqdm: https://github.com/tqdm/tqdm/#ipython-jupyter-integration
from tqdm.auto import tqdm, trange
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
   
    p.add_argument('--try_title', type = str, default = "test", help="Experimental Information")
    
    p.add_argument('--model', type = str, default = "eenzeenee/t5-small-korean-summarization", help="HuggingFace Pretrained Model")    
    
    # p.add_argument('--is_lora', type = str, default = 'True', help = "LoRA Applied?")
    true_false_list = ['true', 'yes', "1", 't','y']
    p.add_argument("--is_lora", type= lambda s : s.lower() in true_false_list, required=False, default=True, 
                   help="LoRA or Not : True or False (e.g true,y, 1 | false, n, 0)")
    
    p.add_argument('--lora_r', type = int, default = 4, help="Max Length")
    p.add_argument('--lora_alpha', type = int, default = 32, help="Max Length")
    p.add_argument('--lora_target_modules', type = str, default = "['q', 'v']", help="List of Nodes")
    p.add_argument('--lora_dropout_p', type = float, default = 0.05, help="Min LR")
    
    p.add_argument('--seed', type = int, default = 2023, help="Seed")
    p.add_argument('--num_beams', type = int, default = 2, help="Number of BEAMS for BEAMSEARCH")
   
    p.add_argument('--valid_batch_size', type = int, default = 16, help="Valid Batch Size")
    
    p.add_argument('--max_length', type = int, default = 512, help="Max Length")
    p.add_argument('--target_max_length', type = int, default = 65, help="Target Max Length")
    
    p.add_argument('--num_sentences', type = int, default = 4, help="Number of Senternces for infer during training")
    
    p.add_argument('--learning_rate', type = float, default = 1e-5, help="lr")
    p.add_argument('--weight_decay', type = float, default = 1e-6, help="Weight Decay")
    
    # p.add_argument('--device', type = str, default = "cuda", help="CUDA or MPS or CPU?")

    config = p.parse_args()
    return config
  
  
############################# main  #############################
def main(config):
     
    
    #################### Data #############################
    test = pd.read_csv(config.data_path + "test.csv")
    print(test.shape)
    print(test.head(3))
    
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
   
  
    
    ############### Load Saved Model ###############
    base_model = LitModel(**model_kwargs).to(device)
    ## Load
    base_model = base_model.load_from_checkpoint(check_path, **model_kwargs)
    print("Model Loaded")
            
        
    ################# Collate_fn #################
    collate_fn = DataCollatorForSeq2Seq(tokenizer, model = base_model.model)
    print("Collate_function Defined")
    
    
    ######################### prepare_ds #################################
    test_ds = MyDataset(test, 
                        tokenizer = tokenizer, 
                        max_length = config.max_length, 
                        target_max_length = config.target_max_length, 
                        mode = "test")

    test_loader = DataLoader(test_ds, 
                             batch_size = config.valid_batch_size, 
                             collate_fn=collate_fn, 
                             num_workers = 2,  
                             shuffle = False, 
                             pin_memory = True, 
                             drop_last= False)
    print("Test Loader Completed")
    
    
    
    ############################# Summarize (Inference) ##################################
    print("Beams: ", config.num_beams)
    test_sentences = summarize(model = base_model, 
                               dataloader = test_loader, 
                               device = device, 
                               num_beams = config.num_beams)
    print("Summarization Completed")
    
    print()
    print("Sample?")
    print("text")
    print(test.text[2])
    print()
    print("Summary")
    print(test_sentences[2])
    print()
   

    
    ################################# Submission File #####################################
    ss = pd.read_csv(config.data_path + "sample_submission.csv")
    print("Before")
    print(ss.shape)
    print(ss.head(3))
    print()
    
    print("After")
    ss['summary'] = test_sentences
    print(ss.shape)
    print(ss.head(3))
    print()
    
    print(ss.info()) # for check 'null'
    print()
    
    ################# Let's Save #######################
    save_name = config.try_title + f"_beams_{config.num_beams}" + "_hf_submission.csv"
    print(save_name)
    print()
    
    ss.to_csv(config.sub_path + save_name, index= False)
    print("Submission File Saved")
    print()
    
    torch.cuda.empty_cache()
    _ = gc.collect()

    print("Inference Completed")
    
if __name__ == '__main__':
    config = define()
    main(config)
    
