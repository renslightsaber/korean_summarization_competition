import gc
import os

import argparse

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


################## summarize function ############################
def summarize(model, 
              tokenizer,
              accelerator, 
              dataloader, 
              num_beams,
              device):

    model.eval()
    bar = tqdm(enumerate(dataloader), total = len(dataloader), desc='Test Loop')
    total_sentences = []
    with torch.no_grad():
        for step, data in bar:
            ids = data['input_ids'].to(device, dtype = torch.long)
            masks = data['attention_mask'].to(device, dtype = torch.long)

        
            # ROUGE Score (num Sentences)
            generated_tokens = accelerator.unwrap_model(model).generate(input_ids = ids, 
                                                                        attention_mask = masks,
                                                                        # pad_token_id = tokenizer.pad_token_id,
                                                                        # #  max_new_tokens = 100,
                                                                        # do_sample = False,
                                                                        num_beams = num_beams,
                                                                        # num_beam_groups = 1,
                                                                        # penalty_alpha = None,
                                                                        # use_cache = True,
                                                                        # temperature = 1.0,
                                                                        min_length=30, 
                                                                        max_length=65
                                                                        )
            generated_tokens = accelerator.pad_across_processes(generated_tokens, dim=1, pad_index=tokenizer.pad_token_id)
                
            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()

            if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            total_sentences += decoded_preds
            
    print(len(total_sentences))
    print("Completed")
    return total_sentences


  
  
  
############### config = define() #########################
def define():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type = str, default = "./data/", help="Data Folder Path")
    p.add_argument('--model_save', type = str, default = "./models/", help="Trained Model Save Path")
    p.add_argument('--sub_path', type = str, default = "./submission/", help="Data Folder Path")
   
    p.add_argument('--try_title', type = str, default = "test", help="Experimental Information")
    
    p.add_argument('--model', type = str, default = "eenzeenee/t5-small-korean-summarization", help="HuggingFace Pretrained Model")    
    p.add_argument('--model_type', type = str, default = "t5", help="HuggingFace Bart or T5")
    p.add_argument('--is_lora', type = str, default = 'True', help = "LoRA Applied?")
    
    p.add_argument('--seed', type = int, default = 2023, help="Seed")
    p.add_argument('--num_beams', type = int, default = 2, help="Number of BEAMS for BEAMSEARCH")
   
    p.add_argument('--valid_batch_size', type = int, default = 16, help="Valid Batch Size")
    
    p.add_argument('--max_length', type = int, default = 512, help="Max Length")
    p.add_argument('--target_max_length', type = int, default = 65, help="Target Max Length")
    
    p.add_argument('--device', type = str, default = "cuda", help="CUDA or MPS or CPU?")

    config = p.parse_args()
    return config
  
  
############################# main  #############################
def main(config):
     
    
    #################### Data #############################
    test = pd.read_csv(config.data_path + "test.csv")
    print(test.shape)
    print(test.head(3))
    
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
    
    
    ################ Check Saved Model Path #################
    peft_path = config.model_save +  f'output_peft_dir'
    output_path = config.model_save + f'output_/'
    
    print("Saved Model Path: ")
    if config.is_lora == "True":
        print(peft_path)
    else:
        print(output_path)
    
    ############### Accelerator ###############
    accelerator = Accelerator()
    
    ############### Define Model ###############
    if config.model_type == "t5":
        
        ################# T5 Base #########################
        # model = AutoModelForSeq2SeqLM.from_pretrained(config.model).to(device)
        
        if config.is_lora == "True":
            ################### LoRA ###############################
            model = AutoModelForSeq2SeqLM.from_pretrained(config.model, load_in_8bit=True, device_map={"":0})
            print("Base Model Loaded")
            model = PeftModel.from_pretrained(model = model, model_id = peft_path, device_map={"":0})
            print("Peft LoRA Model Loaded")
        else:
            ################### Pure T5 ############################
            model = AutoModelForSeq2SeqLM.from_pretrained(output_path, local_files_only=True).to(device)
            print("Saved Model Loaded")
            model = accelerator.prepare(model)
            print("Accelerator applied")
            
    else:
        ################# Bart Base #########################
        # model = AutoModelForSeq2SeqLM.from_pretrained(config.model).to(device)
        
        if config.is_lora == "True":
            ################### LoRA ###############################
            model = AutoModelForSeq2SeqLM.from_pretrained(config.model, load_in_8bit=True, device_map={"":0})
            print("Base Model Loaded")
            model = PeftModel.from_pretrained(model = model, model_id = peft_path, device_map={"":0})
            print("Peft LoRA Model Loaded")
        else:
            ################### Pure Bart ############################
            model = AutoModelForSeq2SeqLM.from_pretrained(output_path, local_files_only=True).to(device)
            print("Saved Model Loaded")
            model = accelerator.prepare(model)
            print("Accelerator applied")
            
        
    ################# Collate_fn #################
    collate_fn= DataCollatorForSeq2Seq(tokenizer, model = model)
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
    test_sentences = summarize(model = model, 
                               tokenizer = tokenizer,
                               accelerator = accelerator, 
                               dataloader = test_loader, 
                               device = device, 
                               num_beams = config.num_beams)
    print("Summarization Completed")
    
    print()
    print("Sample?")
    print("text")
    print(test.text[2])
    print("Summary")
    print(test_sentences[2])
    print()
   

    
    ################################# Submission File #####################################
    ss = pd.read_csv(config.data_path + "sample_submission.csv")
    print("Before")
    print(ss.shape)
    ss.head()
    print()
    
    print("After")
    ss['summary'] = test_sentences
    print(ss.shape)
    ss.head()
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
    
