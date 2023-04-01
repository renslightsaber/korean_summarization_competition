import os
import gc
import time

import copy
from copy import deepcopy

import matplotlib.pyplot as plt

import wandb

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

# For colored terminal text
from colorama import Fore, Back, Style
b_ = Fore.BLUE
y_ = Fore.YELLOW
sr_ = Style.RESET_ALL

from tqdm.auto import tqdm, trange

from accelerate import Accelerator

# accelerator = Accelerator()
# model, optimizer, train_loader, valid_loader = accelerator.prepare(model, optimizer, train_loader, valid_loader)

from utils import *

############################## ROUGE Score (num Sentences) #####################################
def rouge_function(model, accelerator, tokenizer, ids, masks, targets, metric, num_sentences):

    generated_tokens = accelerator.unwrap_model(model).generate(input_ids = ids[:num_sentences], 
                                                                attention_mask = masks[:num_sentences],
                                                                # pad_token_id = tokenizer.pad_token_id,
                                                                # #  max_new_tokens = 100,
                                                                # do_sample = False,
                                                                # num_beams = 1,
                                                                # num_beam_groups = 1,
                                                                # penalty_alpha = None,
                                                                # use_cache = True,
                                                                # temperature = 1.0,
                                                                min_length=30, 
                                                                max_length=65
                                                                )
    generated_tokens = accelerator.pad_across_processes(generated_tokens, dim=1, pad_index=tokenizer.pad_token_id)

    labels = targets[:num_sentences]
    # If we did not pad to max length, we need to pad the labels too
    labels = accelerator.pad_across_processes(targets[:num_sentences], dim=1, pad_index=tokenizer.pad_token_id)
        
    generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
    labels = accelerator.gather(labels).cpu().numpy()
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]

    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    ## Rouge Scores
    rouges = metric.get_scores(decoded_preds, decoded_labels)

    return rouges
  
  

############################## train_one_epoch #####################################
def train_one_epoch(model, 
                    accelerator, 
                    dataloader, 
                    # loss_fn,
                    optimizer, 
                    device, 
                    epoch,
                    tokenizer, 
                    rouge_function, 
                    num_sentences, 
                    max_norm,
                    scheduler, 
                    grad_clipping = False):

    ################ torchmetrics: initialize metric #########################

    metric = Rouge(max_n=2, metrics = ["rouge-n", "rouge-l"] )
    
    ############################################################################

    train_loss = 0
    dataset_size = 0
    train_epoch_loss = 0

    ids_list, mask_list, targets_list = [],  [], []

    bar = tqdm(enumerate(dataloader), total = len(dataloader), desc='Train Loop')
    # bar = tqdm_notebook(enumerate(dataloader), total = len(dataloader), desc='Train Loop', leave=False)

    model.train()
    for step, data in bar:
        ids = data['input_ids'].to(device, dtype = torch.long)
        masks = data['attention_mask'].to(device, dtype = torch.long)
 
        # targets
        targets = data['labels'].to(device, dtype = torch.long)
        targets_ids = data['decoder_input_ids'].to(device, dtype = torch.long)

        # y_preds
        y_preds = model(input_ids = ids, attention_mask = masks, decoder_input_ids = targets_ids, labels=targets) 
        # instead of 'y_preds = model(**data)'
        
        # Loss
        # dims = y_preds.logits.shape[-1]
        # loss = loss_fn(y_preds.logits.view(-1, dims), targets.view(-1))
        # or Simply like below
        loss = y_preds.loss

        optimizer.zero_grad()
        # loss.backward()
        accelerator.backward(loss)
        

        # Gradient-Clipping | source: https://velog.io/@seven7724/Transformer-계열의-훈련-Tricks
        max_norm = max_norm
        if grad_clipping:
            #print("Gradient Clipping Turned On | max_norm: ", max_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()
        
        batch_size = ids.size(0)
        dataset_size += batch_size
        train_loss += float(loss.item() * batch_size) 
        train_epoch_loss = train_loss / dataset_size 

        # ROUGE SCORE
        rouges = rouge_function(model, accelerator, tokenizer, ids, masks, targets, metric, num_sentences)

        bar.set_postfix(Epoch = epoch,  
                        Train_loss = train_epoch_loss,
                        LR = optimizer.param_groups[0]['lr'], 
                        R1 = rouges['rouge-1']['f'],
                        R2 = rouges['rouge-2']['f'],
                        RL = rouges['rouge-l']['f']                
                        )
        
        
    print("Train's Epoch Loss: %.3e | (Last Batch's) R1 %.3f |  R2 %.3f |  RL %.3f |" % (train_epoch_loss, rouges['rouge-1']['f'], rouges['rouge-2']['f'], rouges['rouge-l']['f']))
    print()

    torch.cuda.empty_cache()
    _ = gc.collect()

    return train_epoch_loss, rouges
  
  
  
  

############################## valid_one_epoch #####################################
@torch.no_grad()
def valid_one_epoch(model, 
                    accelerator, 
                    dataloader, 
                    # loss_fn, 
                    # optimizer, 
                    device, 
                    epoch, 
                    tokenizer,
                    rouge_function, 
                    num_sentences
                    ):

    ################ torchmetrics: initialize metric #########################

    metric = Rouge(max_n=2, metrics = ["rouge-n", "rouge-l"] )
    
    ############################################################################

    valid_loss = 0
    dataset_size = 0
    valid_epoch_loss = 0

    ids_list, mask_list, targets_list = [],  [], []

    bar = tqdm(enumerate(dataloader), total = len(dataloader), desc='Valid Loop')
    # bar = tqdm_notebook(enumerate(dataloader), total = len(dataloader), desc='Train Loop', leave=False)

    model.eval()
    with torch.no_grad():
        for step, data in bar:
            ids = data['input_ids'].to(device, dtype = torch.long)
            masks = data['attention_mask'].to(device, dtype = torch.long)
    
            # targets
            targets = data['labels'].to(device, dtype = torch.long)
            targets_ids = data['decoder_input_ids'].to(device, dtype = torch.long)

            # y_preds
            y_preds = model(input_ids = ids, attention_mask = masks, labels=targets) 
            
            # Loss
            # dims = y_preds.logits.shape[-1]
            # loss = loss_fn(y_preds.logits.view(-1, dims), targets.view(-1))
            # or Simply like below
            loss = y_preds.loss

            batch_size = ids.size(0)
            dataset_size += batch_size
            valid_loss += float(loss.item() * batch_size) 
            valid_epoch_loss = valid_loss / dataset_size


            ids_list.append(ids)
            mask_list.append(masks)
            targets_list.append(targets)

            # ROUGE SCORE
            rouges = rouge_function(model, accelerator, tokenizer, ids, masks, targets, metric, num_sentences)

            bar.set_postfix(Epoch = epoch,  
                            Valid_loss = valid_epoch_loss,                 
                            R1 = rouges['rouge-1']['f'],
                            R2 = rouges['rouge-2']['f'],
                            RL = rouges['rouge-l']['f']
                            )
        

    print("Valid's Epoch Loss: %.3e | (Last Batch's) R1 %.3f |  R2 %.3f |  RL %.3f |" % (valid_epoch_loss, rouges['rouge-1']['f'], rouges['rouge-2']['f'], rouges['rouge-l']['f']))
    print()

    torch.cuda.empty_cache()
    _ = gc.collect()

    return valid_epoch_loss, rouges
  
  
  
  
  
################################## run_train ###########################################
def run_train(model, 
              accelerator, 
              model_save, 
              train_loader, 
              valid_loader, 
              # loss_fn, 
              optimizer, 
              device, 
              tokenizer, 
              rouge_function, 
              num_sentences, 
              max_norm,
              scheduler, 
              grad_clipping, 
              n_epochs):
    
    # To automatically log gradients
    wandb.watch(model, log_freq=100)
    
    if torch.cuda.is_available():
        print("INFO: GPU - {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict()) 
    # inference with models which is saved at best_score, or lowest_loss updated! 
    # Don't Need to save bst_model_wts like above

    lowest_epoch = np.inf
    lowest_loss = np.inf

    train_hs, valid_hs, train_r1s, valid_r1s, train_r2s, valid_r2s, train_ls, valid_ls = [], [], [], [], [], [], [], []
    
    best_score = 0
    best_score_epoch = np.inf
    best_model = None


    for epoch in range(1, n_epochs +1):
        gc.collect()

        train_epoch_loss, train_rouges = train_one_epoch(model, 
                                                         accelerator, 
                                                         train_loader, 
                                                         # loss_fn,
                                                         optimizer, 
                                                         device, 
                                                         epoch, 
                                                         tokenizer, 
                                                         rouge_function, 
                                                         num_sentences,
                                                         max_norm,
                                                         scheduler, 
                                                         grad_clipping)

        valid_epoch_loss, valid_rouges = valid_one_epoch(model, 
                                                         accelerator, 
                                                         valid_loader, 
                                                         # loss_fn,
                                                         # optimizer, 
                                                         device, 
                                                         epoch, 
                                                         tokenizer, 
                                                         rouge_function, 
                                                         num_sentences)

        ## getget
        train_hs.append(train_epoch_loss)
        valid_hs.append(valid_epoch_loss)

        train_r1s.append(train_rouges['rouge-1']['f'])
        valid_r1s.append(valid_rouges['rouge-1']['f'])

        train_r2s.append(train_rouges['rouge-2']['f'])
        valid_r2s.append(valid_rouges['rouge-2']['f'])

        train_ls.append(train_rouges['rouge-l']['f'])
        valid_ls.append(valid_rouges['rouge-l']['f'])

        # Log the metrics
        wandb.log({"train/loss": train_epoch_loss})
        wandb.log({"eval/loss": valid_epoch_loss})

        # Log the metrics
        wandb.log({"train/R1": train_rouges['rouge-1']['f']})
        wandb.log({"eval/R1": valid_rouges['rouge-1']['f']})

        # Log the metrics
        wandb.log({"train/R2": train_rouges['rouge-2']['f']})
        wandb.log({"eval/R2": valid_rouges['rouge-2']['f']})

        # Log the metrics
        wandb.log({"train/RL": train_rouges['rouge-l']['f']})
        wandb.log({"eval/RL": valid_rouges['rouge-l']['f']})

        print()
        print(f"Epoch:{epoch:02d} | TL:{train_epoch_loss:.3e} | VL:{valid_epoch_loss:.3e}")
        print(f"Train's R1: {train_rouges['rouge-1']['f']:.2f} | Valid's R1: {valid_rouges['rouge-1']['f']:.2f} |")
        print(f"Train's R2: {train_rouges['rouge-2']['f']:.2f} | Valid's R2: {valid_rouges['rouge-2']['f']:.2f} |")
        print(f"Train's RL: {train_rouges['rouge-l']['f']:.2f} | Valid's RL: {valid_rouges['rouge-l']['f']:.2f} |")
        print()

        if valid_epoch_loss < lowest_loss:
            print(f"{b_}Validation Loss Improved({lowest_loss:.3e}) --> ({valid_epoch_loss:.3e})")
            lowest_loss = valid_epoch_loss
            lowest_epoch = epoch
            best_model_wts1 = copy.deepcopy(model.state_dict())
            # PATH = model_save + f"model.bin"
            # torch.save(model.state_dict(), PATH)
            PEFT_MODEL_PATH1 = model_save + "peft_loss/"
            model.save_pretrained(PEFT_MODEL_PATH1)
            # PEFT_MODEL_PATH = f'/content/drive/MyDrive/GPT_Competition/exp_{TIME_SERIAL}'
            # peft_model.save_pretrained(PEFT_MODEL_PATH)
            print(f"Better Loss Model Saved{sr_}")

        valid_r1= valid_rouges['rouge-1']['f']
        
        if best_score < valid_r1:
            print(f"{b_}R1 Improved({best_score:.3f}) --> ({valid_r1:.3f})")
            best_score = valid_r1
            best_score_epoch = epoch
            best_model_wts2 = copy.deepcopy(model.state_dict())
            # PATH2 = model_save + f"model_r1.bin"
            # torch.save(model.state_dict(), PATH2)
            PEFT_MODEL_PATH2 = model_save + "peft_r1/"
            model.save_pretrained(PEFT_MODEL_PATH2)
            # PEFT_MODEL_PATH = f'/content/drive/MyDrive/GPT_Competition/exp_{TIME_SERIAL}'
            # peft_model.save_pretrained(PEFT_MODEL_PATH)
            print(f"Better_R1_Model Saved{sr_}")
        print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Loss : %.4e at %d th Epoch" % (lowest_loss, lowest_epoch))
    print("Best F1(W): %.4f at %d th Epoch" % (best_score, best_score_epoch))

    # load best model weights
    # model_a = model.load_state_dict(best_model_wts1)
    # model_b = model.load_state_dict(best_model_wts2)

    result = dict()
    result["train/loss"] = train_hs
    result["eval/loss"] = valid_hs

    result["train/R1"] = train_r1s
    result["eval/R1"] = valid_r1s

    result["train/R2"] = train_r2s
    result["eval/R2"] = valid_r2s

    result["train/RL"] = train_ls
    result["eval/RL"] = valid_ls

    # plot
    make_plot(result, stage = "loss")
    make_plot(result, stage = "R1")
    make_plot(result, stage = "R2")
    make_plot(result, stage = "RL")

    del result, train_hs, valid_hs, train_r1s, valid_r1s, train_r2s, valid_r2s, train_ls, valid_ls

    torch.cuda.empty_cache()
    _ = gc.collect()

    # return model
  
