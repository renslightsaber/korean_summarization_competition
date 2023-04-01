
# import Pytorch Lightning 2.0 
import lightning as L

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

## Accerate
from accelerate import Accelerator

# About tqdm: https://github.com/tqdm/tqdm/#ipython-jupyter-integration
from tqdm.auto import tqdm, trange

# HuggingFace peft 라이브러리
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

from utils import *

# Mecab
from konlpy.tag import Mecab



########################## get_lightning_train_model function #########################################
def get_lightning_train_model(model_kwargs,
                              is_compiled,
                              mode, 
                              device):
    
    base_model = LitModel(**model_kwargs).to(device)

    if is_compiled:
        #### Just Inform
        print(f"model_name: {model_kwargs['model_name']}")
        print(f"is_compiled?: {is_compiled} | MODE: {mode}")
        #### Now torch.compile(l_model)
        return torch.compile(base_model, mode = mode, fullgraph=False)
    else:
        #### Not Compiled
        print("Not Compiled")
        print(f"model_name: {model_kwargs['model_name']}")
        return base_model


####################### LightningModule #################################
class LitModel(L.LightningModule):
    # Instantiate the model
    def __init__(self, 
                 model_name, 
                 is_lora= True, 
                 lora_r = 4, 
                 lora_alpha = 32, 
                 lora_target_modules = ["q", "v"], 
                 lora_dropout_p = 0.05, 
                 num_sentences = 4, 
                 lr = 2e-4, 
                 wd = 1e-6):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.accelerator = Accelerator()
        # self.fabric = Fabric()
        self.model = self.define_model(model_name, is_lora, lora_r, lora_alpha, lora_target_modules, lora_dropout_p)
        self.num_sentences = num_sentences
        self.wd = wd
        self.lr = lr
        self.metric = Rouge(max_n=2, metrics = ["rouge-n", "rouge-l"] )

    def forward(self, input_ids, attention_mask, decoder_input_ids, labels):
        return self.model(input_ids, attention_mask, decoder_input_ids, labels)

    def define_model(self, model_name, is_lora, lora_r, lora_alpha, lora_target_modules, lora_dropout_p):

        based_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if is_lora:
            lora_config = LoraConfig(r = lora_r,
                                    lora_alpha = lora_alpha,
                                    target_modules = lora_target_modules,
                                    lora_dropout = lora_dropout_p,
                                    bias = "none",
                                    task_type = TaskType.SEQ_2_SEQ_LM
                                    )
            # prepare int-8 model for training
            print("Int 8 model for training")
            peft_model = prepare_model_for_int8_training(based_model)
            # add LoRA adaptor
            print("LoRA Adaptor Added")
            peft_model = get_peft_model(peft_model, lora_config)
            peft_model.print_trainable_parameters()
            return peft_model
          
        else:
          
            return based_model
            
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr = self.lr, weight_decay = self.wd)
        scheduler = CosineWarmupScheduler(optimizer = optimizer, warmup=100, max_iters=2000)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

      
    ############### ROUGE Function during Training ###################
    def rouge_function(self, ids, masks, targets):
        generated_tokens = self.accelerator.unwrap_model(self.model).generate(input_ids = ids[:self.num_sentences], 
                                                                    attention_mask = masks[:self.num_sentences],
                                                                    # pad_token_id = self.tokenizer.pad_token_id,
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
        generated_tokens = self.accelerator.pad_across_processes(generated_tokens, dim=1, pad_index=self.tokenizer.pad_token_id)

        labels = targets[:self.num_sentences]
        # If we did not pad to max length, we need to pad the labels too
        labels = self.accelerator.pad_across_processes(targets[:self.num_sentences], dim=1, pad_index=self.tokenizer.pad_token_id)
            
        generated_tokens = self.accelerator.gather(generated_tokens).cpu().numpy()
        labels = self.accelerator.gather(labels).cpu().numpy()
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        ## Rouge Scores
        rouges = self.metric.get_scores(decoded_preds, decoded_labels)

        return rouges

      
    ############### SHARED_STEP ###################
    def shared_step(self, batch, stage):
        # Load the data into variables
        src_ids, src_mask, tgt_ids, decoder_input_ids = batch['input_ids'], batch['attention_mask'], batch['labels'], batch['decoder_input_ids']

        # Run the model and get the logits
        outputs = self.model(input_ids = src_ids, 
                             attention_mask=src_mask, 
                             decoder_input_ids=decoder_input_ids, 
                             labels = tgt_ids)
        loss = outputs.loss

        if stage == "eval":

            ## ROUGE SCORE
            rouges = self.rouge_function(src_ids, src_mask, tgt_ids)

            ## logging
            self.log(f'{stage}/loss', loss, on_epoch=True, on_step=True, prog_bar=True)
            self.log(f'{stage}/R1', rouges['rouge-1']['f'], on_epoch=True, on_step=True, prog_bar=True)
            self.log(f'{stage}/R2', rouges['rouge-2']['f'], on_epoch=True, on_step=True, prog_bar=True)
            self.log(f'{stage}/RL', rouges['rouge-l']['f'] , on_epoch=True, on_step=True, prog_bar=True)
            
            ## Return
            return {f'loss': loss, 
                    f'R1': rouges['rouge-1']['f'],
                    f'R2': rouges['rouge-2']['f'],
                    f'RL': rouges['rouge-l']['f']}
        else:
            ## train

            ## logging
            self.log(f'{stage}/loss', loss, on_epoch=True, on_step=True, prog_bar=True)

            ## Return
            return {f'loss': loss}


    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "eval")
  
    ############### generate text ###################
    def generate_text(self, ids, masks, eval_beams, min_len = 30, max_len = 70):

        # ROUGE Score (num Sentences)
        generated_tokens = self.accelerator.unwrap_model(self.model).generate(input_ids = ids, 
                                                                                attention_mask = masks,
                                                                                # pad_token_id = tokenizer.pad_token_id,
                                                                                # #  max_new_tokens = 100,
                                                                                # do_sample = False,
                                                                                num_beams = eval_beams,
                                                                                # num_beam_groups = 3,
                                                                                # penalty_alpha = None,
                                                                                # use_cache = True,
                                                                                # temperature = 1.0,
                                                                                min_length= min_len, 
                                                                                max_length= max_len)
        
        generated_tokens = self.accelerator.pad_across_processes(generated_tokens, 
                                                                 dim=1, 
                                                                 pad_index = self.tokenizer.pad_token_id)
            
        generated_tokens = self.accelerator.gather(generated_tokens).cpu().numpy()

        if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        return decoded_preds

      
      
      
      
