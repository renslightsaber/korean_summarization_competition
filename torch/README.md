# [🔥torch] How to train or inference in CLI? [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-38FxWfHqZUo8iyHUnlkKycWm9xRiiQh?usp=share_link) [![wandb](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/wako/Korean_Summarization?workspace=user-wako)
 

## Check Jupyter Notebook Version (Just 🔥torch_style Baseline) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11xDISVyASIR1JMNWPowT5L1ZGLdEfQka?usp=share_link) 
 
 <img src="/imgs/스크린샷 2023-04-01 오후 1.07.17.png" width="83%"></img>


## Caution
 - Unfortunately, Data is not Provided
 - This code costs a lot of time for training. 
    - In this code, you can check `ROUGE` scores after each step. 
    - I just wanna visualize the `ROUGE` scores per epoch.
    - You can set the number of sentences for `ROUGE` scores after each step.

## [wandb login in CLI interface](https://docs.wandb.ai/ref/cli/wandb-login)
```python
$ wandb login --relogin '######### your API token ###########'                  
``` 


## Train 
```bash
$ python train.py --data_path '/content/drive/MyDrive/ ... /data/' \
                  --model_save '/content/drive/MyDrive/ ... /test/model/torch_bart_lora/' \
                  --is_sample True \
                  --sample 1000 \
                  --model 'gogamza/kobart-base-v2' \
                  --is_lora True \
                  --lora_r 4 \
                  --lora_alpha 32 \
                  --lora_target_modules "['q_proj', 'v_proj']" \
                  --lora_dropout_p 0.05 \
                  --try_title "test_bart_lora" \
                  --n_epochs 2 \
                  --num_sentences 2 \
                  --max_length 512 \
                  --target_max_length 55 \
                  --train_batch_size 8 \
                  --valid_batch_size 8
```
- `data_path` : Data가 저장된 경로 (Default: `./data/`)
- `model_save`: 학습된 모델이 저장되는 경로
- `model`: Huggingface Pratrained Model (Default: `"eenzeenee/t5-small-korean-summarization"`)
- `model_type`: `"t5"` or `"bart"`
- `is_lora` : LoRA 적용 여부 `True` or `False`
- `lora_r`, `lora_alpha`, `lora_target_modules`, `lora_dropout` : LoRA CONFIG
- `n_epochs` : Epoch
- `num_sentences` : `train` 혹은 `validation` 동안 `Rouge` 스코어를 낼 문장의 개수 (Batch Size보다는 적은 문장 수)
- `device`: GPU를 통한 학습이 가능하다면, `cuda` 로 설정할 수 있다.
- [`train.py`](https://github.com/renslightsaber/korean_summarization_competition/blob/main/torch/train.py) 참고!   


## Inference 
```bash
$ python inference.py --data_path '/content/drive/MyDrive/ ... /data/' \
                      --model_save '/content/drive/MyDrive/ ... /test/model/torch_bart_lora/' \
                      --sub_path '/content/drive/MyDrive/ ... /test/sub/' \
                      --model 'gogamza/kobart-base-v2' \
                      --is_lora True \
                      --lora_r 4 \
                      --lora_alpha 32 \
                      --lora_target_modules "['q_proj', 'v_proj']" \
                      --lora_dropout_p 0.05 \
                      --try_title "test_bart_lora" \
                      --num_beams 3 \
                      --max_length 512 \
                      --target_max_length 55 \
                      --valid_batch_size 8

```
- `base_path` : Data가 저장된 경로 (Default: `./data/`)
- `sub_path`  : `submission.csv` 제출하는 경로
- `device`: GPU를 통한 학습이 가능하다면, `cuda` 로 설정할 수 있다.
- [`inference.py`](https://github.com/renslightsaber/korean_summarization_competition/blob/main/torch/inference.py) 참고!   






