# [🤗Huggingface] How to train or inference in CLI?
 - [AutoModel](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModel) with [SequenceClassifierOutput](https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1455fFTdWik8K4HhcUbr6t098mzPJ-VSe?usp=share_link) [![wandb](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/wako/FB_TWO/groups/hf_automodel_cli_pr-Baseline/workspace?workspace=user-wako)
 - [AutoModelForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSequenceClassification) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c3SLFFqISXPF45HStryBOc1XB_NHd1k8?usp=share_link) [![wandb](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/wako/FB_TWO/groups/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_cli-Baseline/workspace?workspace=user-wako)
 
     <img src="/imgs/스크린샷 2023-03-14 오후 10.17.29.png" width="83%"></img>


## Check Jupyter Notebook Version
- [AutoModel](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModel) with [SequenceClassifierOutput](https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14HjKUnMtbEUPf-9hSV876jSvUi3xQU3S?usp=share_link) [![wandb](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/wako/FB_TWO/groups/HuggingFace_Rough_AutoModel-Baseline/workspace?workspace=user-wako)
- [AutoModelForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSequenceClassification) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1q-k_I3uBHzT9v1z8TuqfyRpAFGs9XpSW?usp=share_link) [![wandb](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/wako/FB_TWO/groups/HuggingFace_Rough_AutoMFSC-Baseline/workspace?workspace=user-wako)
 

     <img src="/imgs/스크린샷 2023-03-14 오후 10.20.54.png" width="83%"></img>

## Download Data Kaggle API Command 
```python
$ kaggle competitions download -c feedback-prize-effectiveness
$ unzip '*.zip'
```
 
## [wandb login in CLI interface](https://docs.wandb.ai/ref/cli/wandb-login)
```python
$ wandb login --relogin '######### your API token ###########'                  
``` 



## Train 
```bash
$ python train.py --base_path '/content/Kaggle_FB2/huggingface/' \
                  --model_save '/content/drive/MyDrive/ ... /Kaggle FB2/hf/cli2/' \
                  --sub_path '/content/drive/MyDrive/ ... /Kaggle FB2/hf/cli2' \
                  --model "microsoft/deberta-v3-base" \
                  --model_type "AutoModel" \
                  --hash "hf_automodel_cli_practice" \
                  --grad_clipping True\
                  --n_folds 3 \
                  --n_epochs 3 \
                  --device 'cuda' \
                  --max_length 256 \
                  --train_bs 8 \
                  --valid_bs 16
```
- `base_path` : Data가 저장된 경로 (Default: `./data/`)
- `sub_path`  : `submission.csv` 제출하는 경로
- `model_save`: 학습된 모델이 저장되는 경로
- `model`: Huggingface Pratrained Model (Default: `"microsoft/deberta-v3-base"`)
- `model_type`: [`AutoModel`](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModel) or [`AutoModelForSequenceClassification`](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSequenceClassification)
- `hash`: Name for WANDB Monitoring
- `n_folds`  : Fold 수
- `n_epochs` : Epoch
- `seed` : Random Seed (Default: 2022)
- `train_bs` : Batch Size (Default: 16)
- `max_length` : Max Length (Default: 128) for HuggingFace Tokenizer
- `grad_clipping`: [Gradient Clipping](https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem)
- `ratio` : 데이터를 Split하여 `train`(학습) 과 `valid`(성능 평가)를 만드는 비율을 의미. 정확히는 `train`의 Size 결정
- `device`: GPU를 통한 학습이 가능하다면, `cuda` 로 설정할 수 있다.
- `learning_rate`, `weight_decay`, `min_lr`, `T_max` 등은 생략 
- [`train.py`](https://github.com/renslightsaber/Kaggle_FB2/blob/main/huggingface/train.py) 참고!   


## 주의
 - CLI 환경에서 train 시킬 때, `tqdm`의 Progress Bar가 엄청 많이 생성된다. 아직 원인과 해결을 못 찾은 상태이다.
 - Colab과 Jupyter Notebook에서는 정상적으로 Progress Bar가 나타난다.



## Inference 
```python

$ python inference.py --base_path './data/' \
                      --model_save '/content/drive/MyDrive/ .. /Kaggle FB2/hf/cli2/' \
                      --sub_path '/content/drive/MyDrive/ ... /Kaggle FB2/hf/cli2/' \
                      --model "microsoft/deberta-v3-base" \
                      --model_type "AutoModel" \
                      --hash "hf_automodel_cli_practici" \
                      --n_folds 3 \
                      --n_epochs 3 \
                      --device 'cuda' \
                      --max_length 256 \
                      --valid_bs 16

```
- `base_path` : Data가 저장된 경로 (Default: `./data/`)
- `sub_path`  : `submission.csv` 제출하는 경로
- `model_save`: 학습된 모델이 저장되는 경로
- `model`: train했을 때의 Huggingface Pratrained Model (Default: `"microsoft/deberta-v3-base"`)
- `n_folds`  : `train.py`에서 진행항 KFold 수
- `n_epochs` : train 했을 때의 Epoch 수 (submission 파일명에 사용)  
- `seed` : Random Seed (Default: 2022)
- `valid_bs` : Batch Size for Inference (Default: 16) 
- `max_length` : Max Length (Default: 256) for HuggingFace Tokenizer
- `device`: GPU를 통한 학습이 가능하다면, `cuda` 로 설정할 수 있다.
- [`inference.py`](https://github.com/renslightsaber/Kaggle_FB2/blob/main/huggingface/inference.py) 참고!   






