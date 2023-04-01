# [[공사 중][AI CONNECT] 노트북으로 GPT 맛보기(Korean Abstractive Summary Competition)](https://aiconnect.kr/competition/detail/223)


<img src="/imgs/스크린샷 2023-03-31 오후 5.40.40.png" width="82%"></img>

## Competition Info
 - Period: 2023.03.20 - 2022.03.30
 - Joined as: `Individual`
 - TEAM_NAME : `heisweak`
 - TASK: `Abstractive Summary`
 - Evaluation Metric: `ROUGE-1(F1)`, `ROUGE-2(F1)`, `ROUGE-L(F1)` | `Tokenizer = Mecab`
 - Environment: `Colab`
 
 ## Result 
 - PUBLIC  : `3rd` / 418 teams (total 601 participants)
   > Hope Higher or this Rank in `Private LeaderBoard` or `Final LeaderBoard`
 - PRIVATE : `Not yet`
 - Final: `Not yet`
 
## Why Joined this Competition
- Studying `NLG` task; models, how to fine-tune, ... 
- Needs Experience for `NLG` task
  - Never experienced before
- Test Lightning 2.0 
  - Not enough time for this. 
  - `Train` and `Inference` with: 
    - 🤗huggingface
    - 🔥torch 
 
## How to train or inference in CLI? 
- [pip install ... ](https://github.com/renslightsaber/korean_summarization_competition/blob/main/pip_install.md)
- [🤗huggingface - Practice in cli](https://github.com/renslightsaber/korean_summarization_competition/tree/main/hf) 
- [🔥torch - Practice in cli](https://github.com/renslightsaber/korean_summarization_competition/tree/main/torch) 
- Check [![wandb](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/wako/Korean_Summarization?workspace=user-wako)

## References
- [🤗Huggingface - Summarization](https://huggingface.co/course/chapter7/5?fw=pt)
- [🤗Huggingface - BartForConditionalGeneration](https://huggingface.co/docs/transformers/v4.27.1/en/model_doc/bart#transformers.BartForConditionalGeneration)
- [KoBART Summarization with PL](https://github.com/seujung/KoBART-summarization)
  - [Rouge Class (Mecab)](https://github.com/seujung/KoBART-summarization/blob/main/rouge_metric.py)
- [[LoRA] allow fine-tuning of the text encoder with LoRA (using peft) #2719](https://github.com/huggingface/diffusers/issues/2719)
- [Efficient Large Language Model training with LoRA and Hugging Face](https://www.philschmid.de/fine-tune-flan-t5-peft)
- [Transformer로 텍스트를 생성하는 다섯 가지 전략](https://littlefoxdiary.tistory.com/46)
