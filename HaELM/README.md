# HaELM
An automatic MLLM hallucination detection framework

## 1. Installing
Install peft
```bash
$ pip install git+https://gitclone.com/github.com/huggingface/peft.git -i https://pypi.mirrors.ustc.edu.cn/simple --trusted-host=pypi.mirrors.ustc.edu.cn
```

## 2. Preparing
Download the checkpoint of llama-7b-hf

## 3. Training
We provide the hallucination training dataset in "***data/train_data.jsonl***" and the manually labeled validation set in "***data/eval_data.jsonl***".
If you want to:
* Retrain
* Use another scale of llama
* Use llama-2
* Use other data

see here.

* Modify the path in lines 19-21 of finetune.py
* Run the command below
```bash
python finetune.py 
```

## 4. Interface
We provide interface templates populated by the output of mPLUG-Owl in "***LLM_output/mPLUG_caption.jsonl***".
* Modify the path in lines 14-16 of interface.py
* Run the command below
```bash
python finetune.py 
```

## 5. Citation
```
@article{wang2023evaluation,
  title={Evaluation and Analysis of Hallucination in Large Vision-Language Models},
  author={Wang, Junyang and Zhou, Yiyang and Xu, Guohai and Shi, Pengcheng and Zhao, Chenlin and Xu, Haiyang and Ye, Qinghao and Yan, Ming and Zhang, Ji and Zhu, Jihua and others},
  journal={arXiv preprint arXiv:2308.15126},
  year={2023}
}
```
