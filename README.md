# The Devil is in the Details: On Models and Training Regimes for Few-Shot Intent Classification

This repository incldues code that we used for the experiments in our EACL 2023 paper: 

```
@InProceedings{mesgar-etal-2023-devil,
  author    = "Mesgar, Mohsen and Tran, Thy Thy and Glavas, Goran and Gurevych, Iryna"
  title     = "The Devil is in the Details: On Models and Training Regimes for Few-Shot Intent Classification",
  booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume",
  publisher = "Association for Computational Linguistics", 
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International}
  url       = "https://arxiv.org/abs/2210.06440"
}
```

> **Abstract:** Few-shot Intent Classification (FSIC) is one of the key challenges in modular task-oriented dialog systems. While advanced FSIC methods are similar in using pretrained language models to encode texts and nearest neighbour-based inference for classification, these methods differ in details. They start from different pretrained text encoders, use different encoding architectures with varying similarity functions, and adopt different training regimes. Coupling these mostly independent design decisions and the lack of accompanying ablation studies are big obstacle to identify the factors that drive the reported FSIC performance. We study these details across three key dimensions: (1) Encoding architectures: Cross-Encoder vs Bi-Encoders; (2) Similarity function: Parameterized (i.e., trainable) functions vs non-parameterized function; (3) Training regimes: Episodic meta-learning vs the straightforward (i.e., non-episodic) training. Our experimental results on seven FSIC benchmarks reveal three important findings. First, the unexplored combination of the cross-encoder architecture (with parameterized similarity scoring function) and episodic meta-learning consistently yields the best FSIC performance. Second, Episodic training yields a more robust FSIC classifier than non-episodic one. Third, in meta-learning methods, splitting an episode to support and query sets is not a must. Our findings paves the way for conducting state-of-the-art research in FSIC and more importantly raise the community's attention to details of FSIC methods. We release our code and data publicly.


Contact person: Mohsen Mesgar, mohsen.mesgar@tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.


## Environmental Setup

We use `Python: 3.6.8` and `cuda/11.1`

```bash
conda create -n venv
source activate venv
conda install pip
pip install -r requirements.txt
```

## Datasets

We conduct our main experiments on the Fewshot IC/SF dataset which was introduced in this [paper](https://arxiv.org/abs/2004.10793), including ATIS, SNIPS, TOP.

For experiments on ATIS, SNIPS or TOP, we create 100 episodes from the training split of each dataset. 
For dev and test, we create as many as possible episodes to cover all samples. 

```bash
python metadataset_creation.py  --task atis   --seed 0
python metadataset_creation.py  --task snips   --seed 0
python metadataset_creation.py  --task fb_top   --seed 0
```


## Usage

### Episodic learning

```bash
seed=1
python episodic_learning.py \
        --train_path "path/to/train" \
        --dev_path "path/to/valid" \
        --test_path "path/to/test" \
        --model_name_or_path {"bert-base-uncased" | "princeton-nlp/sup-simcse-bert-base-uncased" } \
        --output_dir "path/to/output/dir" \
        --seed $seed \
        --max_seq_length 64 \
        --max_iter 10000 \
        --evaluate_every 100 \
        --log_every 10 \
        --learning_rate 0.00002 \
        --early_stop 5 \
        --batch_size 64 \
        --encoder_type {"be" | "ce" } \
        --do_train --do_test
```



### Episodic learning with support and query sets

```bash
seed=1
python episodic_learning_wsq.py \
        --train_path "path/to/train" \
        --dev_path "path/to/valid" \
        --test_path "path/to/test" \
        --model_name_or_path {"bert-base-uncased" | "princeton-nlp/sup-simcse-bert-base-uncased" } \
        --output_dir "path/to/output/dir" \
        --seed $seed \
        --max_seq_length 64 \
        --max_iter 10000 \
        --evaluate_every 100 \
        --log_every 10 \
        --learning_rate 0.00002 \
        --early_stop 5 \
        --batch_size 64 \
        --encoder_type {"be" | "ce" } \
        --do_train --do_test
```


### Non-Episodic learning

```bash
seed=1
python non-episodic-learning.py \
        --train_path "path/to/train" \
        --dev_path "path/to/valid" \
        --test_path "path/to/test" \
        --model_name_or_path {"bert-base-uncased" | "princeton-nlp/sup-simcse-bert-base-uncased" } \
        --output_dir "path/to/output/dir" \
        --encoder_type {"be" | "ce" } \
        --batch_size 16 \
        --learning_rate 0.00002  \
        --early_stop 5 \
        --max_iter 10000 \
        --do_train --do_eval
```


### Reproduce

We notice a slight difference in results when we run our experiments on different GPUs. 
This happens because the dropout layers in BERT behaves differently on different devices. 
Since the behaviour of BERT's dropout layers is out of our control, we re-run all experiments on an identical machine with the following specifications:
```
Linux 3.10.0-1160.11.1.el7.x86_64 #1 SMP Fri Dec 18 16:34:56 UTC 2020 x86_64 x86_64 x86_64 GNU/Linux
GPU: Tesla V100-PCIE-32GB
Mem: 754G
CPU:  Intel(R) Xeon(R) Gold 6154 CPU @ 3.00GHz  width: 32,64 bits (Num:72)
```  


## License
This project is licensed under the terms of the MIT license.