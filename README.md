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

```bash
conda create -n venv
source activate venv
conda install pip
pip install -r requirements.txt
```

## Datasets

Coming soon!

## Usage

Coming soon!

```bash
python ...
```

## License
This project is licensed under the terms of the MIT license.