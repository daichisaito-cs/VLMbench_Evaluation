# VLMbench Evaluation

## Instructions

We assume the following environment for our experiments:

- Python 3.8.10 (pyenv is strongly recommended)
- PyTorch version 2.1.0 with CUDA 11.8 support

### Clone & Install

```bash
git clone XXXXX
cd VLMbench_Evaluation
```

```bash
pyenv virtualenv 3.8.10 vlmbench_evaluation
pyenv local vlmbench_evaluation
pip install -r requirements.txt
```

### Datasets

- Our dataset can be downloaded at [this link](https://vlmbench-evaluation.s3.ap-northeast-1.amazonaws.com/dataset.zip).
  - Unzip and extract the `dataset`.

<!-- ### Checkpoint

The best checkpoint can be downloaded at [this link](https://polos-polaris.s3.ap-northeast-1.amazonaws.com/reprod.zip). -->

### Train & Evaluation

```bash
export PYTHONPATH=`pwd`
python src/main.py
```