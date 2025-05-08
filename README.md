# Balance-aware Sequence Sampling Makes Multimodal Learning Better
This repo is the official implementation of _Retrieval-Balance-aware Sequence Sampling Makes Multimodal Learning Better_ accepted by IJCAI 2025. 

## Framework
<img width="1232" alt="image" src="Figure/framework.png" />

## Environment Configuration
First, clone this repo:
```shell
git clone https://github.com/njustkmg/IJCAI25-BSS.git

cd IJCAI25-BSS
```
First, create a new conda env for RAGPT:
```shell
conda create -n BSS python=3.7
```
Next, activate this env and install the dependencies from the requirements.txt:
```shell
conda activate BSS

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```
