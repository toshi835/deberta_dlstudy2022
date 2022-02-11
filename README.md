# deep learning study in 2022

Fine-tune DeBERTa by SST-2

## How to use
### Preparing
1. install
```
git clone https://github.com/toshi835/deberta_dlstudy2022
```
1. Download SST-2  
ref: https://github.com/nyu-mll/GLUE-baselines
1. set dataset to `data/`

### Training
```
python train.py -gpus 0 -epoch 10
```
### Test
```
python test.py -model_path ../data/sst2_epoch1.pt -gpus 0
```
