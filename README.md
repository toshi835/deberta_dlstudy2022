# deep learning study in 2022

Fine-tune DeBERTa by SST-2

## How to use
### Prepare
1. install
```
git clone https://github.com/toshi835/deberta_dlstudy2022
```
2. Download SST-2  
ref: https://github.com/nyu-mll/GLUE-baselines
3. set SST-2 dataset to `data/`

### Train
```
python train.py -gpu_id 0 -epoch 10 -batch_size 40
```
### Test
```
python test.py -model_path ../data/sst2_epoch1.pt -gpu_id 0
```
