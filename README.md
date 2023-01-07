# LAPGM
Inplementation of two graph matching methods, PCA-GM and IPCA-GM. Course project of (2022-2023-1) AI3067@SJTU: Deep Learning and Its Application.

Runnale code in `./code/`.

## Requirements
For the faster CUDA backend, NVIDIA GPU is required.

Please install the requirements by running this
```bash
pip install -r requirements.txt
```

## Training
Use `main.py` to either train or test a model. For example, if you want to train a PCA-GM model with pretrained VGG finetuned on WillowObject dataset, you should run 
```bash
cd Code/
python main.py --name WHATEVER_EXPERIMENT_NAME_YOU_LIKE --dataset WillowObject --model pca-gm --batch_size 2 --lr 1e-4 --weight_decay 1e-2 --extractor_train
```
This trains such a model with `batch_size=2`, `learning_rate=1e-4`, and `weight_decay=1e-2`. For more options, please checkout `./Code/options.py`.