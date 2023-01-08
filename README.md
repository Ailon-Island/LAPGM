# LAPGM
Implementation of some graph matching methods, PCA-GM, IPCA-GM, and CIE. Hungarian attention and a soft variant is also included. Course project of (2022-2023-1) AI3067@SJTU: Deep Learning and Its Application.

Runnable code in `./Code/`.

## Requirements
For the faster CUDA backend, NVIDIA GPU is required.

Please install the requirements by running this (This is not tested still. If you have any problems, open an issue to tell us.)
```bash
pip install -r requirements.txt
```

## Training
Use `main.py` to either train or test a model. For example, to train a PCA-GM model with soft Hungarian attention loss with hardness=`0.6` on WillowObject dataset, whose pretrained VGG backbone is finetuned simultaneously, you could run 
```bash
cd Code/
python main.py --name WHATEVER_EXPERIMENT_NAME_YOU_LIKE --dataset WillowObject --model pca-gm --batch_size 2 --lr 1e-4 --weight_decay 1e-2 --extractor_train --hungarian_attention --lambda_hungarian 0.6
```
The model is then trained with batch size=`2`, learning rate=`1e-4`, and weight decay=`1e-2`. For more options, please checkout `./Code/options.py`.
