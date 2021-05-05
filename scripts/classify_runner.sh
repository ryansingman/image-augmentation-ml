#!/bin/bash

printf "\n\n*** Running affine Classifier Training***\n\n"
python scripts/classify.py configs/augmentation/affine.yaml configs/classifier/default.yaml --resume_epoch 50
#python scripts/evaluate.py configs/augmentation/affine.yaml configs/classifier/default.yaml --load_epoch 50

printf "\n\n*** Running all Classifier Training***\n\n"
python scripts/classify.py configs/augmentation/all.yaml configs/classifier/default.yaml --resume_epoch 50
#python scripts/evaluate.py configs/augmentation/all.yaml configs/classifier/default.yaml --load_epoch 50

printf "\n\n*** Running frequency Classifier Training***\n\n"
python scripts/classify.py configs/augmentation/frequency.yaml configs/classifier/default.yaml --resume_epoch 50
#python scripts/evaluate.py configs/augmentation/frequency.yaml configs/classifier/default.yaml --load_epoch 50

printf "\n\n*** Running intensity Classifier Training***\n\n"
python scripts/classify.py configs/augmentation/intensity.yaml configs/classifier/default.yaml --resume_epoch 50
#python scripts/evaluate.py configs/augmentation/intensity.yaml configs/classifier/default.yaml --load_epoch 50

printf "\n\n*** Running original Classifier Training***\n\n"
python scripts/classify.py configs/augmentation/original.yaml configs/classifier/default.yaml --resume_epoch 50
#python scripts/evaluate.py configs/augmentation/original.yaml configs/classifier/default.yaml --load_epoch 50
