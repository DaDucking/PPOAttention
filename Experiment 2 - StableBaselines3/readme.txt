Instructions for Baselines

Bash run2.sh bash file can change type of variables for testing models: Attention type and adaptivity

variables:
$seed
$model [CAN(Cross Attending Network), CSAN(Channel-wise Self Attending Network), SAN(Self Attending Network)]
$adaptivity

python train.py --seed $seed --algo ppo --env "${game}NoFrameskip-v4" -tb "tensorboard file name" -f "model file name" --eval-freq 100000 -params policy_kwargs:"dict(features_extractor_class=CustomCNN,features_extractor_kwargs=dict(features_dim=512,attn_type=$model ,adaptive=$adaptivity),)" &

- Model is located in utils/utils.py
- Hyperparameters is located in hyperparams/ppo.yml


Results Analysis:
Tbextractor.py returns CSV files of the tensorboard files
plot.py plots the graph based on the csv files