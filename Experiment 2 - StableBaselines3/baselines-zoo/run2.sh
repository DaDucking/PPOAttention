#!/bin/bash
for game in DemonAttack
do
    for seed in 69 71 142
    do
        HIP_VISIBLE_DEVICES=0 python train.py --seed $seed --algo ppo --env "${game}NoFrameskip-v4" -tb "tb40mRvuAttnlogs/${game}RvuAttn${seed}" -f "Rvu40mAttnlogs/${game}RvuAttn${seed}" --eval-freq 100000 -params policy_kwargs:"dict(features_extractor_class=CustomCNN,features_extractor_kwargs=dict(features_dim=512,attn_type='RvuAttn',adaptive=False),)" &
        HIP_VISIBLE_DEVICES=1 python train.py --seed $seed --algo ppo --env "${game}NoFrameskip-v4" -tb "tb40mAttnlogs/${game}Attn${seed}" -f "40mAttnlogs/${game}Attn${seed}" --eval-freq 100000 -params policy_kwargs:"dict(features_extractor_class=CustomCNN,features_extractor_kwargs=dict(features_dim=512,attn_type='Attn',adaptive=False),)" &
        HIP_VISIBLE_DEVICES=2 python train.py --seed $seed --algo ppo --env "${game}NoFrameskip-v4" -tb "tb40mCrossAttnlogs/${game}CrossAttn${seed}" -f "40mCrossAttnlogs/${game}CrossAttn${seed}" --eval-freq 100000 -params policy_kwargs:"dict(features_extractor_class=CustomCNN,features_extractor_kwargs=dict(features_dim=512,attn_type='CrossAttn',adaptive=False),)" &
        HIP_VISIBLE_DEVICES=3 python train.py --seed $seed --algo ppo --env "${game}NoFrameskip-v4" -tb "tb40mNoAttnlogs/${game}NoAttn${seed}" -f "40mNoAttnlogs/${game}NoAttn${seed}" --eval-freq 100000 &
    done
done