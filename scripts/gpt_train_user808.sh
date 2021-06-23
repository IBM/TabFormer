#!/bin/bash

user_id=808
data_root="./data/credit_card/"
checkpoint="./checkpoints/credit_card/gpt2-userid_${user_id}-nbins_10-hsz_${user_id}"

cd ..
python main.py \
                --lm_type gpt2 \
                --field_ce \
                --flatten \
                --data_root  "${data_root}"\
                --data_extension "userid-${user_id}" \
                --user_ids "${user_id}" \
                --output_dir "${checkpoint}" \
                --checkpoint 0 \
                --do_train \
                --save_steps 1000 \
                --num_train_epochs 2 \
                --stride 3 \
                --field_hs 1020 \
                --cached