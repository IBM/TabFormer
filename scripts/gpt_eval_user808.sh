#!/bin/bash

user_id=808
hsz=1020
checkpoint=1000
bsz=64
stride=3
decode="softmax"
temp=0.5
num_seed_trans=2

cd ..

data_dir="./data/credit_card/"
output_dir="./checkpoints/credit_card/gpt2-userid_${user_id}-nbins_10-hsz_${user_id}"

python gpt_eval.py \
  --output_dir "${output_dir}" \
  --data_dir "${data_dir}" \
  --user_id "${user_id}" \
  --checkpoint "${checkpoint}" \
  --hidden_size "${hsz}" \
  --batch_size "${bsz}" \
  --stride "${stride}" \
  --decoding "${decode}" \
  --temperature "${temp}" \
  --num_seed_trans "${num_seed_trans}" \
  --data_extension "userid-${user_id}" \
  --store_csv