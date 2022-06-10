#!/bin/sh

# shape to run illustration on
s_model='./data/chairs/__chair2/model.obj'
ckpt="./log/main-09-06-2022__14:29:54-NeuralMLS-softmax_temp_1-chair2-rigid/net_final.pth"
# RFF configs
RFF_en=false
RFF_sigma=1
# MLS configs
mls_def='rigid'
# visualization configs
cp_offsets="--cp_offsets 0 -0.2 0 -cpoff 0 -0.2 0 -cpoff 0 -0.2 0 -cpoff 0 -0.2 0 -cpoff 0 0 0 -cpoff 0 0 0 -cpoff 0 0 0 -cpoff 0 0 0 -cpoff 0 0.2 0.2 -cpoff 0 0.2 -0.2"
cp_order="--cp_order 0 1 2 3 4 5 6 7 8 9"
camera_pos="--camera_pos 0.7 0.1 0.3"

# set RFF related stuff
if [ $RFF_en = true ]; then
    RFF="--en_pos_enc --PE_sigma ${RFF_sigma}"
else
    RFF=""
fi

echo "##### Doing temperature illustration for ${ckpt} ######";

python experiments/run_experiment.py \
--ckpt ${ckpt} --source_model ${s_model} ${RFF} --bottleneck_size 256 --not_cuda --phase create_temperature_illustration \
--MLS_deform ${mls_def} ${cp_offsets} ${cp_order} ${camera_pos} --temps 5 4 3 2 1 0.8 0.5 0.3 0.2 --subdir "temperature_illustration"