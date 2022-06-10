#!/bin/sh

# shape and checkpoint to visualize
s_model='./data/chairs/__chair2/model.obj'
ckpt='./log/main-09-06-2022__14:29:54-NeuralMLS-softmax_temp_1-chair2-rigid/net_final.pth'
# RFF configs
RFF_en=false
RFF_sigma=1
# MLS configs
mls_def='rigid'
temp=1
# visualization configs
cp_offsets="--cp_offsets 0 -0.2 0 -cpoff 0 -0.2 0 -cpoff 0 -0.2 0 -cpoff 0 -0.2 0 -cpoff 0.3 0 0 -cpoff 0.3 0 0 -cpoff 0 0 0 -cpoff 0 0 0 -cpoff 0 0.2 0 -cpoff 0 0.2 0"
cp_order="--cp_order 0 -cpord 1 -cpord 2 -cpord 3 -cpord 8 -cpord 9 -cpord 4 -cpord 5"
camera_pos="--camera_pos 1 0.4 0 -campos 1 0.4 0 -campos 1 0.4 0 -campos 1 0.4 0 -campos 1 0.4 0 -campos 1 0.4 0 -campos 1 0.4 0 -campos 1 0.4 0"
cam_radius=4.5

# set RFF related stuff
if [ $RFF_en = true ]; then
    RFF="--en_pos_enc --PE_sigma ${RFF_sigma}"
else
    RFF=""
fi

echo "##### Doing visualization for ${ckpt} ######";

python experiments/run_experiment.py \
--ckpt ${ckpt} --source_model ${s_model} ${RFF} --T ${temp} --bottleneck_size 256 --not_cuda \
--phase visualize --cam_radius ${cam_radius} --MLS_deform ${mls_def} ${cp_offsets} ${cp_order} ${camera_pos}