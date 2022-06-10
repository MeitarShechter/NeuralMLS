#!/bin/sh

# shape to run ablation on
s_model='./data/chairs/__chair2/model.obj'
s_name="chair2"
# RFF configs
RFF_en=false
RFF_sigma=1
# MLS configs
mls_def='rigid'
spec_norm_en=false
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

echo "##### Running MLS ablation for ${s_model}######";
for mls_eps in 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 
do
    for mls_alpha in 5 4 3 2 1 0.8 0.5 0.3 0.2 0.1 0.05 
    do
        echo "#####epsilon=${mls_eps}, alpha=${mls_alpha}######";
        python experiments/run_experiment.py \
        --name ${s_name} --source_model ${s_model} ${RFF} ${dataset} --bottleneck_size 256 \
        --not_cuda --phase mls_ablation --MLS_deform ${mls_def} --MLS_eps ${mls_eps} --MLS_alpha ${mls_alpha} \
        ${cp_offsets} ${cp_order} ${camera_pos} --log_dir './log/mls_ablation' --subdir "eps_${mls_eps}_alpha_${mls_alpha}"
    done
done            