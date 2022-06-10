#!/bin/sh

# RFF configs
RFF_en=false
RFF_sigma=2
# MLS configs
mls_def='rigid'
temp=1

# set RFF related stuff
if [ $RFF_en = true ]; then
    RFF="--en_pos_enc --PE_sigma ${RFF_sigma}"
    RFF_name="-RFF_sigma_${RFF_sigma}"
else
    RFF=""
    RFF_name=""
fi

name="NeuralMLS${RFF_name}-softmax_temp_${temp}-${mls_def}"; 

echo "##### Doing ${name} ######";
python experiments/run_experiment.py \
--name ${name} ${RFF} --lr 6e-4 --nepochs 72 --T ${temp} --bottleneck_size 256 \
--not_cuda --MLS_deform ${mls_def} --phase create_2d_illustration