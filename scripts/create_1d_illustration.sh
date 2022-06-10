#!/bin/sh

# RFF configs
RFF_en=false
RFF_sigma=2
# MLS configs
mls_def='rigid'
temp=1
MLS_alpha=1
MLS_eps=1e-6
# run config
n_epoches=500
do_geometric_awareness=true


if [[ $do_geometric_awareness = true ]]; then
    ge="--do_geometric_awareness"
else
    ge=""
fi

# set RFF related stuff
if [ $RFF_en = true ]; then
    RFF="--en_pos_enc --PE_sigma ${RFF_sigma}"
    RFF_name="-RFF_sigma_${RFF_sigma}"
else
    RFF=""
    RFF_name=""
fi

name="NeuralMLS${RFF_name}-softmax_temp_${temp}-${mls_def}-${n_epoches}_epoches-alpha_${MLS_alpha}-eps_${MLS_eps}";

echo "##### Doing ${name} ######";
python experiments/run_experiment.py \
--name ${name} ${RFF} --lr 6e-4 --nepochs ${n_epoches} --T ${temp} --bottleneck_size 256 --not_cuda \
--MLS_deform ${mls_def} --phase create_1d_illustration --MLS_alpha ${MLS_alpha} --MLS_eps ${MLS_eps} ${ge}