#!/bin/sh

# shape to run comparisons on
s_model="./data/airplanes/ShapeNetPlane1/model.obj"
# RFF configs
RFF_en=false
RFF_sigma=1
# MLS configs
mls_def='rigid'
temp=1
# whether to do KPD comparisons - must have a pre-trained checkpoint
do_KPD=true

# set shape name and KPD relevant ckpt
if [[ "$s_model" =~ "airplane_1" ]]; then
    s_name="airplane_1"
elif [[ "$s_model" =~ "armadillo_1k.off" ]]; then
    s_name="armadillo"
elif [[ "$s_model" =~ "chair1" ]]; then
    s_name="chair1"
elif [[ "$s_model" =~ "ShapeNetChair1" ]]; then
    s_name="ShapeNetChair1"
    n_keypoints=12
    KPDckpt='./logs/KPD_logs/chair-12kpt/checkpoints/net_final.pth'
elif [[ "$s_model" =~ "ShapeNetChair2" ]]; then
    s_name="ShapeNetChair2"
    n_keypoints=12
    KPDckpt='./logs/KPD_logs/chair-12kpt/checkpoints/net_final.pth'
elif [[ "$s_model" =~ "ShapeNetChair3" ]]; then
    s_name="ShapeNetChair3"
    n_keypoints=12
    KPDckpt='./logs/KPD_logs/chair-12kpt/checkpoints/net_final.pth'
elif [[ "$s_model" =~ "ShapeNetPlane1" ]]; then
    s_name="ShapeNetAirplane1"
    n_keypoints=8
    KPDckpt='./logs/KPD_logs/airplane-8kpt/checkpoints/net_final.pth'
elif [[ "$s_model" =~ "ShapeNetPlane2" ]]; then
    s_name="ShapeNetAirplane2"
    n_keypoints=8
    KPDckpt='./logs/KPD_logs/airplane-8kpt/checkpoints/net_final.pth'
elif [[ "$s_model" =~ "ShapeNetPlane3" ]]; then
    s_name="ShapeNetAirplane3"
    n_keypoints=8
    KPDckpt='./logs/KPD_logs/airplane-8kpt/checkpoints/net_final.pth'
elif [[ "$s_model" =~ "ShapeNetCar1" ]]; then
    s_name="ShapeNetCar1"
    n_keypoints=8
    KPDckpt='./logs/KPD_logs/car-8kpt/checkpoints/net_final.pth'
elif [[ "$s_model" =~ "ShapeNetCar2" ]]; then
    s_name="ShapeNetCar2"
    n_keypoints=8
    KPDckpt='./logs/KPD_logs/car-8kpt/checkpoints/net_final.pth'
elif [[ "$s_model" =~ "ShapeNetCar3" ]]; then
    s_name="ShapeNetCar3"
    n_keypoints=8
    KPDckpt='./logs/KPD_logs/car-8kpt/checkpoints/net_final.pth'
fi

# KPD stuff
if [[ $do_KPD = true ]]; then
    KPD_command="--n_keypoints ${n_keypoints} --KPDckpt ${KPDckpt}"
    KPD_name="-KPD"
else
    KPD_command=""
    KPD_name=""
fi

# set RFF related stuff
if [ $RFF_en = true ]; then
    RFF="--en_pos_enc --PE_sigma ${RFF_sigma}"
    RFF_name="-RFF_sigma${RFF_sigma}"
else
    RFF=""
    RFF_name=""
fi

name="NeuralMLS${RFF_name}-softmax_temp_${temp}-${s_name}-${mls_def}${KPD_name}"; 

echo "##### Doing ${name} ######";

python experiments/run_experiment.py \
--name ${name} --source_model ${s_model} ${RFF} --lr 6e-4 --nepochs 48 --T ${temp} \
--bottleneck_size 256 --not_cuda --MLS_deform ${mls_def} --phase create_comparisons ${KPD_command}