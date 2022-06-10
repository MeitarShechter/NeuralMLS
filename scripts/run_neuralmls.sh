#!/bin/sh

# RFF configs
RFF_en=false
RFF_sigma=1
# MLS configs
mls_def='rigid'
temp=1

# whether to deform the shape based on a pre-trained model
while getopts "s:c:" flag; do
case "$flag" in
    c) checkpoint=${OPTARG};;
    s) s_model=${OPTARG};;
esac
done
if [ -z "$s_model" ]; then
    echo "ERROR: MUST PROVIDE SOURCE SHAPE"
fi
if [ -z "$checkpoint" ]; then
    deform_shape=false
else
    deform_shape=true
fi

# set shape name
s_name="user_given_shape"
if [[ "$s_model" =~ "airplane_1" ]]; then
    s_name="airplane_1"
elif [[ "$s_model" =~ "armadillo_1k.off" ]]; then
    s_name="armadillo"
elif [[ "$s_model" =~ "chair1" ]]; then
    s_name="chair1"
elif [[ "$s_model" =~ "chair2" ]]; then
    s_name="chair2"
elif [[ "$s_model" =~ "ShapeNetChair1" ]]; then
    s_name="ShapeNetChair1"
elif [[ "$s_model" =~ "ShapeNetChair2" ]]; then
    s_name="ShapeNetChair2"
elif [[ "$s_model" =~ "ShapeNetChair3" ]]; then
    s_name="ShapeNetChair3"
elif [[ "$s_model" =~ "ShapeNetPlane1" ]]; then
    s_name="ShapeNetAirplane1"
elif [[ "$s_model" =~ "ShapeNetPlane2" ]]; then
    s_name="ShapeNetAirplane2"
elif [[ "$s_model" =~ "ShapeNetPlane3" ]]; then
    s_name="ShapeNetAirplane3"
elif [[ "$s_model" =~ "ShapeNetCar1" ]]; then
    s_name="ShapeNetCar1"
elif [[ "$s_model" =~ "ShapeNetCar2" ]]; then
    s_name="ShapeNetCar2"
elif [[ "$s_model" =~ "ShapeNetCar3" ]]; then
    s_name="ShapeNetCar3"
fi

# if running deform shape
if [ $deform_shape = true ]; then
    phase='--phase deform_shape'
    ckpt="--ckpt ${checkpoint}"
else
    phase=""
    ckpt=""
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

echo "##### Doing ${name} with deform_shape = ${deform_shape} ######";

python main.py \
--name ${name} --source_model ${s_model} ${RFF} --lr 6e-4 --nepochs 48 --T ${temp} \
--bottleneck_size 256 --not_cuda --MLS_deform ${mls_def} ${phase} ${ckpt}