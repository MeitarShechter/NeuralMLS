#!/bin/sh
temp=1

for shape in 'airplane' 'car' 'chair'
do
    if [[ "$shape" =~ "chair" ]]; then
        s1="./data/chairs/ShapeNetChair1/model.obj"
        s2="./data/chairs/ShapeNetChair2/model.obj"
        s3="./data/chairs/ShapeNetChair3/model.obj"
        n_keypoints=12
        KPDckpt='./log/KPD_logs/chair-12kpt/checkpoints/net_final.pth'
        RFF_en=false
        mls_def='rigid'
    elif [[ "$shape" =~ "airplane" ]]; then
        s1="./data/airplanes/ShapeNetPlane1/model.obj"
        s2="./data/airplanes/ShapeNetPlane2/model.obj"
        s3="./data/airplanes/ShapeNetPlane3/model.obj"
        n_keypoints=8
        KPDckpt='./logs/KPD_logs/airplane-8kpt/checkpoints/net_final.pth'
        RFF_en=true
        RFF_sigma=2
        mls_def='affine'
    elif [[ "$shape" =~ "car" ]]; then
        s1="./data/cars/ShapeNetCar1/model.obj"
        s2="./data/cars/ShapeNetCar2/model.obj" 
        s3="./data/cars/ShapeNetCar3/model.obj" 
        n_keypoints=8
        KPDckpt='./log/KPD_logs/car-8kpt/checkpoints/net_final.pth'
        RFF_en=false
        mls_def='rigid'
    fi
    KPD_command="--n_keypoints ${n_keypoints} --KPDckpt ${KPDckpt}"

    # set RFF related stuff
    if [ $RFF_en = true ]; then
        RFF="--en_pos_enc --PE_sigma ${RFF_sigma}"
        RFF_name="-RFF_sigma_${RFF_sigma}"
    else
        RFF=""
        RFF_name=""
    fi

    for s_model in ${s1} ${s2} ${s3}
    do        
        if [[ "$s_model" =~ "ShapeNetChair1" ]]; then
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

        name="NeuralMLS-${s_name}${RFF_name}-softmax_temp_${temp}-${mls_def}_USERSTUDY"; 

        echo "##### Doing ${name} ######";

        python experiments/run_experiment.py \
        --name ${name} --source_model ${s_model} ${RFF} --lr 6e-4 --nepochs 72 \
        --T ${temp} --bottleneck_size 256 --not_cuda --MLS_deform ${mls_def} \
        ${KPD_command} --phase create_user_study --MLS_alpha 1 --camera_pos 2.65 1.5 2.65
    done
done




