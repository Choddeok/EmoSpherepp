export PYTHONPATH=.
DEVICE=1;

# data
CONFIG="egs/egs_bases/tts/emospherepp.yaml";
seen_binary_data_dir="/workspace/choddeok/hd0/dataset/binary/Fianl_ESD_VAD2ESV_I2I_IQR"
unseen_binary_data_dir="/workspace/choddeok/hd0/dataset/binary/Fianl_ESD_VAD2ESV_I2I_IQR_unseen_test"

# code
task_class="tasks.tts.EmoSpherepp.EmoSpherepp";
p_dataset_class="tasks.tts.dataset_utils.EmoSpherepp_Dataset";
up_dataset_class="tasks.tts.dataset_utils.EmoSpherepp_Dataset_infer";
model_class="models.tts.EmoSpherepp.EmoSpherepp";

run() {
    local MODEL_NAME=$1
    for config_suffix in "p_seen" "up_seen" "p_unseen" "up_unseen" "weak" "medium" "strong"; do
        local FINAL_MODEL_NAME=${MODEL_NAME}_${config_suffix}
        local GEN_DIR=/workspace/choddeok/hd0/out/prmlspeech/${MODEL_NAME}/generated_1100000_${config_suffix}/wavs

        if [ "$config_suffix" == "p_seen" ]; then
            echo "config_suffix: $config_suffix"
            local HPARAMS="binary_data_dir=$seen_binary_data_dir,task_cls=$task_class,dataset_cls=$p_dataset_class,model_cls=$model_class,gen_dir_name=$config_suffix"

            # Train
            CUDA_VISIBLE_DEVICES=$DEVICE python tasks/run.py \
                --config $CONFIG \
                --exp_name $MODEL_NAME \
                --hparams=$HPARAMS \
                --reset

            # Infer
            CUDA_VISIBLE_DEVICES=$DEVICE python tasks/run.py \
                --config $CONFIG \
                --exp_name $MODEL_NAME \
                --infer \
                --hparams=$HPARAMS \
                --reset

        elif [ "$config_suffix" == "up_seen" ]; then
            echo "config_suffix: $config_suffix"
            local HPARAMS="binary_data_dir=$seen_binary_data_dir,task_cls=$task_class,dataset_cls=$up_dataset_class,model_cls=$model_class,gen_dir_name=$config_suffix"
            # Infer
            CUDA_VISIBLE_DEVICES=$DEVICE python tasks/run.py \
                --config $CONFIG \
                --exp_name $MODEL_NAME \
                --infer \
                --hparams=$HPARAMS \
                --reset

        elif [ "$config_suffix" == "p_unseen" ]; then
            echo "config_suffix: $config_suffix"
            local HPARAMS="binary_data_dir=$unseen_binary_data_dir,task_cls=$task_class,dataset_cls=$p_dataset_class,model_cls=$model_class,gen_dir_name=$config_suffix"
            # Infer
            CUDA_VISIBLE_DEVICES=$DEVICE python tasks/run.py \
                --config $CONFIG \
                --exp_name $MODEL_NAME \
                --infer \
                --hparams=$HPARAMS \
                --reset

        elif [ "$config_suffix" == "up_unseen" ]; then
            echo "config_suffix: $config_suffix"
            local HPARAMS="binary_data_dir=$unseen_binary_data_dir,task_cls=$task_class,dataset_cls=$up_dataset_class,model_cls=$model_class,gen_dir_name=$config_suffix"
            # Infer
            CUDA_VISIBLE_DEVICES=$DEVICE python tasks/run.py \
                --config $CONFIG \
                --exp_name $MODEL_NAME \
                --infer \
                --hparams=$HPARAMS \
                --reset

        else
            echo "config_suffix: $config_suffix"
            local HPARAMS="binary_data_dir=$seen_binary_data_dir,task_cls=$task_class,dataset_cls=$p_dataset_class,model_cls=$model_class,gen_dir_name=$config_suffix,intensity=$config_suffix"
            # Infer
            CUDA_VISIBLE_DEVICES=$DEVICE python tasks/run.py \
                --config $CONFIG \
                --exp_name $MODEL_NAME \
                --infer \
                --hparams=$HPARAMS \
                --reset

        fi
    done
}

#########################
#   Run for the model   #
#########################
run "EmoSpherepp"