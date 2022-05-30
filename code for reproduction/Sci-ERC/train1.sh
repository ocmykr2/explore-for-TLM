TASK=sciie
SCALE=small
OUTPUT_DIR=./results
SAVENAME=$TASK-$SCALE-scale
MLM_WEIGHT=20
EXTERNAL_RATIO=300
LR=1e-4
WD=0.00
WARMUP=3000

if [[ $TASK == "imdb" ]]
then
MAXLEN=512
else 
MAXLEN=128
fi

accelerate launch --config_file ./accelerate_config/example_dist_config.yaml src/run.py \
    --max_train_steps 10000 \
    --steps_to_eval 3000 \
    --steps_to_save 15000 \
    --steps_to_log 3000 \
    --external_dataset_name small_external.csv \
    --preprocessing_num_workers 32 \
    --max_ckpts_to_keep 3 \
    --max_length $MAXLEN \
    --pad_to_max_length \
    --config_dir yxchar/tlm-${TASK}-${SCALE}-scale \
    --model_name_or_path $OUTPUT_DIR/$SAVENAME/final \
    --output_dir $OUTPUT_DIR/$SAVENAME \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 16 \
    --cuda_devices 0,1,2,3,4,5,6,7 \
    --task_name $TASK \
    --save_final \
    --mlm_weight $MLM_WEIGHT \
    --external_ratio $EXTERNAL_RATIO \
    --weight_decay $WD \
    --learning_rate $LR \
    --num_warmup_steps $WARMUP \
    --seed 0
