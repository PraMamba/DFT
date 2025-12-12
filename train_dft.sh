#!/bin/bash

set -eu

# Disable tokenizer parallelism warning
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

# === Environment Setup ===

source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Axolotl

# === Navigate to Project Directory ===
cd ~/DFT

# === Experiment Parameters ===
LAUNCH_TYPE=DeepSpeed       # Options: DeepSpeed, TorchRun
MODEL_PATH="/gpfs/Mamba/Project/Single_Cell/Training/Qwen2.5-7B-Instruct_Manual_Resize_Block-8_Size-32_Num-256_ExplicitTokens_Continued-Pretrain=TOP-500_GeneNames_Sample-0.1_MetaQA=Cell_Type_NEW-RVQ_V1/checkpoint-72000"
DATE_SUFFIX=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/data/Mamba/Project/Single_Cell/Training/Qwen2.5-7B-Instruct_Manual_Resize_Block-8_Size-32_Num-256_ExplicitTokens_Continued-Pretrain=TOP-500_GeneNames_Sample-0.1_MetaQA=Cell_Type_NEW-RVQ_DFT"
RUN_NAME=model_training

# === Training Data ===
DATA_FILES=(
    "/gpfs/Mamba/Project/Single_Cell/Data/RVQ-NC=8-CS=32/MetaQA_SingleCell_Individual_Tasks_w_Reasoning/filtered_rows_with_all_fields_stratify-sample_num-100000/train_conversation_fixed.jsonl"
)

# === DFT-Specific Parameters ===
DFT_ALPHA=0.8
USE_SIMPLE_DFT=True

# === Model Parameters ===
TORCH_DTYPE=bfloat16
ATTN_IMPLEMENTATION=flash_attention_2
TRUST_REMOTE_CODE=True
MAX_LENGTH=4096
ENABLE_GRADIENT_CHECKPOINTING=False

# === Training Hyperparameters ===
NUM_EPOCHS=5
TRAIN_BATCH_SIZE=2
EVAL_BATCH_SIZE=2
GRAD_ACCUM_STEPS=8
LEARNING_RATE=1e-6
WEIGHT_DECAY=0.1
WARMUP_RATIO=0.1
LR_SCHEDULER_TYPE=cosine
SEED=42

# === Data Processing ===
PREPROCESSING_NUM_WORKERS=8
DATALOADER_NUM_WORKERS=8
VALIDATION_SPLIT_PERCENTAGE=0.0
DEBUG_DATA_PROCESSING=True

# === Checkpoint & Logging ===
SAVE_STEPS=500
EVAL_STEPS=500
SAVE_TOTAL_LIMIT=5
SAVE_ONLY_MODEL=True
LOGGING_STEPS=1
LOG_METRICS_STEPS=500
REDUCE_LOGGING=True
REPORT_TO=swanlab       # Options: swanlab, wandb, tensorboard, none

# === DeepSpeed Configuration ===
DEEPSPEED_CONFIG=./config/zero2.json

# === GPU Configuration ===
GPU_DEVICES="0,1,2,3"
NUM_GPUS=4
MASTER_PORT=29500

# === Log Setup ===
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/train_dft.log"
echo "Logging to ${LOG_FILE}"

echo "=============================================="
echo "Real-Time Training Log Monitoring"
echo "tail -f ${LOG_FILE}"
echo "=============================================="

# === Common Arguments ===
common_args="
    --model_name_or_path $MODEL_PATH \
    --torch_dtype $TORCH_DTYPE \
    --attn_implementation $ATTN_IMPLEMENTATION \
    --trust_remote_code $TRUST_REMOTE_CODE \
    --data_files ${DATA_FILES[@]} \
    --max_length $MAX_LENGTH \
    --preprocessing_num_workers $PREPROCESSING_NUM_WORKERS \
    --validation_split_percentage $VALIDATION_SPLIT_PERCENTAGE \
    --debug_data_processing $DEBUG_DATA_PROCESSING \
    --enable_gradient_checkpointing $ENABLE_GRADIENT_CHECKPOINTING \
    --dft_alpha $DFT_ALPHA \
    --use_simple_dft $USE_SIMPLE_DFT \
    --reduce_logging $REDUCE_LOGGING \
    --log_metrics_steps $LOG_METRICS_STEPS \
    --bf16 True \
    --fp16 False \
    --seed $SEED \
    --data_seed $SEED \
    --optim adamw_torch_fused \
    --output_dir $OUTPUT_DIR \
    --run_name $RUN_NAME \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --eval_strategy steps \
    --eval_steps $EVAL_STEPS \
    --save_strategy steps \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --save_only_model $SAVE_ONLY_MODEL \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type $LR_SCHEDULER_TYPE \
    --logging_strategy steps \
    --logging_steps $LOGGING_STEPS \
    --logging_dir $LOG_DIR \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --ddp_find_unused_parameters False \
    --ddp_timeout 30000 \
    --log_on_each_node False \
    --logging_first_step True \
    --save_safetensors True \
    --remove_unused_columns False \
    --disable_tqdm False \
    --report_to $REPORT_TO
"

if [[ "$LAUNCH_TYPE" == "TorchRun" ]]; then
    echo "Using TorchRun"
    CUDA_VISIBLE_DEVICES=${GPU_DEVICES} torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} train_dft.py \
        $common_args >> "${LOG_FILE}" 2>&1 &
else
    echo "Using DeepSpeed"
    DEEPSPEED_DEVICES="localhost:${GPU_DEVICES}"
    deepspeed --include ${DEEPSPEED_DEVICES} --master_port ${MASTER_PORT} train_dft.py \
        --deepspeed $DEEPSPEED_CONFIG $common_args >> "$LOG_FILE" 2>&1 &
fi

echo "Training started in background, PID: $!"
echo "Monitor logs: tail -f $LOG_FILE"
