#!/bin/bash

ml load Miniforge3/24.1.2-0
conda activate mmfm
cd ${HOME}/code/mmfm/benchmark/dgp_waves

# Create logs directory
mkdir -p logs

# Define arrays for each parameter
# For explanation of the parameters please see the train_mmmfm.py file
SEEDS=(0 1 2 3 4)
MAX_GRAD_NORM=(false)  
P_UNCONDITIONAL=(0.0 0.1 0.2)
NS_PER_T_AND_C=(50)
X_LATENT_DIM=(16 32 64 96 128)
TIME_EMBED_DIM=(16 32 64 96 128)
COND_EMBED_DIM=(16 32 64 96 128)
CONDITIONAL_MODEL=(true)
EMBEDDING_TYPE=("free")
SUM_TIME_EMBED=(false)
SUM_COND_EMBED=(false)
NORMALIZATION=("None")  
AFFINE_TRANSFORM=(false)
MAX_NORM_EMBEDDING=(true)
INIT_WEIGHTS=("xavier_normal")
ACTIVATION=("LeakyReLU")
LRS=("cosine")
INTERPOLATION=("cubic" "linear")
N_EPOCHS=(300)
COUPLING=("cot")
BATCH_SIZE=("None")
TRAIN_TEST_SPLIT=(0.5)
LR=(0.1 0.01 0.001)
FLOW_VARIANCE=(1.0 0.1 0.01 "adaptive2-4-0")
OPTIMIZER_NAME=("adam")
DGP=("i")
OFF_DIAGONAL=(0.0)
DATA_STD=(0.025)  
DIMENSION=(2)
NUM_OUT_LAYERS=(3)
SPECTRAL_NORM=(false)
DROPOUT=(0.0)
CONDITIONAL_BIAS=(false)
KEEP_CONSTANTS=(false)
CLASSIFIER_FREE=(false)
MODEL_TYPE=("fm" "mmfm")
MATCHING=("emd")

# Loop over all parameter combinations
for seed in "${SEEDS[@]}"; do
for max_grad_norm in "${MAX_GRAD_NORM[@]}"; do
for p_unconditional in "${P_UNCONDITIONAL[@]}"; do
for ns_per_t_and_c in "${NS_PER_T_AND_C[@]}"; do
for x_latent_dim in "${X_LATENT_DIM[@]}"; do
for time_embed_dim in "${TIME_EMBED_DIM[@]}"; do
for cond_embed_dim in "${COND_EMBED_DIM[@]}"; do
for conditional_model in "${CONDITIONAL_MODEL[@]}"; do
for embedding_type in "${EMBEDDING_TYPE[@]}"; do
for sum_time_embed in "${SUM_TIME_EMBED[@]}"; do
for sum_cond_embed in "${SUM_COND_EMBED[@]}"; do
for normalization in "${NORMALIZATION[@]}"; do
for init_weights in "${INIT_WEIGHTS[@]}"; do
for activation in "${ACTIVATION[@]}"; do
for lrs in "${LRS[@]}"; do
for interpolation in "${INTERPOLATION[@]}"; do
for n_epochs in "${N_EPOCHS[@]}"; do
for coupling in "${COUPLING[@]}"; do
for affine_transform in "${AFFINE_TRANSFORM[@]}"; do
for max_norm_embedding in "${MAX_NORM_EMBEDDING[@]}"; do
for batch_size in "${BATCH_SIZE[@]}"; do
for train_test_split in "${TRAIN_TEST_SPLIT[@]}"; do
for lr in "${LR[@]}"; do
for flow_variance in "${FLOW_VARIANCE[@]}"; do
for dgp in "${DGP[@]}"; do
for optimizer_name in "${OPTIMIZER_NAME[@]}"; do
for off_diagonal in "${OFF_DIAGONAL[@]}"; do
for data_std in "${DATA_STD[@]}"; do
for dimension in "${DIMENSION[@]}"; do
for num_out_layers in "${NUM_OUT_LAYERS[@]}"; do
for spectral_norm in "${SPECTRAL_NORM[@]}"; do
for dropout in "${DROPOUT[@]}"; do
for conditional_bias in "${CONDITIONAL_BIAS[@]}"; do
for keep_constants in "${KEEP_CONSTANTS[@]}"; do
for classifier_free in "${CLASSIFIER_FREE[@]}"; do
for model_type in "${MODEL_TYPE[@]}"; do
for matching in "${MATCHING[@]}"; do
# for smoothness_penalty in "${SMOOTHNESS_PENALTY[@]}"; do

    # If model_type is fm and interpolation is not linear, skip
    if [ $model_type == "fm" ] && [ $interpolation != "cubic" ]; then
        continue
    fi

    # if X_LATENT_DIM is 64 and TIME_EMBED_DIM < 16, skip
    if [ $x_latent_dim -eq 64 ] && [ $time_embed_dim -lt 16 ]; then
        continue
    fi
    # If X_LATENT_DIM is 64 and COND_EMBED_DIM < 16, skip
    if [ $x_latent_dim -eq 64 ] && [ $cond_embed_dim -lt 16 ]; then
        continue
    fi
    # if X_LATENT_DIM is 32 and TIME_EMBED_DIM < 4, skip
    if [ $x_latent_dim -eq 32 ] && [ $time_embed_dim -lt 4 ]; then
        continue
    fi
    # If X_LATENT_DIM is 32 and COND_EMBED_DIM < 4, skip
    if [ $x_latent_dim -eq 32 ] && [ $cond_embed_dim -lt 4 ]; then
        continue
    fi

    # Create a unique job name
    JOB_NAME="dgp_waves_${dgp}_${seed}_${lr}_${flow_variance}_${num_out_layers}_${max_grad_norm}_${p_unconditional}_${ns_per_t_and_c}_${x_latent_dim}_${time_embed_dim}_${cond_embed_dim}_${normalization}_${init_weights}_${activation}_${lrs}_${interpolation}_${conditional_model}_${classifier_free}_${embedding_type}_${sum_time_embed}_${sum_cond_embed}_${n_epochs}_${coupling}_${affine_transform}_${max_norm_embedding}_${batch_size}_${train_test_split}_${off_diagonal}_${data_std}_${dimension}_${optimizer_name}_${spectral_norm}_${dropout}_${conditional_bias}_${keep_constants}_${matching}_${model_type}"
    JOB_NAME=$(echo "$JOB_NAME" | sed 's/true/True/g; s/false/False/g')

    # Check if results_mmfm / JOB_NAME /  "df_results.csv"  exists
    if [ -f "/home/rohbeckm/scratch/results/dgp_waves/results_mmfm/${JOB_NAME}/df_results.csv" ]; then
        echo "Job found, skipping..."
    else
        # Submit the job
        echo "Job not found, submitting..."
        bsub -J "${JOB_NAME}" \
            -o "logs/${JOB_NAME}_%J.out" \
            -e "logs/${JOB_NAME}_%J.err" \
            -n 4 \
            -q short \
            -gpu "num=1:j_exclusive=no" \
            -W 180 \
            -M 4000 \
            python train_mmfm.py \
            --seed ${seed} \
            --max_grad_norm ${max_grad_norm} \
            --p_unconditional ${p_unconditional} \
            --ns_per_t_and_c ${ns_per_t_and_c} \
            --x_latent_dim ${x_latent_dim} \
            --time_embed_dim ${time_embed_dim} \
            --cond_embed_dim ${cond_embed_dim} \
            --conditional_model ${conditional_model} \
            --classifier_free ${classifier_free} \
            --embedding_type ${embedding_type} \
            --sum_time_embed ${sum_time_embed} \
            --sum_cond_embed ${sum_cond_embed} \
            --normalization ${normalization} \
            --init_weights ${init_weights} \
            --activation ${activation} \
            --lrs ${lrs} \
            --interpolation ${interpolation} \
            --n_epochs ${n_epochs} \
            --coupling ${coupling} \
            --affine_transform ${affine_transform} \
            --max_norm_embedding ${max_norm_embedding} \
            --batch_size ${batch_size} \
            --lr ${lr} \
            --flow_variance ${flow_variance} \
            --num_out_layers ${num_out_layers} \
            --optimizer_name ${optimizer_name} \
            --dgp ${dgp} \
            --train_test_split ${train_test_split} \
            --off_diagonal ${off_diagonal} \
            --data_std ${data_std} \
            --dimension ${dimension} \
            --spectral_norm ${spectral_norm} \
            --dropout ${dropout} \
            --conditional_bias ${conditional_bias} \
            --keep_constants ${keep_constants} \
            --model_type ${model_type} \
            --matching ${matching}
    fi
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
