#!/bin/bash

ml load Miniforge3/24.1.2-0
conda activate mmfm
cd ${HOME}/code/mmfm/benchmark/dgp_iccite

# Create logs directory
mkdir -p logs

# Define arrays for each parameter
SEEDS=(0 1 2)
MAX_GRAD_NORM=(false)
P_UNCONDITIONAL=(0.0 0.1 0.2 0.3) 
N_SAMPLES_PER_C_IN_B=(100 250)
X_LATENT_DIM=(32 64 128)
TIME_EMBED_DIM=(32 64 128)
COND_EMBED_DIM=(32 64 128)
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
TRAIN_TEST_SPLIT=(0.8)
LR=(0.02 0.01 0.001 0.0001)
FLOW_VARIANCE=(1.0 0.1 0.01)
OPTIMIZER_NAME=("adam")
DGP=("a")
HVG=("None")
USE_PCA=(10 25)
SUBSAMPLE_FRAC=("None")
NUM_OUT_LAYERS=(3)
SPECTRAL_NORM=(false)
DROPOUT=(0.0)
CONDITIONAL_BIAS=(false)
KEEP_CONSTANTS=(false)
TOP_N_EFFECTS=("None")
LEAVE_OUT_MID=("None")
LEAVE_OUT_END=("None")
PRESET=("z")
CLASSIFIER_FREE=(false)
MODEL_TYPE=("fm" "mmfm")
MATCHING=("None")


# Loop over all parameter combinations
for seed in "${SEEDS[@]}"; do
for max_grad_norm in "${MAX_GRAD_NORM[@]}"; do
for p_unconditional in "${P_UNCONDITIONAL[@]}"; do
for n_samples_per_c_in_b in "${N_SAMPLES_PER_C_IN_B[@]}"; do
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
for hvg in "${HVG[@]}"; do
for use_pca in "${USE_PCA[@]}"; do
for subsample_frac in "${SUBSAMPLE_FRAC[@]}"; do
for num_out_layers in "${NUM_OUT_LAYERS[@]}"; do
for spectral_norm in "${SPECTRAL_NORM[@]}"; do
for dropout in "${DROPOUT[@]}"; do
for conditional_bias in "${CONDITIONAL_BIAS[@]}"; do
for keep_constants in "${KEEP_CONSTANTS[@]}"; do
for top_n_effects in "${TOP_N_EFFECTS[@]}"; do
for leave_out_mid in "${LEAVE_OUT_MID[@]}"; do
for leave_out_end in "${LEAVE_OUT_END[@]}"; do
for preset in "${PRESET[@]}"; do
for classifier_free in "${CLASSIFIER_FREE[@]}"; do
for model_type in "${MODEL_TYPE[@]}"; do
for matching in "${MATCHING[@]}"; do

    if [ $x_latent_dim -eq 128 ] && ([ $time_embed_dim -lt 32 ] || [ $cond_embed_dim -lt 32 ]); then
        continue
    fi
    if [ $x_latent_dim -eq 256 ] && ([ $time_embed_dim -lt 64 ] || [ $cond_embed_dim -lt 64 ]); then
        continue
    fi

    # If model_type is fm and interpolation is not linear, skip
    if [ $model_type == "fm" ] && [ $interpolation != "cubic" ]; then
        continue
    fi

    # Create a unique job name
    JOB_NAME="dgp_iccite_${dgp}_${seed}_${lr}_${flow_variance}_${num_out_layers}_${max_grad_norm}_${p_unconditional}_${x_latent_dim}_${time_embed_dim}_${cond_embed_dim}_${conditional_model}_${classifier_free}_${embedding_type}_${sum_time_embed}_${sum_cond_embed}_${normalization}_${affine_transform}_${max_norm_embedding}_${init_weights}_${activation}_${lrs}_${interpolation}_${n_epochs}_${coupling}_${batch_size}_${train_test_split}_${hvg}_${use_pca}_${n_samples_per_c_in_b}_${subsample_frac}_${optimizer_name}_${top_n_effects}_${leave_out_mid}_${leave_out_end}_${preset}_${spectral_norm}_${dropout}_${conditional_bias}_${keep_constants}_${matching}_${model_type}"
    JOB_NAME=$(echo "$JOB_NAME" | sed 's/true/True/g; s/false/False/g')

    # echo "Hello! Submitting job ${JOB_NAME}"

    # Check if results_mmfm / JOB_NAME /  "df_results.csv"  exists, if yes, skip else submit the job
    if [ -f "/home/rohbeckm/scratch/results/dgp_iccite/results_mmfm/${JOB_NAME}/df_results.csv" ]; then
        echo "Job already completed, skipping..."        
        continue
    else
        echo "Job not found, submitting..."
        # Submit the job
        bsub -J "${JOB_NAME}" \
            -o "logs/${JOB_NAME}_%J.out" \
            -e "logs/${JOB_NAME}_%J.err" \
            -n 4 \
            -q preempt \
            -gpu "num=1:j_exclusive=no" \
            -W 180 \
            -M 4000 \
            python train_mmfm.py \
            --seed ${seed} \
            --max_grad_norm ${max_grad_norm} \
            --p_unconditional ${p_unconditional} \
            --n_samples_per_c_in_b ${n_samples_per_c_in_b} \
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
            --hvg ${hvg} \
            --use_pca ${use_pca} \
            --subsample_frac ${subsample_frac} \
            --spectral_norm ${spectral_norm} \
            --dropout ${dropout} \
            --conditional_bias ${conditional_bias} \
            --keep_constants ${keep_constants} \
            --top_n_effects ${top_n_effects} \
            --leave_out_mid ${leave_out_mid} \
            --leave_out_end ${leave_out_end} \
            --preset ${preset} \
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
done
done
done
done