#!/bin/bash

# Function to count number of running jobs
count_jobs() {
    jobs -p | wc -l
}

# Maximum number of parallel jobs
MAX_JOBS=32

for interpolation in "cubic" "linear"; do
for seed in 0 1 2; do
for max_grad_norm in false; do
for p_unconditional in 0.1 0.2 0.3; do   # 0.0 
for ns_per_t_and_c in 50; do  # 125
for x_latent_dim in 4 8 16 32; do  #64
for time_embed_dim in 4 8 16 32; do  #64
for cond_embed_dim in 4 8 16 32; do  #64
for lr in 0.02 0.01 0.001; do  # 0.005
for flow_variance in 1.0 0.1 0.01; do   #  0.001
for conditional_model in true; do
for embedding_type in "free"; do
for sum_time_embed in false; do
for sum_cond_embed in false; do
for normalization in "None"; do
for affine_transform in false; do
for max_norm_embedding in true; do
for init_weights in "xavier_normal"; do
for activation in "LeakyReLU"; do
for lrs in "cosine"; do
for n_epochs in 300; do
for coupling in "cot"; do
for batch_size in "None"; do
for train_test_split in 0.5; do
for optimizer_name in "adam"; do
for dgp in "a"; do  # "b"
for dimension in 1; do
for num_out_layers in 3; do
for spectral_norm in false; do
for dropout in 0.0; do
for conditional_bias in false; do
for keep_constants in false; do
for classifier_free in false; do
for model_type in "mmfm"; do
for matching in "emd"; do

    if [ $model_type == "fm" ] && [ $interpolation != "cubic" ]; then
        continue
    fi
    if [ $x_latent_dim -eq 64 ] && [ $time_embed_dim -lt 16 ]; then
        continue
    fi
    if [ $x_latent_dim -eq 64 ] && [ $cond_embed_dim -lt 16 ]; then
        continue
    fi
    if [ $x_latent_dim -eq 32 ] && [ $time_embed_dim -lt 4 ]; then
        continue
    fi
    if [ $x_latent_dim -eq 32 ] && [ $cond_embed_dim -lt 4 ]; then
        continue
    fi

    JOB_NAME="dgp_weather_${dgp}_${seed}_${lr}_${flow_variance}_${num_out_layers}_${max_grad_norm}_${p_unconditional}_${ns_per_t_and_c}_${x_latent_dim}_${time_embed_dim}_${cond_embed_dim}_${normalization}_${init_weights}_${activation}_${lrs}_${interpolation}_${conditional_model}_${classifier_free}_${embedding_type}_${sum_time_embed}_${sum_cond_embed}_${n_epochs}_${coupling}_${affine_transform}_${max_norm_embedding}_${batch_size}_${train_test_split}_${dimension}_${optimizer_name}_${spectral_norm}_${dropout}_${conditional_bias}_${keep_constants}_${matching}_${model_type}"
    JOB_NAME=$(echo "$JOB_NAME" | sed 's/true/True/g; s/false/False/g')

    if [ -f "/data/m015k/results/dgp_weather/results_mmfm/${JOB_NAME}/df_results.csv" ]; then
        echo "Job found, skipping... :)"
    else
        # Wait if we already have MAX_JOBS running
        while [ $(count_jobs) -ge $MAX_JOBS ]; do
            sleep 1
        done

        echo "Job not found, submitting..."
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
            --dimension ${dimension} \
            --spectral_norm ${spectral_norm} \
            --dropout ${dropout} \
            --conditional_bias ${conditional_bias} \
            --keep_constants ${keep_constants} \
            --model_type ${model_type} \
            --matching ${matching} &
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

# Wait for all remaining jobs to complete
wait
