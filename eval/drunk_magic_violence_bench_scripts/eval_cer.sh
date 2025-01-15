source /nfs/volume-411-10/jeremyyang/miniconda3/bin/activate funasr_1.1.6

# source /nfs/volume-242-2/miniconda3n/bin/activate
# conda activate qwen2audio

#N C R S D I
#all cor sub del ins

gpu_id=5
for cls in drunk magic violence
do  
    echo $gpu_id
    experiment_name=Qwen2Audio-Train-100h_merge_without_insertion-5epoch-4batch-train-first_last2_attn+projector
    log_folder=/nfs/dataset-411-391/guoruizheng/logs1/${experiment_name}/${cls}
    log_name=$(date +"%m-%d_%H-%M").log

    result_folder=/nfs/dataset-411-391/guoruizheng/results/${experiment_name}
    mkdir -p ${log_folder}
    touch ${log_folder}/${log_name}

    mkdir -p ${result_folder}/${cls}
    export CUDA_VISIBLE_DEVICES=$gpu_id
    python /nfs/volume-242-2/dengh/Qwen2-audio/src/eval/eval_cer.py \
        --tag asr \
        --model_path /nfs/volume-242-5/dengh/ckpts/Qwen2Audio-Train-100h_merge_without_insertion-5epoch-4batch-train-first_last2_attn+projector/checkpoint-4-34545/tfmr \
        --result_path ${result_folder}/${cls}/secret_data_qwen2aduio_asr.txt \
        --final_result_path ${result_folder}/${cls}/secret_data_qwen2aduio_asr_extracted.txt \
        --label_path /nfs/dataset-411-391/guoruizheng/${cls}/label.json \
        --online_asr_path /nfs/dataset-411-391/guoruizheng/${cls}/small_asr.json > ${log_folder}/$log_name 2>&1 &       
    gpu_id=$((gpu_id+1))
done
