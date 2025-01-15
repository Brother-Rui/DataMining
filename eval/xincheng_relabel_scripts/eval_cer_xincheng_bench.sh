source /nfs/volume-242-2/miniconda3n/bin/activate
conda activate qwen2audio


#checkpoint-0-6909  checkpoint-1-13818  checkpoint-2-20727  checkpoint-3-27636  checkpoint-4-34545
gpu_id=0
echo $gpu_id
experiment_name=Qwen2Audio-Train-100h_merge_without_insertion-1epoch-4batch-train-first_last2_attn+projector
log_folder=/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/logs/${experiment_name}
log_name=$(date +"%m-%d_%H-%M").log

result_folder=/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/results/${experiment_name}
mkdir -p ${log_folder}
touch ${log_folder}/${log_name}

mkdir -p ${result_folder}
export CUDA_VISIBLE_DEVICES=$gpu_id
python /nfs/volume-242-2/dengh/Qwen2-audio/src/eval/eval_cer.py \
    --tag asr \
    --model_path /nfs/volume-242-5/dengh/ckpts/Qwen2Audio-Train-100h_merge_without_insertion-5epoch-4batch-train-first_last2_attn+projector/checkpoint-0-6909/tfmr \
    --result_path ${result_folder}/secret_data_qwen2aduio_asr_clean.txt \
    --final_result_path ${result_folder}/secret_data_qwen2aduio_asr_extracted.txt \
    --label_path /nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/bench/xincheng_relabel_bench_clean.json \
    --online_asr_path /nfs/dataset-411-391/guoruizheng/${cls}/small_asr.json > ${log_folder}/$log_name 2>&1 &

gpu_id=1
echo $gpu_id
experiment_name=Qwen2Audio-Train-100h_merge_without_insertion-2epoch-4batch-train-first_last2_attn+projector
log_folder=/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/logs/${experiment_name}
log_name=$(date +"%m-%d_%H-%M").log

result_folder=/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/results/${experiment_name}
mkdir -p ${log_folder}
touch ${log_folder}/${log_name}

mkdir -p ${result_folder}
export CUDA_VISIBLE_DEVICES=$gpu_id
python /nfs/volume-242-2/dengh/Qwen2-audio/src/eval/eval_cer.py \
    --tag asr \
    --model_path /nfs/volume-242-5/dengh/ckpts/Qwen2Audio-Train-100h_merge_without_insertion-5epoch-4batch-train-first_last2_attn+projector/checkpoint-1-13818/tfmr \
    --result_path ${result_folder}/secret_data_qwen2aduio_asr_clean.txt \
    --final_result_path ${result_folder}/secret_data_qwen2aduio_asr_extracted.txt \
    --label_path /nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/bench/xincheng_relabel_bench_clean.json \
    --online_asr_path /nfs/dataset-411-391/guoruizheng/${cls}/small_asr.json > ${log_folder}/$log_name 2>&1 &


gpu_id=2
echo $gpu_id
experiment_name=Qwen2Audio-Train-100h_merge_without_insertion-3epoch-4batch-train-first_last2_attn+projector
log_folder=/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/logs/${experiment_name}
log_name=$(date +"%m-%d_%H-%M").log

result_folder=/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/results/${experiment_name}
mkdir -p ${log_folder}
touch ${log_folder}/${log_name}

mkdir -p ${result_folder}
export CUDA_VISIBLE_DEVICES=$gpu_id
python /nfs/volume-242-2/dengh/Qwen2-audio/src/eval/eval_cer.py \
    --tag asr \
    --model_path /nfs/volume-242-5/dengh/ckpts/Qwen2Audio-Train-100h_merge_without_insertion-5epoch-4batch-train-first_last2_attn+projector/checkpoint-2-20727/tfmr \
    --result_path ${result_folder}/secret_data_qwen2aduio_asr_clean.txt \
    --final_result_path ${result_folder}/secret_data_qwen2aduio_asr_extracted.txt \
    --label_path /nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/bench/xincheng_relabel_bench_clean.json \
    --online_asr_path /nfs/dataset-411-391/guoruizheng/${cls}/small_asr.json > ${log_folder}/$log_name 2>&1 &

gpu_id=3
echo $gpu_id
experiment_name=Qwen2Audio-Train-100h_merge_without_insertion-4epoch-4batch-train-first_last2_attn+projector
log_folder=/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/logs/${experiment_name}
log_name=$(date +"%m-%d_%H-%M").log

result_folder=/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/results/${experiment_name}
mkdir -p ${log_folder}
touch ${log_folder}/${log_name}

mkdir -p ${result_folder}
export CUDA_VISIBLE_DEVICES=$gpu_id
python /nfs/volume-242-2/dengh/Qwen2-audio/src/eval/eval_cer.py \
    --tag asr \
    --model_path /nfs/volume-242-5/dengh/ckpts/Qwen2Audio-Train-100h_merge_without_insertion-5epoch-4batch-train-first_last2_attn+projector/checkpoint-3-27636/tfmr \
    --result_path ${result_folder}/secret_data_qwen2aduio_asr_clean.txt \
    --final_result_path ${result_folder}/secret_data_qwen2aduio_asr_extracted.txt \
    --label_path /nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/bench/xincheng_relabel_bench_clean.json \
    --online_asr_path /nfs/dataset-411-391/guoruizheng/${cls}/small_asr.json > ${log_folder}/$log_name 2>&1 &

gpu_id=4
echo $gpu_id
experiment_name=Qwen2Audio-Train-100h_merge_without_insertion-5epoch-4batch-train-first_last2_attn+projector
log_folder=/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/logs/${experiment_name}
log_name=$(date +"%m-%d_%H-%M").log

result_folder=/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/results/${experiment_name}
mkdir -p ${log_folder}
touch ${log_folder}/${log_name}

mkdir -p ${result_folder}
export CUDA_VISIBLE_DEVICES=$gpu_id
python /nfs/volume-242-2/dengh/Qwen2-audio/src/eval/eval_cer.py \
    --tag asr \
    --model_path /nfs/volume-242-5/dengh/ckpts/Qwen2Audio-Train-100h_merge_without_insertion-5epoch-4batch-train-first_last2_attn+projector/checkpoint-4-34545/tfmr \
    --result_path ${result_folder}/secret_data_qwen2aduio_asr_clean.txt \
    --final_result_path ${result_folder}/secret_data_qwen2aduio_asr_extracted.txt \
    --label_path /nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/bench/xincheng_relabel_bench_clean.json \
    --online_asr_path /nfs/dataset-411-391/guoruizheng/${cls}/small_asr.json > ${log_folder}/$log_name 2>&1 &
