# Computation of CER
source /nfs/volume-411-10/jeremyyang/miniconda3/bin/activate funasr_1.1.6

#before clean
# experiment_name=Qwen2Audio-Train-merge-new1-5epoch-4batch-train-last2+projector-Test
# log_folder=/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/results/${experiment_name}
# label_path=/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/bench/label_cer.json
# label_s1_path=/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/bench/label_cer_s1.json
# label_g1_path=/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/bench/label_cer_g1.json
# pred_path=/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/results/${experiment_name}/secret_data_qwen2aduio_asr.txt
# pred_s1_path=/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/results/${experiment_name}/secret_data_qwen2aduio_asr_s1.txt
# pred_g1_path=/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/results/${experiment_name}/secret_data_qwen2aduio_asr_g1.txt
# json_path='/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/bench/xincheng_relabel_bench.json'

# python /nfs/volume-242-2/dengh/Qwen2-audio/src/secret_data_eval/split_res.py --json_path ${json_path} --model_txt_path ${pred_path}
# python /nfs/volume-242-2/dengh/Qwen2-audio/src/eval/compute_wer.py --char=1 ${label_path} ${pred_path} > ${log_folder}/asr_cer.txt
# python /nfs/volume-242-2/dengh/Qwen2-audio/src/eval/compute_wer.py --char=1 ${label_s1_path} ${pred_s1_path} > ${log_folder}/asr_cer_s1.txt
# python /nfs/volume-242-2/dengh/Qwen2-audio/src/eval/compute_wer.py --char=1 ${label_g1_path} ${pred_g1_path} > ${log_folder}/asr_cer_g1.txt


#clean
experiment_name=Qwen2Audio-Train-100h_merge_without_insertion-1epoch-4batch-train-first_last2_attn+projector
log_folder=/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/results/${experiment_name}
label_path=/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/bench/label_cer_clean.txt
label_s1_path=/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/bench/label_cer_s1_clean.txt
label_g1_path=/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/bench/label_cer_g1_clean.txt
pred_path=/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/results/${experiment_name}/secret_data_qwen2aduio_asr_clean.txt
pred_s1_path=/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/results/${experiment_name}/secret_data_qwen2aduio_asr_s1_clean.txt
pred_g1_path=/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/results/${experiment_name}/secret_data_qwen2aduio_asr_g1_clean.txt
json_path='/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/bench/xincheng_relabel_bench_clean.json'

python /nfs/volume-242-2/dengh/Qwen2-audio/src/eval/split_res.py --json_path ${json_path} --model_txt_path ${pred_path}
python /nfs/volume-242-2/dengh/Qwen2-audio/src/eval/compute_wer.py --char=1 ${label_path} ${pred_path} > ${log_folder}/asr_cer_clean.txt
python /nfs/volume-242-2/dengh/Qwen2-audio/src/eval/compute_wer.py --char=1 ${label_s1_path} ${pred_s1_path} > ${log_folder}/asr_cer_s1_clean.txt
python /nfs/volume-242-2/dengh/Qwen2-audio/src/eval/compute_wer.py --char=1 ${label_g1_path} ${pred_g1_path} > ${log_folder}/asr_cer_g1_clean.txt

# pred_path2=/nfs/dataset-411-391/guoruizheng/results/${experiment_name}/${cls}/secret_data_qwen2aduio_asrcor.txt
# python /nfs/volume-242-2/dengh/Qwen2-audio/src/eval/compute_wer.py --char=1 ${label_path} ${pred_path2} > /nfs/dataset-411-391/guoruizheng/results/${experiment_name}/${cls}/asrcor_cer.txt

# Qwen2Audio-Train-100h_merge_without_insertion-4epoch-4batch-train-first_last2_attn+projector
# Qwen2Audio-Train-100h_merge_without_insertion-3epoch-4batch-train-first_last2_attn+projector
# Qwen2Audio-Train-100h_merge_without_insertion-2epoch-4batch-train-first_last2_attn+projector
# Qwen2Audio-Train-100h_merge_without_insertion-1epoch-4batch-train-first_last2_attn+projector