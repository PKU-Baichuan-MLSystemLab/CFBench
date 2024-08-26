infer_model=ernie35

infer_max_threads=5
infer_in_path=data/cfbench_data.json
infer_out_dir=output/response
python  code/inference.py \
    --infer_model ${infer_model} \
    --in_path ${infer_in_path} \
    --out_dir ${infer_out_dir}  \
    --max_threads ${infer_max_threads}


eval_out_dir=output/judge
eval_score_path=output/scores.xlsx
eval_max_threads=10
python  code/evalaute.py \
    --infer_model ${infer_model} \
    --in_dir ${infer_out_dir} \
    --out_dir ${eval_out_dir}  \
    --score_path ${eval_score_path} \
    --max_threads ${eval_max_threads} \
    --eval_model gpt4o