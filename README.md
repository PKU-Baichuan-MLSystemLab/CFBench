# CFBench: A Comprehensive Constraints-Following Benchmark for LLMs

## 1. Dataset
Typical case illustration  
![Case](https://github.com/zhangtao-tanh/CFBench/blob/main/resources/img/1_introduction_case.png)     

Construction Pipline    
![Construction Pipline](https://github.com/zhangtao-tanh/CFBench/blob/main/resources/img/2_pipline.png)

An example of CFBench. 
```
    {
        "idx": 46,
        "prompt": "目前，在儿童肝移植患者中抗菌药物的使用通常参考药品说明书或成人用法用量进行选择。然而，由于儿童在生理特点和药物代谢能力方面与成人存在显著差异，因此不能简单地将成人的给药方案应用于儿童。当前，国内外尚缺乏针对儿童肝移植患者的抗菌药物药动学参数相关数据，这使得说明书中的抗菌药物用法用量是否适合儿童肝移植患者成为一个需要探讨的问题。此外,研究表明不合理使用抗菌药物，特别是长期使用亚治疗浓度的抗菌药物，会诱导多重耐药菌的产生。因此，确定合理的儿童肝移植患者抗菌药物给药方案需要依赖儿童药代动力学（PK）数据。需要一种高效、简便且高灵敏度的分析方法来获取血浆或血清中的药物浓度数据，以进行药代动力学分析。 \n\n假设你是一位儿科药理学专家，你正在接受文字采访，以上是你回答问题的书面表达，现在对上述文本进行改写，请在回答时采用百科风格，确保内容详尽，同时回答中需包含至少三个与儿童药代动力学相关的关键术语，每个术语出现至少3次。并在解释时用到反问句和设问句这两种句式。回答必须以诗歌的形式呈现，且要符合严格的格律要求，同时不能影响专业内容的表达。",
        "isanswer": "否",
        "gold": "",
        "doamin": "医疗",
        "scenario": "其他",
        "source": "tob",
        "split": "hard",
        "criteria": [
            [
                "是以百科风格重写用户给定的文本",
                "主需",
                "内容约束",
                "其他内容约束"
            ],
            [
                "符合一位儿科药理学专家正在接受文字采访的背景设定",
                "次需",
                "场景约束",
                "任务场景约束"
            ],
            [
                "重写内容有至少三个与儿童药代动力学相关的关键术语，且每个术语出现至少3次",
                "次需",
                "数值约束",
                "单词数值约束"
            ],
            [
                "解释关键术语时使用了反问句和设问句这两种句式",
                "次需",
                "数值约束",
                "句子数值约束"
            ],
            [
                "回答是百科风格的文字采访，因此没有使用诗歌的形式呈现，并向用户进行了说明",
                "次需",
                "不合理约束",
                "不合理约束"
            ]
        ]
    },
```

## 2. Result
full leaderboard    
![leaderboard](https://github.com/zhangtao-tanh/CFBench/blob/main/resources/img/leaderboard.png)

constraints results  
![constraints_results](https://github.com/zhangtao-tanh/CFBench/blob/main/resources/img/4_constraints_results.png)

domains and nlp tasks results    
![domain_nlp_results](https://github.com/zhangtao-tanh/CFBench/blob/main/resources/img/5_domain_nlp_results.png)


## 3. Evaluation
### step1: Environments
To set up the environment, use the following commands:
```
conda create -n CFBench python=3.11
conda activate CFBench
```
Then, install the required Python packages:
```
pip install -r requirements.txt
```
### step2: model inference
We provide implementations of the most popular LLMs, including the GPT series and Claude, as well as Chinese API models, and some open-source models.
```
sh run.sh
```
* run.sh
    ```
    infer_model=ernie35
    infer_max_threads=5
    infer_in_path=data/cfbench_data.json
    infer_out_dir=output/response
    python  code/inference.py \
        --infer_model ${infer_model} \
        --in_path ${infer_in_path} \
        --out_dir ${infer_out_dir}  \
        --max_threads ${infer_max_threads}
    ```
### step3: evaluate
```
sh run.sh
```
* run.sh
    ```
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
    ```

⚠️ Note: If the model you want to test is not in the models folder, which contains the models we have already implemented, you need to create **models/{module_name}.py** and implement the class **${module_name}**.

### step4: score
The evaluation results are stored in **output/score.xlsx**, and CSR, ISR, and PSR metrics are calculated for the easy, hard, and full sets.



## 4.Citation
```
@article{zhang2024cfbench,
  title={CFBench: A Comprehensive Constraints-Following Benchmark for LLMs},
  author={Zhang, Tao and Shen, Yanjun and Luo, Wenjing and Zhang, Yan and Liang, Hao and Yang, Fan and Lin, Mingan and Qiao, Yujing and Chen, Weipeng and Cui, Bin and others},
  journal={arXiv preprint arXiv:2408.01122},
  year={2024}
}
```
Please cite our paper if you find our research and code useful.