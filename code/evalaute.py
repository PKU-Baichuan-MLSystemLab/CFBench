#!/usr/bin/env python
#coding:utf-8

import os, glob, random, json, re, time
from tqdm import tqdm
import numpy as np
import pandas as pd
import importlib
from concurrent.futures import ThreadPoolExecutor
import arrow
import argparse
random.seed(9876)

gpt_judge_for_cfbench = lambda prompt, gold_ans, response, criteria: f'''
我想让你扮演一个质量评估器的角色。你需要结合 [用户指令],[参考答案],[模型答案],[考察点]对于模型答案进行评估,按照0或1打分.
具体而言:[参考答案]和[模型答案]都是对[用户指令]的回复,[考察点]中定义了模型答案应该满足、需要被考察的点。你需要严格按照[考察点]中的每条考察点，对[于模型答案]进行打分。如果完全满足则为1，如果不满足则为0.
需要注意: [参考答案]如果为空,则[考察点]的满足不考虑[参考答案].
输出格式: 1.严格按照[考察点]中的考察点的顺序,每行输出一条，行与行之间使用"\n\n"切割;  
        2.每行首先输出[考察点]中对应内容，然后用'\t'分割，在后面输出对应[评估打分],0或1。 
        3.请直接输出你的评估,不要输出其他任何内容。

可参考下面样例。

[样例1]:
    [用户指令]: 请对给定的句子\"我爱吃苹果\"做以下编辑步骤\n\n1.将句子中的“我”改成“小明”。\n\n只输出编辑后的句子。
    
    [参考答案]: "小明爱吃苹果"
    
    [模型答案]: "好的，修改后为：小明爱吃苹果"

    [评估标准]:
    和参考答案结果一致，替换准确\n\n不存在其他冗余信息，引号可忽略
    
    [你的评估]: 
    和参考答案结果一致，替换准确\t1\n\n不存在其他冗余信息，引号可忽略\t0

[样例2]:
    [用户指令]: 给我生成一些唐代诗人李白的诗句，主题是友情。输出结果应为JSON格式，包含两个key："诗句"和"诗名"。提供三组不同的诗句，每个诗句控制在5个字，一组10个字。

    [参考答案]: ""
    
    [模型答案]: '"诗句": ["诗句": "明月照友心", "诗名": "友情如月","诗句": "高山流水情", "诗名": "知音难觅","诗句": "酒逢知己饮", "诗名": "醉卧沙场笑"]'

    [评估标准]: 
    生成唐代诗人李白的诗句。\n\n诗句主题是友情。\n\n输出结果应为JSON格式，JSON格式包含两个key：""诗句""和""诗名""。\n\n提供三组不同的诗句。\n\n每个诗句控制在5个字，一组10个字。

    [你的评估]: 
    生成唐代诗人李白的诗句。\t0\n\n诗句主题是友情。\t1\n\n输出结果应为JSON格式，JSON格式包含两个key：""诗句""和""诗名""。\t0\n\n提供三组不同的诗句。\t1\n\n每个诗句控制在5个字，一组10个字。\t0

[用户指令]: {prompt}

[参考答案]: {gold_ans}

[模型答案]: {response}

[评估标准]: {criteria}

[你的评估]: 
'''

class Evaluation():
    def __init__(self, infer_model, in_dir, out_dir, score_path, para_num, eval_model="gpt4o", temperature=0.01):
        self.infer_model = infer_model
        self.eval_model = eval_model
        self.in_path = os.path.join(in_dir, f"{infer_model}_infer.json")
        self.out_path = os.path.join(out_dir, f"{infer_model}_eval.json")
        self.score_path = score_path
        self.para_num = para_num
        self.temperature = temperature
        self.eval_model = self._get_model(self.eval_model)
        print(f"infer_model:{self.infer_model}\t in_path:{self.in_path}\t out_path:{self.out_path}\t" 
              f"score_path:{self.score_path}\t para_num:{self.para_num}\t temperature:{self.temperature}")

    def _get_model(self, model_name):
        try:
            module = importlib.import_module(f"models.{model_name}")
            model_class = getattr(module, model_name)
            return model_class(temperature=self.temperature)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"{model_name} is not defined: {e}, Please ensure that module_name is equal to model_name and that both have been defined.")
        except Exception as e:
            print(f'error:{e}')

    def _get_judge_scores(self, task):
        judge_response = task["judge_response"]
        criteria = [crit[0] for crit in task["criteria"]]
        scores = []
        try:
            judges = judge_response.split("\n\n")
            assert len(criteria) == len(judges)         # 判断1:评估条目和原来数量对不上;
            for idx, judge in enumerate(judges):
                crit, score = judge.split("\t")         # 判断2:评估条目\t没有切分
                assert crit == criteria[idx]            # 判断3:gpt4评估的返回结果中criteria和原始task中的criteria对不上;
                assert float(score) in [0, 1]           # 判断4:score不在不是0or1          
                scores.append(score.strip())
        except:
            return None
        return scores

    def _get_judge_scores_guaranteed(self, task):
        judge_response = task["judge_response"]
        scores = []
        judges = judge_response.split("\n")
        for idx, judge in enumerate(judges):
            if len(judge.split("\t")) == 2:
                crit, score = judge.split("\t") 
                if float(score) and float(score) in [0, 1]:
                    scores.append(score.strip())
        return scores


    def _judge_by_gpt(self, inputs):
        prompt, gold_ans, response, criteria = inputs
        formated_input = gpt_judge_for_cfbench(prompt, gold_ans, response, criteria)   # todo:more reasonable prompt formated
        MAX_RETRY_NUM = 10
        for i in range(MAX_RETRY_NUM):
            try:
                response = self.eval_model(formated_input)
                return response
            except Exception as e:
                print(f"Error in gpt judge, retrying...{i}/{MAX_RETRY_NUM}")
                print(e)
                continue
        print(f"Failed after {MAX_RETRY_NUM} retries.")
        return 'Error'


    def _judge_one(self, task):    
        prompt, gold_ans, response = task['prompt'], task['gold'], task['response']
        criteria = "\n\n".join([crit[0] for crit in task['criteria']])
        inputs = (prompt, gold_ans, response, criteria)
        completion = self._judge_by_gpt(inputs)                  # 2
        task['judge_response'] = completion
        return task

    def _judged_parallel(self, tasks, para_num):
        results = []
        with ThreadPoolExecutor(para_num) as executor:
            for entry in tqdm(executor.map(self._judge_one, tasks), total=len(tasks), desc=f'eval {self.infer_model}'):
                results.append(entry)
                if len(results) % 500 == 0: print(f"results:{results[-1]}")
        if None in results:
            raise ValueError("Some entries are not annotated due to errors in judge_one, please inspect and retry.")
        return results

    def _judged(self, tasks, para_num):
        MAX_RETRY_NUM = 10
        tasks_remained, tasks_judged = tasks, []
        for _ in range(MAX_RETRY_NUM):
            tasks_judged_ = self._judged_parallel(tasks_remained, para_num)         
            tasks_remained = []
            for task in tasks_judged_:
                if self._get_judge_scores(task):
                    tasks_judged.append(task)
                    task['judge_score'] = self._get_judge_scores(task)
                    task['judge_parsing'] = "1"
                else:
                    tasks_remained.append(task)

            if len(tasks_remained) == 0:
                break
            else:
                print(f"try:{_}/{MAX_RETRY_NUM}, Still {len(tasks_remained)} tasks remained to be judged. try...")

        if len(tasks_remained) > 0:
            print(f"Max retries ({MAX_RETRY_NUM}) reached. The model's response may lack a valid answer.")
            for task in tasks_remained:
                task['judge_score'] = self._get_judge_scores_guaranteed(task)
                if not task['judge_score']:
                    task['judge_score'] = ["0"]     # todo: more reasonable approach
                    task['judge_parsing'] = "-1"
                else:
                    task['judge_parsing'] = "0"
                tasks_judged.append(task)
        
        assert len(tasks_judged) == len(tasks), "The number of judged tasks doesn't match the input tasks."
        return tasks_judged

    def _score_compute(self, task):
        judge_score = [float(score) for score in task["judge_score"]]
        csr, isr, psr = 0, 0, 0
        csr = round(np.mean(judge_score),2)
        isr = 0 if csr < 1 else 1
        if task["judge_parsing"] == "-1":
            psr = 0
        elif task["judge_parsing"] == "0":
            psr = 1 if csr > 0.8 else 0
        elif task["judge_parsing"] == "1":
            assert len(task["criteria"]) == len(judge_score), "judge score error"
            primary, secondary = [], []
            for idx, crit in enumerate(task["criteria"]):
                assert crit[1] in ["主需", "次需"]
                if crit[1] == "主需":
                    primary.append(judge_score[idx])
                elif crit[1] == "次需":
                    secondary.append(judge_score[idx])
            if 0 in primary:
                psr = 0
            else:
                if len(primary) == 0 and (len(secondary) == len(judge_score)):     # 全是次需
                    psr = 1 if csr > 0.8 else 0
                elif len(primary) == len(judge_score):                             #  全是主需
                    psr = 1
                elif secondary:                                                    # 主需+次需
                    tmp = 0.5 + 0.5 * round(np.mean(secondary),2)
                    psr = 1 if tmp > 0.8 else 0
                else:
                    raise ValueError("psr value error")
        else:
            raise ValueError("judge_parsing value error")

        judge_metric = {"CSR": csr, "ISR": isr, "PSR": psr}

        return judge_metric

    def _scores(self, results):
        # 分数统计
        CSRs, ISRs, PSRs = [], [], []
        CSRs_easy, ISRs_easy, PSRs_easy = [], [], []
        CSRs_hard, ISRs_hard, PSRs_hard = [], [], []
        for exm in results:
            if exm["split"] == "easy":
                CSRs_easy.append(exm["judge_metric"]["CSR"])
                ISRs_easy.append(exm["judge_metric"]["ISR"])
                PSRs_easy.append(exm["judge_metric"]["PSR"])
            elif exm["split"] == "hard":
                CSRs_hard.append(exm["judge_metric"]["CSR"])
                ISRs_hard.append(exm["judge_metric"]["ISR"])
                PSRs_hard.append(exm["judge_metric"]["PSR"])
            CSRs.append(exm["judge_metric"]["CSR"])
            ISRs.append(exm["judge_metric"]["ISR"])
            PSRs.append(exm["judge_metric"]["PSR"])
        CSR, ISR, PSR = round(np.mean(CSRs),2), round(np.mean(ISRs),2), round(np.mean(PSRs),2)
        CSR_e, ISR_e, PSR_e = round(np.mean(CSRs_easy),2), round(np.mean(ISRs_easy),2), round(np.mean(PSRs_easy),2)
        CSR_h, ISR_h, PSR_h = round(np.mean(CSRs_hard),2), round(np.mean(ISRs_hard),2), round(np.mean(PSRs_hard),2)

        current_time = arrow.now().to('Asia/Shanghai').format('YYYY-MM-DD-HH:mm:ss')
        final_score = {"time":[current_time], "name":[self.infer_model],
                    "CSR_easy":[CSR_e], "ISR_easy":[ISR_e], "PSR_easy":[PSR_e],
                    "CSR_hard":[CSR_h], "ISR_hard":[ISR_h], "PSR_hard":[PSR_h],
                    "CSR":[CSR], "ISR":[ISR], "PSR":[PSR]}
        print(f"final_score:{final_score}")

        return final_score
    

    def __call__(self):
        start_time = time.time()
        # judge
        data = json.load(open(self.in_path, "r", encoding="utf-8"))
        judge_results = self._judged(data, self.para_num)
        results = []
        for task in judge_results:
            task['judge_metric'] = self._score_compute(task)
            results.append(task)
        if not os.path.exists(os.path.dirname(self.out_path)):
            os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        json.dump(results, open(self.out_path, "w"), ensure_ascii=False, indent=4)
        
        # score
        final_score = self._scores(results)
        if not os.path.exists(self.score_path):
            with pd.ExcelWriter(self.score_path, engine='openpyxl') as writer:
                pd.DataFrame(final_score).to_excel(writer, sheet_name="score", startrow=0, startcol=0, index=False)
        else:
            df = pd.read_excel(self.score_path, sheet_name="score")
            df = pd.concat([df, pd.DataFrame(final_score)], ignore_index=True)
            df.to_excel(self.score_path, sheet_name="score", index=False)
        
        end_time = time.time()
        print(f"**** Evaluation Done, Total Cost {end_time-start_time} s*********")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_model", type=str, default="moonshot")
    parser.add_argument("--in_dir", type=str, default="../output/response")
    parser.add_argument("--out_dir", type=str, default="../output/judge")
    parser.add_argument("--score_path", type=str, default="../output/scores.xlsx")
    parser.add_argument("--max_threads", type=int, default=10)
    parser.add_argument("--eval_model", type=str, default='gpt4o')
    args = parser.parse_args()

    Evaluation(args.infer_model, args.in_dir, args.out_dir, args.score_path, args.max_threads)()