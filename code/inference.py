#!/usr/bin/env python
#coding:utf-8

import json, os, glob, random, sys, argparse
import concurrent.futures
import traceback
import threading 
import importlib
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from models import *

class CFBench():
    def __init__(self, infer_model, in_path, out_dir, para_num):
        self.infer_model = infer_model
        self.out_path = os.path.join(out_dir, f"{infer_model}_infer.json")
        self.in_path = in_path
        self.example_num = 0
        self.para_num = para_num
        self.model = self._get_model(self.infer_model)
    
    def _get_model(self, model_name):
        try:
            module = importlib.import_module(f"models.{model_name}")
            model_class = getattr(module, model_name)
            print(f"module:{module}, model_class:{model_class}")
            return model_class()
        except (ImportError, AttributeError) as e:
            raise ValueError(f"model_name:'{model_name}' is not defined: {e}")
        except Exception as e:
            print(f'error:{e}')
    
    def _load_examples(self, in_path):
        try:
            data = json.load(open(in_path,"r",encoding="utf-8"))
            self.example_num = len(data)
            return data
        except:
            raise ValueError(f"Dataset eroor, please check data or in_path")
    
    def _infer_one(self, task):
        prompt = task["prompt"]
        response = self.model(prompt)
        task['response'] = response
        task["infer_model"] = self.infer_model
        return task
    
    def _infer_parallel(self, tasks, para_num):
        results = []
        with ThreadPoolExecutor(para_num) as executor:
            for entry in tqdm(executor.map(self._infer_one, tasks), total=len(tasks),  \
                        desc=f'{self.infer_model} inference:'):
                results.append(entry)
        return results

    def _save_result(self, result):
        try:
            if not os.path.exists(self.out_path):
                os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
            # assert len(result) == self.example_num, "Data Error, Infer Result != Raw Data, pleace check inference program"
            json.dump(result, open(self.out_path,'w',encoding='utf-8'),ensure_ascii=False,indent=4)
        except Exception as e:
            print(f"save result error, {e}")

    def __call__(self):
        datas = self. _load_examples(self.in_path)
        result = self._infer_parallel(datas, self.para_num)
        self._save_result(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_model", type=str, default="moonshot")
    parser.add_argument("--in_path", type=str, default="../data/cfbench_data.json")
    parser.add_argument("--out_dir", type=str, default="../output/respone")
    parser.add_argument("--max_threads", type=int, default=10)
    args = parser.parse_args()
    cfbench_infer = CFBench(args.infer_model, args.in_path, args.out_dir, args.max_threads)
    cfbench_infer()