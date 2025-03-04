import os
import jsonlines
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import argparse

parser = argparse.ArgumentParser("The data prepare script")
parser.add_argument("--medxpertqa_root", type=str, default=None,
                    help="The root of the MedXpertQA Text dataset.")
parser.add_argument("--medqa_usmle_root", type=str, required=True,
                    help="The root of the MedQA-USMLE dataset")
parser.add_argument("--usmle_nums", type=int, default=1090,
                    help="The number of the samples to select from the MedQA-USMLE dataset")
parser.add_argument("--xpert_test_size", type=float, default=0.8,
                    help="the ratio the test dataset for MedXpertQA dataset, we sample it to eval the model's performance.")
parser.add_argument("--output_dir", type=str, default="./data/medxpert_usmle")

args = parser.parse_args()

def convert2dataframe(data):
    #convert the data to pandas DataFrame
    out_dict = {}
    for line in data:
        for k in line.keys():
            if k not in out_dict:
                out_dict[k] = [line[k]]
            else:
                out_dict[k].append(line[k])
    return pd.DataFrame(out_dict)

medxpertqa_root = args.medxpertqa_root
output_dir = args.output_dir
usmle_root = args.medqa_usmle_root
xpert_test_size = args.xpert_test_size
usmle_nums = args.usmle_nums
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
train_set = None
if medxpertqa_root is not None:
    with open(os.path.join(medxpertqa_root, "Text", "test.jsonl"), encoding="utf-8") as f:
        data = jsonlines.Reader(f)
        datas = []
        for line in data:
            datas.append(line)

    dataset = []
    types = []
    data_map = {}
    type_id = 0
    for data_item in datas:
        question = data_item["question"]
        data_type = data_item["question_type"]
        label = data_item["label"]
        options = data_item["options"]
        data = {
            "question": question,
            "answer": label,
            "option": options 
        }
        dataset.append(data)
        if data_type not in data_map:
            data_map[data_type] = type_id
            type_id += 1
        types.append(
            data_map[data_type]
        )

    train_set, test_set = train_test_split(dataset, test_size=xpert_test_size, shuffle=True, stratify=types)
    val_frame = convert2dataframe(test_set)
    with open(os.path.join(output_dir, "test.jsonl"), encoding="utf-8", mode="w") as f:
        jsonlines.Writer(f).write_all(test_set)
    val_frame.to_parquet(os.path.join(output_dir, "test.parquet"))


med_qa_path = os.path.join(usmle_root, "train.jsonl")
with open(med_qa_path, encoding="utf-8") as f:
    med_qa_data = jsonlines.Reader(f)
    med_qa_lines = []
    for line in med_qa_data:
        med_qa_lines.append(line)

select_data = np.random.choice(med_qa_lines, usmle_nums)
out_datas = []
for data in select_data:
    question = data["question"]
    options = data["options"]
    opt_str = "\nAnswer Choices: "
    for opt in options.keys():
        opt_str += f"({opt}) {options[opt]} "
    question = question + opt_str
    label = data["answer_idx"]
    options = data["answer"]
    out_data = {
        "question": question,
        "answer": label,
        "option": options 
    }
    out_datas.append(out_data)
if train_set:
    final_dataset = out_datas.extend(train_set)
else:
    final_dataset = out_datas
    
with open(os.path.join(output_dir, "train.jsonl"), encoding="utf-8", mode="w") as f:
    jsonlines.Writer(f).write_all(final_dataset)

train_frame = convert2dataframe(final_dataset)
train_frame.to_parquet(os.path.join(output_dir, "train.parquet"))
