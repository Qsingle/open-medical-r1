# Medical R1-Zero Reproduce

## Model and Dataset

Similar to the paper [MED-RLVR](https://arxiv.org/pdf/2502.19655), we have tried the Qwen2.5-1.5B, Qwen2.5-3B, and Qwen2.5-7B as the base model and train by the GRPO. However, the model can not occur the "aha moment", which is the self-validation behaviour. The model's behaviour during training is similar to MED-RLVR, so we do not report the results of these models. We conjecture the model's behaviour is related to the knowledge and the training data. Thus, we use the [HuatuoGPT-o1-7B](https://huggingface.co/FreedomIntelligence/HuatuoGPT-o1-7B) as the base model to do the experiments on the following three datasets:

1. [MedQA-USMLE](https://github.com/jind11/MedQA): We randomly choose 1090 samples from the training dataset. We called it dataset1 in the following.
2. MedQA-USMLE+[MedXpertQA](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA): We randomly choose 600 samples from the training dataset of MedQA-USMLE and 490 samples from MedXpertQA. We called it dataset2 in the following.
3. MedXpertQA: We randomly choose 490 samples from the MedXpertQA dataset. We called it dataset3 in the following.

## Experiments

### Settings

We train the model on 4 Nvidia Tesla A100 40GB SXM GPUs and implement it based on the Open-R1. We removed the cosine reward and reasoning reward, and we found that the cosine reward may lead to the completion being shorter in our experiments. We hypothesise that the model makes it hard to sample the correct answer when the domain knowledge is lacking, and we think the model can give its reasoning steps.

### Metric Results

We evaluate the model at MedMCQA, MedQA-USMLE, PubMedQA, MMLU-Pro Medical, GPQA Medical, and MedXpertQA.

|      Model      | MedMCQA | MedQA-USMLE | PubMedQA | MMLU-Pro Medical | GPQA Medical | MedXpertQA |
| :-------------: | :---------:  | :----------: | :------: | :--------------: | :----------: | :--------: |
| HuatuoGPT-o1-7B | 63.57 | 71.56 | 78.50 | 67.17 | 52.56 |\|
|   Our(Dataset1)    | \ | \ | \ | \ | \ |\|
|   Our(Dataset2)    | \ | \ | \ | \ | \ |\|
|   Our(Dataset3)    | \ | \ | \ | \ | \ |\|

### Logs

Fig.1 shows the experiments conducted on the dataset1. From the picture, we can observe that the length of the competition is large, the reward is higher, and a decrease in the length may result in a lower reward.

![log_dataset1](assets/log_dataset1.png)

**Fig. 1 The training log for experiments on dataset1.**

Then, we also checked the model's output during training. We found a phenomenon like "Aha Moment" in Deepseek's technical report. Fig.2 is one sample; the red box labels the content that seems to be self-validation of the model. But the model may think of the problem twice, and we think it may caused by the base model. In our experiments, this phenomenon will resolved if we mix the MedXpertQA data to train the model. We conjecture that the complex reasoning data may improve the model's performance better.

![out_dataset1](assets/model_out_dataset1.png)

**Fig.2 The sample for the self-validation of the model trained on dataset1.**

Similar to the experiments on dataset1, the length of the competition is larger, the reward is higher, and a decrease in the length may result in a lower reward.

![log_dataset1](assets/log_dataset1.png)

**Fig.3 The training log for experiments on dataset2.**

Similar to the experiment on the dataset2, we also found the self-validation step on the model's output, but the format for the model is more better compare the model trained on the dataset1.

![out_dataset2](assets/model_out_dataset2.png)

**Fig.4 The sample for the self-validation of the model trained on dataset2.**

Our experiment on dataset3 was interrupted due to other reasons, but we will resume it as soon as possible if conditions allow. We observed that the model faces significant challenges in training on this dataset. Notably, the model's outputs tend to be excessively long, often exceeding the maximum output length of 8192 tokens that we configured. We believe this difficulty arises because the model struggles with the reasoning required for this task, likely due to a lack of the necessary knowledge to perform effectively in this context. We also find that the model tries to think multiple times to solve the problem, but the length is not enough. We may explore the length of the model's output to check if it helps the model to solve a complex problem.

## Usage

### Training

To prepare the environment, you can follow the steps in [Open-R1](https://github.com/huggingface/open-r1?tab=readme-ov-file#installation). Then, run the following instructions to do the training. Note that the `num_processes` must equal the number of GPUs - 1.


```shell
export TASK="grpo"

ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
            --num_processes 3 \ 
            --main_process_port 6688 \
            --config_file recipes/accelerate_configs/zero3.yaml \
            src/open_r1/$TASK.py --config recipes/HuatuoGPT-o1/grpo/config_medxpert_usmle.yaml
```

## Data prepare

We suggest using one dataset containing easy and hard samples to help the model learn better and generate the correct chain of thought. Curriculum learning and mixing the complex and easy samples in one batch may help the training.  You can use our provide scripts to prepare the data from [MedXpertQA](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA) dataset and [MedQA-USMLE](https://github.com/jind11/MedQA) dataset.

```shell
python scripts/data_prepare.py \ 
		--medxpertqa_root /path/to/medxpertqa \
		--medqa_usmle_root /path/to/medqa_usmle \
		--output_dir ./output/xpert_usmle
```

## TODO

- [x] Release the code.
- [ ] Release the checkpoints.
- [ ] Release the technical report.
- [ ] Release the evaluation results & evaluation codes.
- [ ] Complete the experiments on dataset3.

## Acknowledge

[open-r1](https://github.com/huggingface/open-r1)

[Qwen2.5](https://github.com/QwenLM/Qwen2.5)

[HuatuoGPT-o1](https://github.com/FreedomIntelligence/HuatuoGPT-o1)

## Future Plan

+ Use existing models to construct a QA dataset that differentiates difficulty levels.
+ Explore the ratio of easy to difficult data and how to split the data during training.
+ Investigate enabling the model to learn to use tool calls to retrieve knowledge from a local knowledge base for reasoning. E.g. [Search-R1](https://github.com/PeterGriffinJin/Search-R1)
+ Explore the reinforcement learning algorithm for multi-modality scenes.

