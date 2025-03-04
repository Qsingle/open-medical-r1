"""Reward functions for GRPO training."""

import math
import re
import logging
import os
import datetime

def verify(content, reference, option):
    answer = re.search(r"<answer>(.*)</answer>", content, re.DOTALL)
    reward = 0
    if answer is not None:
        _answer = answer.group(1)
        # print(_answer)
        _answer = _answer.strip()
        refer_box = f"({reference.strip()})"
        if _answer == reference.strip() or _answer == refer_box:
            reward = 2.0 #higher 

        else:
            if option is not None:
                try:
                    pattern = re.compile(re.escape(option), re.IGNORECASE)
                    matched = pattern.match(_answer)
                    if matched is not None:
                        reward = 1.0 #right option is in the answer, we give a reward
                except:
                    reward = 0
            else:
                reward = 0
    else:
        reward = 0 #format
    return reward

def accuracy_reward(completions, reference, option, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, refer, opt in zip(contents, reference, option):
        
        if not os.path.exists(os.path.join("logs", __name__+".log")):
            with open(os.path.join("logs", __name__+".log"), "w") as f:
                f.write(datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S output:"+content+f"\nGT:{refer} Option:{refer}.{opt}\n"))
        else:
            with open(os.path.join("logs", __name__+".log"), "a") as f:
                f.write(datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S output:"+content+f"\nGT:{refer} Option:{refer}.{opt}\n"))
        reward = verify(content, refer, opt)
        rewards.append(reward)
    return rewards

def strict_format(completions, **kwargs):
    """Reward function taht checks the specific format strictly

    Args:
        completions ([type]): [description]
    """
    pattern = r"^<think>.*?</think>[\s\n]?<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>[\s\n]{0,1}<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    soft_format = [1.0 if match else 0.0 for match in matches]
    strict = strict_format(completions, **kwargs)
    rewards = [s1+s2 for s1, s2 in zip(soft_format, strict)] 
    return rewards

def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>") == 1:
        count+=0.25
    if text.count("</think>") == 1:
        count+=0.25
    if text.count("<answer>") == 1:
        count+=0.25
    if text.count("</answer>") == 1:
        count+=0.25
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def reasoning_steps_reward(completions, **kwargs):
    """Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic nubmer 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def length_reward(completions, **kwargs):
    return [1.0 if(len(completion) > 200) else 0 for completion in completions]

def get_repetition_penalty(ngram_size: int, max_penalty: float, generation: str) -> float:
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    if max_penalty == 0:
        return 0

    ngrams = set()
    total = 0
    for ng in zipngram(generation, ngram_size):
        ngrams.add(ng)
        total += 1

    scaling = 1 - len(ngrams) / ( total + 1e-6) #avoid to divided by zero
    return scaling * max_penalty


# Source:
# https://stackoverflow.com/questions/21883108/fast-optimize-n-gram-implementations-in-python
def zipngram(text: str, ngram_size: int):
    words = text.lower().split()
    return zip(*[words[i:] for i in range(ngram_size)])

def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    repetition_max_penalty=-0.5,
    repetition_ngram_size=20,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, reference, option, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of reference answer

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            repetition_max_penalty: Maximum penalty for the repetition
            repetition_max_penalty: The ngram size for repetition
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, refer, opt in zip(contents, reference, option):
            score = verify(content, refer, opt)
            is_correct = False
            if score > 0:
                is_correct = True
            # think = re.search(r"<think>(.*)</think>", content, re.DOTALL)
            # if think is not None:
            #     gen_len = think.group(0)
            # else:
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
                rep_penalty = 0
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong
                rep_penalty = get_repetition_penalty(max_penalty=repetition_max_penalty, ngram_size=repetition_ngram_size, generation=content)

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            reward += rep_penalty
            rewards.append(float(reward))
        return rewards
    return cosine_scaled_reward