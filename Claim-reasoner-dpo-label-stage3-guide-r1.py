import re
import json
from random import random

from vllm import LLM, SamplingParams

Instruction = '''Task: Validate the following claim using the provided context.
Your goal is to determine whether the claim can be supported by the context. Choose between "support" or "refute".

Instructions:
1. Analyze the claim step by step, verifying each crucial component in the claim as they appear.
2. Structure your reasoning on crucial components in the claim in detailed steps, from 1 to a maximum of 10. Make sure each step is the smallest possible logical unit necessary for validation.
3. Ensure that your reasoning correlates consistently with your conclusion. Use "##" to format each step clearly, e.g., "## Reasoning Step 1".
4. Finally, conclude with either "support" or "refute" enclosed in a pair of curly braces, noting the overall judgment regarding the claim.
'''
if __name__ == "__main__":
    model_path = f"********************/reasoner-sft"
    vllm_model = LLM(model=model_path, gpu_memory_utilization=0.90, max_model_len=4000, max_num_seqs=64)
    sampling_params = SamplingParams(temperature=0.75, top_p=0.95, max_tokens=4000)
    file_path1 = 'trainingset/Feverous_train.json'
    file_path2 = 'trainingset/Hover_train.json'

    data1 = []
    data2 = []
    # Open the file and read line by line
    with open(file_path1, 'r', encoding='utf-8') as file:
        raw_data1 = json.load(file)
    for item in raw_data1:
        data1.append(item)
    with open(file_path2, 'r', encoding='utf-8') as file:
        raw_data2 = json.load(file)
    for item in raw_data2:
        data2.append(item)
    data = data1 + data2
    prompt_list_run1 = []
    prompt_list_run2 = []
    for now in range(len(data)):
        # Now `data` contains all the JSON objects from the file
        prompt_judge1 = f'''<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Task: Validate the following claim using the provided context.
Your goal is to determine whether the claim can be supported by the context. Choose between "support" or "refute".

Instructions:
1. Analyze the claim step by step, verifying each crucial component in the claim as they appear.
2. Structure your reasoning on crucial components in the claim in detailed steps, from 1 to a maximum of 10. Make sure each step is the smallest possible logical unit necessary for validation.
3. Ensure that your reasoning correlates consistently with your conclusion. Use "##" to format each step clearly, e.g., "## Reasoning Step 1".
4. Finally, conclude with either "support" or "refute" enclosed in a pair of curly braces, noting the overall judgment regarding the claim.

Context: {data[now]['evidence']}

Claim: {data[now]['claim']}


The ground truth is

---

Answer: support. You mush generate results that match the ground truth.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

'''
        prompt_judge2 = f'''<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Task: Validate the following claim using the provided context.
Your goal is to determine whether the claim can be supported by the context. Choose between "support" or "refute".

Instructions:
1. Analyze the claim step by step, verifying each crucial component in the claim as they appear.
2. Structure your reasoning on crucial components in the claim in detailed steps, from 1 to a maximum of 10. Make sure each step is the smallest possible logical unit necessary for validation.
3. Ensure that your reasoning correlates consistently with your conclusion. Use "##" to format each step clearly, e.g., "## Reasoning Step 1".
4. Finally, conclude with either "support" or "refute" enclosed in a pair of curly braces, noting the overall judgment regarding the claim.

Context: {data[now]['evidence']}

Claim: {data[now]['claim']}


The ground truth is

---

Answer: refute. You mush generate results that match the ground truth.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

'''
        prompt_list_run2.append(prompt_judge2)
        prompt_list_run1.append(prompt_judge1)
    outputs1 = vllm_model.generate(prompt_list_run1, sampling_params)
    outputs2 = vllm_model.generate(prompt_list_run2, sampling_params)
    training_dataset = []
    for i in range(len(outputs1)):
        label = data[i]['label']
        if label == 'Refutes' or label == 'refutes' or label == 'CONTRADICT':
            label_unified = "refute"
        elif label == 'UNKNOWN' or label == 'Neutral':
            label_unified = "refute"
        elif label == 'SUPPORT' or label == 'Supports' or label == 'supports':
            label_unified = "support"
        user_prompt = f'''Context: {data[i]['evidence']}

Claim: {data[i]['claim']}
'''
        generated_text1 = outputs1[i].outputs[0].text
        generated_text2 = outputs2[i].outputs[0].text
        match1 = re.findall(r'\{([^{}]*)\}', generated_text1)
        match2 = re.findall(r'\{([^{}]*)\}', generated_text2)

        if len(generated_text2) > 2000 or len(generated_text1) > 2000 or len(user_prompt)>3000:
            continue
        if match1 == [] and match2 == []:
            continue

        if match1 == [] and match2 != []:
            predict2 = re.findall(r'\{([^{}]*)\}', generated_text2)[-1]
            if predict2.strip() == label_unified.strip():
                save_dict = {"instruction": Instruction, "input": user_prompt, "chosen": generated_text2,
                             "rejected": generated_text1}
                training_dataset.append(save_dict)
                continue
            else:
                continue

        if match1 != [] and match2 == []:
            predict1 = re.findall(r'\{([^{}]*)\}', generated_text1)[-1]
            if predict1.strip() == label_unified.strip():
                save_dict = {"instruction": Instruction, "input": user_prompt, "chosen": generated_text1,
                             "rejected": generated_text2}
                training_dataset.append(save_dict)
                continue
            else:
                continue

        predict1 = re.findall(r'\{([^{}]*)\}', generated_text1)[-1]
        predict2 = re.findall(r'\{([^{}]*)\}', generated_text2)[-1]
        if predict1.strip() == predict2.strip():
            continue
        if predict1.strip() == label_unified.strip():
            save_dict = {"instruction": Instruction, "input": user_prompt, "chosen": generated_text1,
                         "rejected": generated_text2}
            training_dataset.append(save_dict)
        elif predict2.strip() == label_unified.strip():
            save_dict = {"instruction": Instruction, "input": user_prompt, "chosen": generated_text2,
                         "rejected": generated_text1}
            training_dataset.append(save_dict)

    file_path = 'trainingset/Healthver_train.json'

    data = []
    # Open the file and read line by line
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_data = json.load(file)
    for item in raw_data:
        data.append(item)
    prompt_list_run1 = []
    prompt_list_run2 = []
    prompt_list_run3 = []
    for now in range(len(data)):
        prompt_judge_support = f'''<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Task: Validate the following claim using the provided context. 
Your goal is to determine whether the claim can be supported with the context. Choose between "support", "refute", or "not enough information".

Instructions:
1. Analyze the claim step by step, verifying each crucial component in the claim as they appear.
2. Structure your reasoning on crucial components in the claim in detailed steps, from 1 to a maximum of 10. Make sure each step is the smallest possible logical unit necessary for validation.
3. Ensure that your reasoning correlates consistently with your conclusion. Use "##" to format each step clearly, e.g., "## Reasoning Step 1".
4. Finally, conclude with "support", "refute", or "not enough information" enclosed in a pair of curly braces, noting the overall judgment regarding the claim.

Context: {data[now]['evidence']}

Claim: {data[now]['claim']}


The ground truth is

---

Answer: support. You mush generate results that match the ground truth.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

'''
        prompt_judge_refute = f'''<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Task: Validate the following claim using the provided context. 
Your goal is to determine whether the claim can be supported with the context. Choose between "support", "refute", or "not enough information".

Instructions:
1. Analyze the claim step by step, verifying each crucial component in the claim as they appear.
2. Structure your reasoning on crucial components in the claim in detailed steps, from 1 to a maximum of 10. Make sure each step is the smallest possible logical unit necessary for validation.
3. Ensure that your reasoning correlates consistently with your conclusion. Use "##" to format each step clearly, e.g., "## Reasoning Step 1".
4. Finally, conclude with "support", "refute", or "not enough information" enclosed in a pair of curly braces, noting the overall judgment regarding the claim.

Context: {data[now]['evidence']}

Claim: {data[now]['claim']}


The ground truth is

---

Answer: refute. You mush generate results that match the ground truth.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

'''
        prompt_judge_nei = f'''<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Task: Validate the following claim using the provided context. 
Your goal is to determine whether the claim can be supported with the context. Choose between "support", "refute", or "not enough information".

Instructions:
1. Analyze the claim step by step, verifying each crucial component in the claim as they appear.
2. Structure your reasoning on crucial components in the claim in detailed steps, from 1 to a maximum of 10. Make sure each step is the smallest possible logical unit necessary for validation.
3. Ensure that your reasoning correlates consistently with your conclusion. Use "##" to format each step clearly, e.g., "## Reasoning Step 1".
4. Finally, conclude with "support", "refute", or "not enough information" enclosed in a pair of curly braces, noting the overall judgment regarding the claim.

Context: {data[now]['evidence']}

Claim: {data[now]['claim']}


The ground truth is

---

Answer: not enough information. You mush generate results that match the ground truth.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

'''
        prompt_list_run3.append(prompt_judge_nei)
        prompt_list_run2.append(prompt_judge_refute)
        prompt_list_run1.append(prompt_judge_support)
    outputs1 = vllm_model.generate(prompt_list_run1, sampling_params)
    outputs2 = vllm_model.generate(prompt_list_run2, sampling_params)
    outputs3 = vllm_model.generate(prompt_list_run3, sampling_params)
    training_dataset = []
    for i in range(len(outputs1)):
        label = data[i]['label']
        if label == 'Refutes' or label == 'refutes' or label == 'CONTRADICT':
            label_unified = "refute"
        elif label == 'UNKNOWN' or label == 'Neutral':
            label_unified = "not enough information"
        elif label == 'SUPPORT' or label == 'Supports' or label == 'supports':
            label_unified = "support"
        user_prompt = f'''Context: {data[i]['evidence']}

    Claim: {data[i]['claim']}
    '''
        generated_text1 = outputs1[i].outputs[0].text
        generated_text2 = outputs2[i].outputs[0].text
        generated_text3 = outputs3[i].outputs[0].text
        match1 = re.findall(r'\{([^{}]*)\}', generated_text1)
        match2 = re.findall(r'\{([^{}]*)\}', generated_text2)
        match3 = re.findall(r'\{([^{}]*)\}', generated_text3)

        if len(generated_text2) > 2000 or len(generated_text1) > 2000 or len(generated_text3) > 2000 or len(
                user_prompt) > 3000:
            continue
        if match1 == [] or match2 == [] or match3 == []:
            continue
        predict1 = re.findall(r'\{([^{}]*)\}', generated_text1)[-1]
        predict2 = re.findall(r'\{([^{}]*)\}', generated_text2)[-1]
        predict3 = re.findall(r'\{([^{}]*)\}', generated_text3)[-1]
        if predict1.strip() == 'support' and predict2.strip() == 'refute' and predict3.strip() == 'not enough information':
            if label_unified == 'refute':
                save_dict1 = {"instruction": Instruction, "input": user_prompt, "chosen": generated_text2,
                              "rejected": generated_text3}
                save_dict2 = {"instruction": Instruction, "input": user_prompt, "chosen": generated_text2,
                              "rejected": generated_text1}
                training_dataset.append(save_dict1)
                training_dataset.append(save_dict2)

            elif label_unified == 'support':
                save_dict1 = {"instruction": Instruction, "input": user_prompt, "chosen": generated_text1,
                              "rejected": generated_text2}
                save_dict2 = {"instruction": Instruction, "input": user_prompt, "chosen": generated_text1,
                              "rejected": generated_text3}
                training_dataset.append(save_dict1)
                training_dataset.append(save_dict2)

            elif label_unified == 'not enough information':
                save_dict1 = {"instruction": Instruction, "input": user_prompt, "chosen": generated_text3,
                              "rejected": generated_text2}
                save_dict2 = {"instruction": Instruction, "input": user_prompt, "chosen": generated_text3,
                              "rejected": generated_text1}
                training_dataset.append(save_dict1)
    random.shuffle(training_dataset)
    with open('Training_claim_reason_guide_alpaca_merge_nei_r1.json', 'w', encoding='utf-8') as f:
        json.dump(training_dataset, f, ensure_ascii=False, indent=4)
