import re
import json
from vllm import LLM, SamplingParams


if __name__ == "__main__":
    model_name = 'guide-r2'
    model_path = f"/home/z/zhengzhi/Code/LLM-communication-hallucination/Claim-reason-nei/reasoner-{model_name}"
    vllm_model = LLM(model=model_path, gpu_memory_utilization=0.90, max_model_len=4000, max_num_seqs=64)
    sampling_params = SamplingParams(temperature=0., top_p=1., max_tokens=4000)
    for file_name in ['FEVEROUS','HOVER','Open_FEVEROUS','Open_HOVER','LLM-AggreFact_test']:
        file_path = f'testset/{file_name}.json'

        data = []
        # Open the file and read line by line
        with open(file_path, 'r', encoding='utf-8') as file:
            raw_data = json.load(file)
        for item in raw_data:
            data.append(item)
        prompt_list = []
        for now in range(len(data)):
            # Now `data` contains all the JSON objects from the file
            context = data[now]['evidence']
            prompt_judge1 = f'''<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Task: Validate the following claim using the provided context.
Your goal is to determine whether the claim can be supported by the context. Choose between "support" or "refute".

Instructions:
1. Analyze the claim step by step, verifying each crucial component in the claim as they appear.
2. Structure your reasoning on crucial components in the claim in detailed steps, from 1 to a maximum of 10. Make sure each step is the smallest possible logical unit necessary for validation.
3. Ensure that your reasoning correlates consistently with your conclusion. Use "##" to format each step clearly, e.g., "## Reasoning Step 1".
4. Finally, conclude with either "support" or "refute" enclosed in a pair of curly braces, noting the overall judgment regarding the claim.

Context: {context}

Claim: {data[now]['claim']}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

'''
            prompt_list.append(prompt_judge1)
        outputs = vllm_model.generate(prompt_list, sampling_params)
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            match = re.findall(r'\{([^{}]*)\}', generated_text)
            with open(f'{file_name}-llama3-8b-nei-guide-r3.txt', 'a') as file:
                # Append new content to the file
                file.write(f'{match}\n')
    for file_name in ['Healthver_test','Open_Healthver_test', 'Scifact_dev','Scifact_train', 'VitaminC_dev','VitaminC_test']:
        file_path = f'testset/{file_name}.json'

        data = []
        # Open the file and read line by line
        with open(file_path, 'r', encoding='utf-8') as file:
            raw_data = json.load(file)
        for item in raw_data:
            data.append(item)
        prompt_list = []
        for now in range(len(data)):
            # Now `data` contains all the JSON objects from the file
            context = data[now]['evidence']
            prompt_judge1 = f'''<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Task: Validate the following claim using the provided context. 
Your goal is to determine whether the claim can be supported with the context. Choose between "support", "refute", or "not enough information".

Instructions:
1. Analyze the claim step by step, verifying each crucial component in the claim as they appear.
2. Structure your reasoning on crucial components in the claim in detailed steps, from 1 to a maximum of 10. Make sure each step is the smallest possible logical unit necessary for validation.
3. Ensure that your reasoning correlates consistently with your conclusion. Use "##" to format each step clearly, e.g., "## Reasoning Step 1".
4. Finally, conclude with "support", "refute", or "not enough information" enclosed in a pair of curly braces, noting the overall judgment regarding the claim.

Context: {context}

Claim: {data[now]['claim']}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

'''
            prompt_list.append(prompt_judge1)
        outputs = vllm_model.generate(prompt_list, sampling_params)
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            match = re.findall(r'\{([^{}]*)\}', generated_text)
            with open(f'{file_name}-llama3-8b-nei-guide-r3.txt', 'a') as file:
                # Append new content to the file
                file.write(f'{match}\n')

    for file_name in ['Healthver_test','Open_Healthver_test', 'Scifact_dev','Scifact_train', 'VitaminC_dev','VitaminC_test']:
        file_path = f'testset/{file_name}.json'

        data = []
        # Open the file and read line by line
        with open(file_path, 'r', encoding='utf-8') as file:
            raw_data = json.load(file)
        for item in raw_data:
            data.append(item)
        prompt_list = []
        for now in range(len(data)):
            # Now `data` contains all the JSON objects from the file
            context = data[now]['evidence']
            prompt_judge1 = f'''<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Task: Validate the following claim using the provided context.
Your goal is to determine whether the claim can be supported by the context. Choose between "support" or "refute".

Instructions:
1. Analyze the claim step by step, verifying each crucial component in the claim as they appear.
2. Structure your reasoning on crucial components in the claim in detailed steps, from 1 to a maximum of 10. Make sure each step is the smallest possible logical unit necessary for validation.
3. Ensure that your reasoning correlates consistently with your conclusion. Use "##" to format each step clearly, e.g., "## Reasoning Step 1".
4. Finally, conclude with either "support" or "refute" enclosed in a pair of curly braces, noting the overall judgment regarding the claim.

Context: {context}

Claim: {data[now]['claim']}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

'''
            prompt_list.append(prompt_judge1)
        outputs = vllm_model.generate(prompt_list, sampling_params)
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            match = re.findall(r'\{([^{}]*)\}', generated_text)
            with open(f'{file_name}-llama3-8b-wo-nei-guide-r2.txt', 'a') as file:
                # Append new content to the file
                file.write(f'{match}\n')