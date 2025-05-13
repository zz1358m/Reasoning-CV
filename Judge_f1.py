import pandas as pd
import json

from numpy.ma.extras import average
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

model_name = 'llama3-8b-nei-sft' # LLM_SFT
# model_name = 'llama3-8b-nei-guide-r1-final' # LLM_SI1
# model_name = 'llama3-8b-nei-guide-r2-final' # LLM_SI2


# Run for report results.

file_path = './testset/FEVEROUS.json'
save = []
dataset_name = []
def process_line(line, label):
    line = line.strip()  # Remove any leading/trailing whitespace
    if 'support' in line or 'Support' in line:
        return 1
    else:
        return 0


def process_file(file_path, labels):
    results = []
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if i >= len(labels):
                raise IndexError(f"Not enough labels for line {i + 1}")
            processed_value = process_line(line, labels[i])
            results.append(processed_value)
    return results


prediction_path = f'results_llama/FEVEROUS-{model_name}.txt'

label = []
with open(file_path, 'r', encoding='utf-8') as file:
    raw_data = json.load(file)
for item in raw_data:
    label.append(int(item['label'] == 'supports'))
prediction = process_file(prediction_path, label)

# 计算准确率
accuracy = sum([int(label[i] == prediction[i]) for i in range(len(label))]) / len(label)
print(f'Accuracy: {accuracy: .4f}')

macro_f1 = f1_score(label, prediction, average='macro')
print(f'Macro F1 Score: {macro_f1: .4f}')

# 计算混淆矩阵
tn, fp, fn, tp = confusion_matrix(label, prediction).ravel()

type1_error = fp / (fp + tn) if (fp + tn) > 0 else 0
print(f'Type 1 Error (False Positive Rate): {type1_error: .2f}')

type2_error = fn / (fn + tp) if (fn + tp) > 0 else 0
print(f'Type 2 Error (False Negative Rate): {type2_error: .2f}')

print(f'{accuracy * 100: .2f} {macro_f1 * 100: .2f} {type1_error * 100: .2f} {type2_error * 100: .2f}')
save.append(f'{macro_f1 * 100: .2f}')
dataset_name.append('FEVEROUS')

for hop in [2, 3, 4]:
    file_path = './testset/HOVER.json'
    prediction_path = f'results_llama/HOVER-{model_name}.txt'

    label0 = []
    # Open the file and read line by line
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_data = json.load(file)
    for item in raw_data:
        label0.append(int(item['label'] == 'supports'))
    prediction0 = process_file(prediction_path, label0)
    label = []
    prediction = []
    for i in range(len(prediction0)):
        if raw_data[i]['num_hops'] == hop:
            label.append(label0[i])
            prediction.append(prediction0[i])

    accuracy = sum([int(label[i] == prediction[i]) for i in range(len(label))]) / len(label)
    print(f'Accuracy: {accuracy: .4f}')

    macro_f1 = f1_score(label, prediction, average='macro')
    print(f'Macro F1 Score: {macro_f1: .4f}')

    tn, fp, fn, tp = confusion_matrix(label, prediction).ravel()

    type1_error = fp / (fp + tn) if (fp + tn) > 0 else 0
    print(f'Type 1 Error (False Positive Rate): {type1_error: .2f}')

    type2_error = fn / (fn + tp) if (fn + tp) > 0 else 0
    print(f'Type 2 Error (False Negative Rate): {type2_error: .2f}')
    save.append(f'{macro_f1 * 100: .2f}')
    dataset_name.append(f'HOVER-{hop}hop')

    print(f'{accuracy * 100: .2f} {macro_f1 * 100: .2f} {type1_error * 100: .2f} {type2_error * 100: .2f}')

file_path = './testset/FEVEROUS.json'
prediction_path = f'results_llama/Open_FEVEROUS-{model_name}.txt'

label = []
with open(file_path, 'r', encoding='utf-8') as file:
    raw_data = json.load(file)
for item in raw_data:
    label.append(int(item['label'] == 'supports'))
prediction = process_file(prediction_path, label)

# 计算准确率
accuracy = sum([int(label[i] == prediction[i]) for i in range(len(label))]) / len(label)
print(f'Accuracy: {accuracy: .4f}')

macro_f1 = f1_score(label, prediction, average='macro')
print(f'Macro F1 Score: {macro_f1: .4f}')

tn, fp, fn, tp = confusion_matrix(label, prediction).ravel()

type1_error = fp / (fp + tn) if (fp + tn) > 0 else 0
print(f'Type 1 Error (False Positive Rate): {type1_error: .2f}')

type2_error = fn / (fn + tp) if (fn + tp) > 0 else 0
print(f'Type 2 Error (False Negative Rate): {type2_error: .2f}')

print(f'{accuracy * 100: .2f} {macro_f1 * 100: .2f} {type1_error * 100: .2f} {type2_error * 100: .2f}')
save.append(f'{macro_f1 * 100: .2f}')
dataset_name.append('Open_FEVEROUS')

for hop in [2, 3, 4]:
    file_path = './testset/HOVER.json'
    prediction_path = f'results_llama/Open_HOVER-{model_name}.txt'

    label0 = []
    # Open the file and read line by line
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_data = json.load(file)
    for item in raw_data:
        label0.append(int(item['label'] == 'supports'))
    prediction0 = process_file(prediction_path, label0)
    label = []
    prediction = []
    for i in range(len(prediction0)):
        if raw_data[i]['num_hops'] == hop:
            label.append(label0[i])
            prediction.append(prediction0[i])

    accuracy = sum([int(label[i] == prediction[i]) for i in range(len(label))]) / len(label)
    print(f'Accuracy: {accuracy: .4f}')

    macro_f1 = f1_score(label, prediction, average='macro')
    print(f'Macro F1 Score: {macro_f1: .4f}')

    tn, fp, fn, tp = confusion_matrix(label, prediction).ravel()

    type1_error = fp / (fp + tn) if (fp + tn) > 0 else 0
    print(f'Type 1 Error (False Positive Rate): {type1_error: .2f}')

    type2_error = fn / (fn + tp) if (fn + tp) > 0 else 0
    print(f'Type 2 Error (False Negative Rate): {type2_error: .2f}')
    save.append(f'{macro_f1 * 100: .2f}')
    dataset_name.append(f'Open_HOVER-{hop}hop')

    print(f'{accuracy * 100: .2f} {macro_f1 * 100: .2f} {type1_error * 100: .2f} {type2_error * 100: .2f}')

file_path = './testset/LLM-AggreFact_test.json'
prediction_path = f'results_llama/LLM-AggreFact_test-{model_name}.txt'

label = []
with open(file_path, 'r', encoding='utf-8') as file:
    raw_data = json.load(file)
for item in raw_data:
    label.append(int(item['label'] == 1))
prediction = process_file(prediction_path, label)

accuracy = sum([int(label[i] == prediction[i]) for i in range(len(label))]) / len(label)
print(f'Accuracy: {accuracy: .4f}')

macro_f1 = f1_score(label, prediction, average='macro')
print(f'Macro F1 Score: {macro_f1: .4f}')

tn, fp, fn, tp = confusion_matrix(label, prediction).ravel()

type1_error = fp / (fp + tn) if (fp + tn) > 0 else 0
print(f'Type 1 Error (False Positive Rate): {type1_error: .2f}')

type2_error = fn / (fn + tp) if (fn + tp) > 0 else 0
print(f'Type 2 Error (False Negative Rate): {type2_error: .2f}')

print(f'{accuracy * 100: .2f} {macro_f1 * 100: .2f} {type1_error * 100: .2f} {type2_error * 100: .2f}')
save.append(f'{macro_f1 * 100: .2f}')
dataset_name.append('LLM-AggreFact')


def process_line_nei(line, label):
    line = line.strip()  # Remove any leading/trailing whitespace
    if line == "['support']" or line == "['Support']":
        return 2
    elif line == "['refute']" or line == "['Refute']":
        return 0
    else:
        return 1


def process_file_nei(file_path, labels):
    results = []
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if i >= len(labels):
                raise IndexError(f"Not enough labels for line {i + 1}")
            processed_value = process_line_nei(line, labels[i])
            results.append(processed_value)
    return results


# for name in ['Scifact_train', 'Scifact_dev', 'Healthver_test']:
for name in ['Healthver_test', 'Open_Healthver_test', 'Scifact_train', 'Scifact_dev', 'VitaminC_dev', 'VitaminC_test']:

    file_path = f'./testset/{name}.json'
    # Example usage
    prediction_path = f'results_llama/{name}-{model_name}.txt'

    label = []
    # Open the file and read line by line
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_data = json.load(file)
    if name == 'Healthver_test' or name == 'Open_Healthver_test':
        for item in raw_data:
            if item['label'] == 'Supports':
                label.append(int(2))
            elif item['label'] == 'Neutral':
                label.append(int(1))
            else:
                label.append(int(0))
    elif name == 'VitaminC_dev' or name == 'VitaminC_test':
        for item in raw_data:
            if item['label'] == 'SUPPORTS':
                label.append(int(2))
            elif item['label'] == 'NOT ENOUGH INFO':
                label.append(int(1))
            else:
                label.append(int(0))
    else:
        for item in raw_data:
            if item['label'] == 'SUPPORT':
                label.append(int(2))
            elif item['label'] == 'UNKNOWN':
                label.append(int(1))
            else:
                label.append(int(0))

    prediction = process_file_nei(prediction_path, label)

    # 计算准确率
    accuracy = sum([int(label[i] == prediction[i]) for i in range(len(label))]) / len(label)
    print(f'Accuracy: {accuracy: .4f}')

    macro_f1 = f1_score(label, prediction, average='macro')
    print(f'Macro F1 Score: {macro_f1: .4f}')

    print(f'{accuracy * 100: .2f} {macro_f1 * 100: .2f}')
    save.append(f'{macro_f1 * 100: .2f}')
    dataset_name.append(name + ' w NEI')

# for name in ['Scifact_train', 'Scifact_dev', 'Healthver_test']:
for name in ['Healthver_test', 'Open_Healthver_test', 'Scifact_train', 'Scifact_dev', 'VitaminC_dev', 'VitaminC_test']:

    file_path = f'./testset/{name}.json'
    # Example usage
    prediction_path = f'results_llama/{name}-{model_name}.txt'

    label = []
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_data = json.load(file)
    if name == 'Healthver_test' or name == 'Open_Healthver_test':
        for item in raw_data:
            if item['label'] == 'Supports':
                label.append(int(1))
            else:
                label.append(int(0))
    elif name == 'VitaminC_dev' or name == 'VitaminC_test':
        for item in raw_data:
            if item['label'] == 'SUPPORTS':
                label.append(int(1))
            else:
                label.append(int(0))
    else:
        for item in raw_data:
            if item['label'] == 'SUPPORT':
                label.append(int(1))
            else:
                label.append(int(0))
    prediction = process_file(prediction_path, label)

    accuracy = sum([int(label[i] == prediction[i]) for i in range(len(label))]) / len(label)
    print(f'Accuracy: {accuracy: .4f}')

    macro_f1 = f1_score(label, prediction, average='macro')
    print(f'Macro F1 Score: {macro_f1: .4f}')

    print(f'{accuracy * 100: .2f} {macro_f1 * 100: .2f}')
    save.append(f'{macro_f1 * 100: .2f}')
    dataset_name.append(name + ' w/o NEI')
import pandas as pd
from tabulate import tabulate

show_rank = []
show_rank.append([0, 1, 2, 3, 15, 9])
show_rank.append([4, 5, 6, 7, 16, 10])
show_rank.append([8, 17, 11, 18, 12, 19, 13, 20, 14])
dataset_name_show = []
performance_show = []
for rank in show_rank:
    dataset_temp = []
    save_temp = []
    for number in rank:
        dataset_temp.append(dataset_name[number])
        save_temp.append(save[number])
    dataset_name_show.append(dataset_temp)
    performance_show.append(save_temp)
df1 = pd.DataFrame({
    "Dataset Name": dataset_name_show[0],
    "Performance": performance_show[0]
})
df2 = pd.DataFrame({
    "Dataset Name": dataset_name_show[1],
    "Performance": performance_show[1]
})
df3 = pd.DataFrame({
    "Dataset Name": dataset_name_show[2],
    "Performance": performance_show[2]
})

for table in range(3):
    for i in range(len(performance_show[table])):
        print(f'{performance_show[table][i]}', end=' ')
    print('')
table = tabulate(df1.T, headers='keys', tablefmt='fancy_grid')
print(table)
table = tabulate(df2.T, headers='keys', tablefmt='fancy_grid')
print(table)
table = tabulate(df3.T, headers='keys', tablefmt='fancy_grid')
print(table)
