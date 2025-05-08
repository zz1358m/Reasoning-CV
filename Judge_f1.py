import pandas as pd
import json

from numpy.ma.extras import average
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
name = 'llama3-3b-nei-guide-r2'
file_path = './testset/FEVEROUS.json'
save = []

def process_line(line, label):
    line = line.strip()  # Remove any leading/trailing whitespace
    if line == "['support']" or line == "['Support']":
        return 1
    elif line == "['refute']" or line == "['Refute']":
        return 0
    else:
        # Invert the label: if label is 1, return 0, and vice versa
        print(line)
        return 0

def process_line_llm(line, label):
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


# Example usage
prediction_path = f'results_llama/FEVEROUS-{name}.txt'

label = []
# Open the file and read line by line
with open(file_path, 'r', encoding='utf-8') as file:
    raw_data = json.load(file)
for item in raw_data:
    label.append(int(item['label'] == 'supports'))
    
prediction = process_file(prediction_path, label)


accuracy = sum([int(label[i] == prediction[i]) for i in range(len(label))])/len(label)
print(f'Accuracy: {accuracy: .4f}')


macro_f1 = f1_score(label, prediction, average='macro')
print(f'Macro F1 Score: {macro_f1: .4f}')


tn, fp, fn, tp = confusion_matrix(label, prediction).ravel()


type1_error = fp / (fp + tn) if (fp + tn) > 0 else 0
print(f'Type 1 Error (False Positive Rate): {type1_error: .2f}')


type2_error = fn / (fn + tp) if (fn + tp) > 0 else 0
print(f'Type 2 Error (False Negative Rate): {type2_error: .2f}')

print(f'{accuracy*100: .2f} {macro_f1*100: .2f} {type1_error*100: .2f} {type2_error*100: .2f}')
save.append(f'{macro_f1*100: .2f}')
import pandas as pd
import json

from numpy.ma.extras import average
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix



for hop in [2,3,4]:
    file_path = './testset/HOVER.json'
    # Example usage
    prediction_path = f'results_llama/HOVER-{name}.txt'

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

    
    accuracy = sum([int(label[i] == prediction[i]) for i in range(len(label))])/len(label)
    print(f'Accuracy: {accuracy: .4f}')


    macro_f1 = f1_score(label, prediction, average='macro')
    print(f'Macro F1 Score: {macro_f1: .4f}')

    
    tn, fp, fn, tp = confusion_matrix(label, prediction).ravel()

    
    type1_error = fp / (fp + tn) if (fp + tn) > 0 else 0
    print(f'Type 1 Error (False Positive Rate): {type1_error: .2f}')

    
    type2_error = fn / (fn + tp) if (fn + tp) > 0 else 0
    print(f'Type 2 Error (False Negative Rate): {type2_error: .2f}')
    save.append(f'{macro_f1*100: .2f}')

    print(f'{accuracy*100: .2f} {macro_f1*100: .2f} {type1_error*100: .2f} {type2_error*100: .2f}')




# Example usage
file_path = './testset/LLM-AggreFact_test.json'
prediction_path = f'results_llama/LLM-AggreFact_test-{name}.txt'

label = []
# Open the file and read line by line
with open(file_path, 'r', encoding='utf-8') as file:
    raw_data = json.load(file)
for item in raw_data:
    label.append(int(item['label'] == 1))


prediction = process_file_llm(prediction_path, label)


accuracy = sum([int(label[i] == prediction[i]) for i in range(len(label))])/len(label)
print(f'Accuracy: {accuracy: .4f}')


macro_f1 = f1_score(label, prediction, average='macro')
print(f'Macro F1 Score: {macro_f1: .4f}')


tn, fp, fn, tp = confusion_matrix(label, prediction).ravel()


type1_error = fp / (fp + tn) if (fp + tn) > 0 else 0
print(f'Type 1 Error (False Positive Rate): {type1_error: .2f}')


type2_error = fn / (fn + tp) if (fn + tp) > 0 else 0
print(f'Type 2 Error (False Negative Rate): {type2_error: .2f}')

print(f'{accuracy*100: .2f} {macro_f1*100: .2f} {type1_error*100: .2f} {type2_error*100: .2f}')
save.append(f'{macro_f1*100: .2f}')


for i in range(len(save)):
    print(f'{save[i]}', end=' ')
