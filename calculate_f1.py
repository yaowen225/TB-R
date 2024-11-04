import json
from sklearn.metrics import f1_score

# Load the data from the provided files
with open('C:/Users/USER/Desktop/code/Tbrain/競賽訓練資料集/競賽資料集/Output/pred_retrieve.json', 'r', encoding='utf-8') as pred_file, \
     open('C:/Users/USER/Desktop/code/Tbrain/競賽訓練資料集/競賽資料集/dataset/preliminary/ground_truths_example.json', 'r', encoding='utf-8') as truth_file:
    
    pred_data = json.load(pred_file)
    truth_data = json.load(truth_file)

# Extract predictions and ground truths
y_pred = [entry['retrieve'] for entry in pred_data['answers']]
y_true = [entry['retrieve'] for entry in truth_data['ground_truths']]

# Calculate F1 Score
f1 = f1_score(y_true, y_pred, average='macro')

print("F1 Score:", f1)
