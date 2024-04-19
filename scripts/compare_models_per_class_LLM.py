import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Load the classification results from the CSV files
df_Llama2 = pd.read_csv("../Llama2/classification_results.csv")
df_Mistral = pd.read_csv("../Mistral/classification_results.csv")

# Combine the two dataframes
df_combined = pd.concat([df_Llama2, df_Mistral], axis=1)
df_combined.columns = ['Response_Llama2', 'Gold_Standard_Llama2', 'Response_Mistral', 'Gold_Standard_Mistral']

# Get the unique classes from the Gold_Standard column
classes = np.unique(df_combined['Gold_Standard_Llama2'])

# Initialize lists to store accuracy and F1-score per class for both models
accuracy_Llama2 = []
accuracy_Mistral = []
f1_Llama2 = []
f1_Mistral = []

# Calculate accuracy and F1-score per class for both models
for cls in classes:
    # Filter dataframe for each class
    df_cls_Llama2 = df_combined[df_combined['Gold_Standard_Llama2'] == cls]
    df_cls_Mistral = df_combined[df_combined['Gold_Standard_Mistral'] == cls]

    # Calculate accuracy
    accuracy_Llama2.append(accuracy_score(df_cls_Llama2['Gold_Standard_Llama2'], df_cls_Llama2['Response_Llama2']))
    accuracy_Mistral.append(accuracy_score(df_cls_Mistral['Gold_Standard_Mistral'], df_cls_Mistral['Response_Mistral']))
    
    # Calculate F1-score
    f1_Llama2.append(f1_score(df_cls_Llama2['Gold_Standard_Llama2'], df_cls_Llama2['Response_Llama2'], average='macro'))
    f1_Mistral.append(f1_score(df_cls_Mistral['Gold_Standard_Mistral'], df_cls_Mistral['Response_Mistral'], average='macro'))

# Plot accuracy per class for both models
plt.figure(figsize=(10, 6))
plt.plot(classes, accuracy_Llama2, label='Llama2', marker='o')
plt.plot(classes, accuracy_Mistral, label='Mistral', marker='o')
plt.title('Accuracy per Class')
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.xticks(classes)
plt.legend()
plt.grid(True)
plt.savefig('./compare_accuracy_LLM.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot F1-score per class for both models
plt.figure(figsize=(10, 6))
plt.plot(classes, f1_Llama2, label='Llama2', marker='o')
plt.plot(classes, f1_Mistral, label='Mistral', marker='o')
plt.title('F1-score per Class')
plt.xlabel('Class')
plt.ylabel('F1-score')
plt.xticks(classes)
plt.legend()
plt.grid(True)
plt.savefig('./compare_f1_LLM.png', dpi=300, bbox_inches='tight')
plt.show()
