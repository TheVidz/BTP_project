import pandas as pd
import matplotlib.pyplot as plt

# Dictionary mapping experiment IDs to their strategy names for clarity
strategies = {
    '3000' : 'no attackers',
    # '3001' : '20 attackers no defense',
    # '3002' : '20 attackers strategy:mean',
    # '3003' : '20 attackers strategy:median',
    # '3004' : '10 attackers strategy: krum',
    '3005' : '10 attackers strategy: trimmed mean'
}

# --- Data Extraction ---
all_accuracy_data = {}
all_loss_data = {}

for exp_id in strategies.keys():
    try:
        file_path = f'{exp_id}_results.csv'
        with open(file_path, 'r') as f:
            lines = f.readlines()

        accuracy_list = []
        loss_list = []

        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                accuracy_list.append(float(parts[0]))
                loss_list.append(float(parts[1]))
        
        all_accuracy_data[exp_id] = accuracy_list
        all_loss_data[exp_id] = loss_list

    except FileNotFoundError:
        print(f"File {file_path} not found.")

# --- Plotting ---
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(14, 7))

# Plot Accuracy
plt.subplot(1, 2, 1)
for exp_id, accuracy_list in all_accuracy_data.items():
    plt.plot(range(len(accuracy_list)), accuracy_list, label=strategies[exp_id])
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
for exp_id, loss_list in all_loss_data.items():
    plt.plot(range(len(loss_list)), loss_list, label=strategies[exp_id])
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()