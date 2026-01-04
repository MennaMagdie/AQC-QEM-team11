import pandas as pd


df = pd.read_csv("quantum_dataset_updated.csv")
def fix_noise_label(val):
    if val < 0.5: return 0  # Low / QAOA
    if val < 1.5: return 1  # Medium
    return 2                # High

df['noise_level'] = df['noise_level'].apply(fix_noise_label)
df.to_csv("quantum_dataset_long.csv", index=False)
print("Dataset standardized. All noise levels are now 0, 1, or 2.")