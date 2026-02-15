import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#data
data = pd.read_csv("day19_transform.csv")
df19 = pd.DataFrame(data)
print(df19)

#=================log1p===================

df19["spend_log1p"] = np.log1p(df19["spend"])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(df19["spend"], ax=axes[0]); axes[0].set_title("Spend raw")
sns.histplot(df19["spend_log1p"], ax=axes[1]); axes[1].set_title("Spend log1p")
plt.show()

print("-"*30, "spend with log1p", "-"*30)
print(df19)

#=================Sqrt===================

df19["spend_sqrt"] = np.sqrt(df19["spend"])

figr, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(df19["spend"], ax=axes[0], discrete=True); axes[0].set_title("spend raw")
sns.histplot(df19["spend_sqrt"], ax=axes[1]); axes[1].set_title("spend_sqrt")
plt.show()

print("-"*30, "spend with sqrt", "-"*30)
print(df19)
