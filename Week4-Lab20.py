import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

#===============data===============
data = pd.read_csv("day20_integration.csv")
df20 = pd.DataFrame(data)
print("", "-"*30, "Original Data", "-"*30)
print(df20)

#===============OneHotEncoder===============

ohe = OneHotEncoder(handle_unknown="ignore")
city_encoded = ohe.fit_transform(df20[["city"]])
print("", "-"*30, "OneHotEncoder", "-"*30)
print("Encoded shape:", city_encoded.shape, "Categories:", ohe.categories_)

#===============Min-Max Scaling===============

mm_scaler = MinMaxScaler()
df20[["m_pages_viewed", "m_session_minutes"]] = mm_scaler.fit_transform(df20[["pages_viewed", "session_minutes"]])
print("\n", "-"*30, "MinMax scaler", "-"*30)
print(df20)

#=================log1p===================

df20["basket_value_log1p"] = np.log1p(df20["basket_value"])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(df20["basket_value"], ax=axes[0]); axes[0].set_title("basket_value raw")
sns.histplot(df20["basket_value_log1p"], ax=axes[1]); axes[1].set_title("basket_value log1p")
plt.show()

print("-"*30, "basket_value with log1p", "-"*30)
print(df20)

#=================Sqrt===================

df20["basket_value_sqrt"] = np.sqrt(df20["basket_value"])

figr, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(df20["basket_value"], ax=axes[0], discrete=True); axes[0].set_title("basket_value raw")
sns.histplot(df20["basket_value_sqrt"], ax=axes[1]); axes[1].set_title("basket_value_sqrt")
plt.show()

print("-"*30, "basket_value with sqrt", "-"*30)
print(df20)
