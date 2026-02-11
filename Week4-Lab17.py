import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

df = pd.read_csv("day17_scaling.csv")
df17 = pd.DataFrame(df)

mm_scaler = MinMaxScaler()
df17_mm = df17.copy()
df17_mm[["CRIM_mm", "RM_mm"]] = mm_scaler.fit_transform(df17[["CRIM", "RM"]])

std_scaler = StandardScaler()
df17_std = df17.copy()
df17_std[["CRIM_std", "RM_std"]] = std_scaler.fit_transform(df17[["CRIM", "RM"]])

rob_scaler = RobustScaler()
df17_rob = df17.copy()
df17_rob[["CRIM_rob", "RM_rob"]] = rob_scaler.fit_transform(df17[["CRIM", "RM"]])


print("---MinMax Scaler:----")
print(df17_mm[["CRIM_mm", "RM_mm"]].describe())
print("\n" + "="*50 + "\n")

print("---Standard Scaler:---")
print(df17_std[["CRIM_std", "RM_std"]].agg(["mean", "std", "min", "max"]))
print("\n" + "="*50 + "\n")

print("---Robust Scaler:---")
print(df17_rob[["CRIM_rob", "RM_rob"]].describe())
