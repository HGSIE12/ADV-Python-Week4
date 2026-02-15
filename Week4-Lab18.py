from sklearn.preprocessing import OneHotEncoder
import pandas as pd

df = pd.read_csv("day18_binning.csv")
df18 = pd.DataFrame(df)

#equal-width bins
bins = [0, 18, 35, 50, 100]
labels = ["Child", "YoungAdult", "Adult", "Senior"]
df18["age_bins"] = pd.cut(df18["age"], bins = bins, labels = labels, right = False)

print("-" * 20 ,"values", "-" * 20,"\n")
print(df18["age_bins"].value_counts())

#equal-frequency
df18["age_bins_quantiles"] = pd.qcut(df18["age"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

print("-" * 20 ,"Frequency values", "-" * 20,"\n")
print(df18["age_bins_quantiles"].value_counts())

#domain bins
age_edges = [0, 13, 18, 65, 120]
age_labels = ["Child", "Teen", "Adult", "Senior"]
df18["age_group"] = pd.cut(df18["age"], bins=age_edges, labels=age_labels, right=False)

print("-" * 20 ,"Domain Values", "-" * 20,"\n")
print(df18["age_group"].value_counts())

# One-hot encode bins and compare
ohe = OneHotEncoder( handle_unknown="ignore")
age_encoded = ohe.fit_transform(df18[["age_group"]]).toarray()

print("-" * 20 ,"One hot encoded", "-" * 20,"\n")
print("Encoded shape:", age_encoded.shape, "Categories:", ohe.categories_)

