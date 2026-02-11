import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv("day16_encoding.csv")
df16 = pd.DataFrame(data)

le_city = LabelEncoder()
df16["city_label"] = le_city.fit_transform(df16["city"])
print("Classes:", le_city.classes_)

ohe = OneHotEncoder(handle_unknown="ignore")
city_encoded = ohe.fit_transform(df16[["city"]])
print("\nEncoded shape:", city_encoded.shape, "Categories:", ohe.categories_)
