import pandas as pd

df = pd.read_csv("train_data.csv")
df = df[["text", "label"]]
df.to_csv("train_data.csv", index=False)
