import random
import pandas as pd

# Unique column for hairless breeds
df = pd.read_excel('xlsx/main.xlsx')
df["isHairless"] = 0
r = df[df["Breed"] == "Sphynx"]
df.loc[df["Breed"] == "Sphynx", "isHairless"] = 1

# Adding new instances for breeds with lack of instances
def addInstances(df, breed, percentage = 500):
    count = round(len(df[df["Breed"] == breed]) * percentage / 100)
    r = random.random()
    for i in range(count):
        df.loc[len(df)] = [max(df["ID"]) + 1, breed, *[round(df[df["Breed"] == breed].iloc[:, k].mean() + r) for k in range(2, df.shape[1] - 1)], 1 if breed == "Sphynx" else 0]
        # print(df.loc[len(df) - 1])
        r = (r * 2) % 1

breed_counts = df["Breed"].value_counts()
for i in range(len(breed_counts)):
    if breed_counts.iloc[i] < 250:
        addInstances(df, breed_counts.index[i])

# Adding previously removed instances of undefined breeds
idf = pd.read_excel('xlsx/initial_data.xlsx')
not_specified = idf[(idf["Race"] == "NR") | (idf["Race"] == "NSP")]
for index, i in not_specified.iterrows():
    df.loc[len(df)] = [max(df["ID"]) + 1, "Not Specified", i.iloc[8], i.iloc[9], i.iloc[11], i.iloc[12], i.iloc[13], i.iloc[14], i.iloc[15], i.iloc[16], i.iloc[17], i.iloc[18], i.iloc[19], i.iloc[20], i.iloc[21], i.iloc[22], i.iloc[23], i.iloc[24], i.iloc[26], i.iloc[27], 0]
    # print(df.loc[len(df) - 1])

oriental_breed = idf[idf["Race"] == "ORI"]
for index, i in oriental_breed.iterrows():
    df.loc[len(df)] = [max(df["ID"]) + 1, "Oriental", i.iloc[8], i.iloc[9], i.iloc[11], i.iloc[12], i.iloc[13], i.iloc[14], i.iloc[15], i.iloc[16], i.iloc[17], i.iloc[18], i.iloc[19], i.iloc[20], i.iloc[21], i.iloc[22], i.iloc[23], i.iloc[24], i.iloc[26], i.iloc[27], 0]

# Save
df.to_excel('xlsx/main.xlsx', index=False)