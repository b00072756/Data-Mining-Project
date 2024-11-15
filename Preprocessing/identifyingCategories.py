import pandas as pd

df = pd.read_csv('Dataset/Final_Upwork_Dataset.csv')

with open("unique_values.txt", "w") as file:
    for i in range(1, 10):
        column_name = f"Category_{i}"
        unique_values = df[column_name].dropna().unique().tolist()  # Convert to list to avoid truncation
        file.write(f"Unique values in {column_name}:\n")
        file.write(f"{', '.join(map(str, unique_values))}\n\n")

print("Unique values have been saved to 'unique_values.txt'")
