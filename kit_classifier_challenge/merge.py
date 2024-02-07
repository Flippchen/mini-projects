import pandas as pd

classes = ['Bicycle', 'Bridge', 'Bus', 'Car', 'Chimney', 'Crosswalk', 'Hydrant', 'Motorcycle', 'Other', 'Palm', 'Stair', 'Traffic Light']

base_path = '/home/luke'

csv1 = f'{base_path}/predictions_1'
csv2 = f'{base_path}/predictions_2'
csv3 = f'{base_path}/predictions_3'

df1 = pd.read_csv(csv1)
df2 = pd.read_csv(csv2)
df3 = pd.read_csv(csv3)

# Add the DataFrames and divide by number of DataFrames
df2[classes] = (df2[classes] + df3[classes] + df1[classes]) / 3

# Export the DataFrame to a CSV file
df2.to_csv(f"{base_path}/predictions_merged.csv", index=False)