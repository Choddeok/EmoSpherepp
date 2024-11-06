import pandas as pd

file_path = 'VAD.txt'

data = pd.read_csv(file_path, sep=",", header=None)
data.columns = ['emotion_id', 'A', 'V', 'D']

neutral_data = data[data['emotion_id'] == 2]

average_coordinates = neutral_data[['A', 'V', 'D']].mean()

print("Average Coordinates for Neutral (Label 2):")
print(average_coordinates)
