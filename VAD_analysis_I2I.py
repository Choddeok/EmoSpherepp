import numpy as np
import pandas as pd
from scipy.optimize import minimize

file_path = 'VAD.txt'

data = pd.read_csv(file_path, sep=",", header=None)
data.columns = ['emotion_id', 'A', 'V', 'D']

emotions = {0: 'Angry', 1: 'Happy', 3: 'Sad', 4: 'Surprise'}
neutral_data = data[data['emotion_id'] == 2][['A', 'V', 'D']].values

def mean_distance(M, E):
    return np.mean(np.linalg.norm(E - M, axis=1))

def ratio_function(M, Ek, En):
    M = np.array(M)
    mean_dist_within = mean_distance(M, Ek)
    mean_dist_between = mean_distance(M, En)
    return -mean_dist_within / mean_dist_between  # maximize ratio by minimizing the negative ratio

optimal_coords = {}

for emotion_id, emotion_name in emotions.items():
    emotion_data = data[data['emotion_id'] == emotion_id][['A', 'V', 'D']].values
    initial_guess = np.mean(emotion_data, axis=0)
    
    result = minimize(ratio_function, initial_guess, args=(emotion_data, neutral_data), method='L-BFGS-B', bounds=[(0, 1), (0, 1), (0, 1)])
    optimal_coords[emotion_name] = result.x

for emotion, coord in optimal_coords.items():
    print(f"Optimal coordinates for {emotion}: {coord}")
