import torch
import torch.nn as nn
import sys
import nltk
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
nltk.download('punkt')
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
sys.path.insert(0, '..')
import pickle
from torch.autograd import grad as torch_grad


print(torch.__version__)
print("GPU Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = "cpu"

def get_selected_features(SearchFeatures, include_segments=2):
    all_selected_eeg = []
    all_selected_ids = []
    for index in range(len(SearchFeatures[1])):
        sentence_eeg = SearchFeatures[0][index]
        sentence_ids = SearchFeatures[1][index]

        selected_ids = sentence_ids[-include_segments:]
        selected_eeg = sentence_eeg[-include_segments:]

        all_selected_eeg.append(selected_eeg)
        all_selected_ids.append(selected_ids)

    return all_selected_eeg

def get_search_x_y(SearchFeatures, label):
    X_data = []
    Y_Data = []

    for index in range(len(SearchFeatures)):
        sentence_eeg = SearchFeatures[index]
        for eeg in sentence_eeg:
            #print(eeg.shape)
            eeg = eeg.reshape(68, 9)
            X_data.append(eeg)
            Y_Data.append(label)
    return X_data, Y_Data

def combine_data(NeedToSearch_X, CorrectSearch_X, IncorrectSearch_X, NeedToSearch_Y, CorrectSearch_Y, IncorrectSearch_Y):
    X_data = NeedToSearch_X + CorrectSearch_X[:len(NeedToSearch_X)] + IncorrectSearch_X[:len(NeedToSearch_X)]
    Y_data = NeedToSearch_Y + CorrectSearch_Y[:len(NeedToSearch_X)] + IncorrectSearch_Y[:len(NeedToSearch_X)]
    return X_data, Y_data

def get_all_subject_x_y(data, include_segments=2):
    X_data_all = []
    Y_data_all = []

    if include_segments < 2:
        raise ValueError("include_segments must be greater than 1")

    for key in data.keys():
        subject = data[key]


        NeedToSearchFeatures, CorrectSearchFeatures, IncorrectSearchFeatures = subject
        Selected_NeedToSearchFeatures = get_selected_features(NeedToSearchFeatures, include_segments)
        Selected_CorrectSearchFeatures = get_selected_features(CorrectSearchFeatures, include_segments)
        Selected_IncorrectSearchFeatures = get_selected_features(IncorrectSearchFeatures, include_segments)

        NeedToSearch_X, NeedToSearch_Y = get_search_x_y(Selected_NeedToSearchFeatures, label=0)
        CorrectSearch_X, CorrectSearch_Y = get_search_x_y(Selected_CorrectSearchFeatures, label=1)
        IncorrectSearch_X, IncorrectSearch_Y = get_search_x_y(Selected_IncorrectSearchFeatures, label=1)

        X_data, Y_data = combine_data(NeedToSearch_X, CorrectSearch_X, IncorrectSearch_X, NeedToSearch_Y, CorrectSearch_Y, IncorrectSearch_Y)
        X_data_all += X_data
        Y_data_all += Y_data
    return X_data_all, Y_data_all

def create_dataloader(EEG_word_level_embeddings):
    #EEG_word_level_embeddings_normalize = (EEG_word_level_embeddings - np.mean(EEG_word_level_embeddings)) / np.std(EEG_word_level_embeddings)

    # Assuming EEG_synthetic is the generated synthetic EEG data
    #EEG_synthetic_denormalized = (EEG_synthetic * np.max(np.abs(EEG_word_level_embeddings))) + np.mean(EEG_word_level_embeddings)


    EEG_word_level_embeddings_normalize = (EEG_word_level_embeddings - np.mean(EEG_word_level_embeddings)) / np.max(np.abs(EEG_word_level_embeddings))


    float_tensor = torch.tensor(EEG_word_level_embeddings_normalize, dtype=torch.float)
    float_tensor = float_tensor.unsqueeze(1)

    #print(EEG_word_level_embeddings_normalize)
    # Calculate mean and standard deviation
    print(torch.isnan(float_tensor).any())

    train_data = []
    for i in range(len(float_tensor)):
       train_data.append(float_tensor[i])
    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=64)
    return trainloader


def create_noise(batch_size, z_size, mode_z):
    if mode_z == 'uniform':
        input_z = torch.rand(batch_size, z_size)*2 - 1
    elif mode_z == 'normal':
        input_z = torch.randn(batch_size, z_size)
    return input_z

def generate_samples(g_model, input_z):
    # Create random noise as input to the generator
    # Generate samples using the generator model
    with torch.no_grad():
        g_output = g_model(input_z)

    return g_output

def generate_synthetic_samples(gen_model, EEG_word_level_embeddings):
    z_size = 100

    input_z = create_noise(1, z_size, "uniform").to(device)
    g_output = generate_samples(gen_model, input_z)
    # Assuming EEG_synthetic is the generated synthetic EEG data
    EEG_synthetic_denormalized = (g_output * np.max(np.abs(EEG_word_level_embeddings))) + np.mean(
        EEG_word_level_embeddings)

    EEG_synthetic_denormalized = EEG_synthetic_denormalized[0][0].reshape(612)
    return EEG_synthetic_denormalized


