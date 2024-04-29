import pickle
import re
import random
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset


def read_EEG_embeddings_labels(path):
    with open(path, 'rb') as file:
        EEG_word_level_embeddings = pickle.load(file)
        EEG_word_level_labels = pickle.load(file)
    return EEG_word_level_embeddings, EEG_word_level_labels



def encode_labels(y):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(y)

    y_categorical = to_categorical(encoded_labels)

    return y_categorical


def get_sentences_EEG(labels, EEG_embeddings):
    Sentences = []
    current_sentence = []

    EEG_Sentencs = []
    EEG_index = 0
    for i in range(len(labels)):
        # Check if the word marks the start of a new sentence
        word = labels[i]
        if word == "SOS":
            # If it does, append the current sentence to the list of sentences
            if len(current_sentence) > 0:
                Sentences.append(current_sentence)
                sentence_length = len(current_sentence)
                #print(EEG_index)
                #print(sentence_length)
                EEG_segment = EEG_embeddings[EEG_index:EEG_index+sentence_length]
                EEG_index += sentence_length
                EEG_Sentencs.append(EEG_segment)

                # Start a new sentence
                current_sentence = []
        else:
            # Add the word to the current sentence
            current_sentence.append(word)

    return Sentences, EEG_Sentencs


def pad_sentences(EEG_embeddings, max_length):
    # Pad the sentences to the maximum length
    padded_EEG_sentences = []
    for index in range(len(EEG_embeddings)):
        sentence = EEG_embeddings[index]
        sentence_length = len(sentence)
        if sentence_length < max_length:
            padding_length = max_length - sentence_length
            for _ in range(padding_length):
                sentence.append(np.zeros((105,8)))
        padded_EEG_sentences.append(sentence)
    return padded_EEG_sentences


def reshape_data(X):
    #reshape the data to 840
    new_list = []
    for i in range(len(X)):
        array_list = X[i]
        arrays_list_reshaped = [arr.reshape(-1) for arr in array_list]
        new_list.append(arrays_list_reshaped)

    new_list = np.array(new_list)
    return new_list


if __name__ == '__main__':
    train_path = r"C:\Users\gxb18167\PycharmProjects\EEG-To-Text\SIGIR_Development\EEG-GAN\EEG_Text_Pairs_Sentence.pkl"
    test_path = r"C:\Users\gxb18167\PycharmProjects\EEG-To-Text\SIGIR_Development\EEG-GAN\Test_EEG_Text_Pairs_Sentence.pkl"


    # Load the EEG embeddings and labels

    EEG_word_level_embeddings, EEG_word_level_labels = read_EEG_embeddings_labels(train_path)
    Test_EEG_word_level_embeddings, Test_EEG_word_level_labels = read_EEG_embeddings_labels(test_path)

    EEG_word_level_sentences, EEG_sentence_embeddings = get_sentences_EEG(EEG_word_level_labels,
                                                                          EEG_word_level_embeddings)
    Test_EEG_word_level_sentences, Test_EEG_sentence_embeddings = get_sentences_EEG(Test_EEG_word_level_labels,
                                                                                    Test_EEG_word_level_embeddings)

    with open(
            r'C:\Users\gxb18167\PycharmProjects\SIGIR_EEG_GAN\Development\Sentiment_Analysis\EEG_Sentiment_Labels.pkl',
            'rb') as f:
        # read two arrays
        train_labels = pickle.load(f)
        test_labels = pickle.load(f)

    max_length = max([len(sentence) for sentence in EEG_word_level_sentences])

    X_train = pad_sentences(EEG_sentence_embeddings, max_length)
    X_train = reshape_data(X_train)

    X_test = pad_sentences(Test_EEG_sentence_embeddings, max_length)
    X_test = reshape_data(X_test)

    train_labels_encoded = encode_labels(train_labels)
    test_labels_encoded = encode_labels(test_labels)

    x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(train_labels_encoded, dtype=torch.float32)

    x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(test_labels_encoded, dtype=torch.float32)

    # Create a custom dataset
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    # Define batch size
    batch_size = 32  # Adjust according to your preference

    # Create the train loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # classifier
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



