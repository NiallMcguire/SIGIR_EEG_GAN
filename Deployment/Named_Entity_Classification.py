import argparse
import pickle
import numpy as np
import nltk
import torch
import torch.nn as nn
import Networks
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, TensorDataset

# Assuming x_train and y_train are your input data and target labels, respectively

nltk.download('punkt')
from gensim.models import Word2Vec
from keras.utils import to_categorical
from nltk.tokenize import word_tokenize
import pickle
from sklearn.preprocessing import LabelEncoder

def save_lists_to_file(path):
    # Open the pickle file in binary write mode
    with open(path, 'rb') as f:
    # Load each list from the file
        NE = pickle.load(f)
        EEG_segments = pickle.load(f)
        Classes = pickle.load(f)

    return NE, EEG_segments, Classes


def padding_x_y(EEG_segments, Classes, named_entity_list):
    X = []
    y = []
    NE = []
    for i in range(len(EEG_segments)):
        named_entity = named_entity_list[i]
        label = Classes[i][0]
        #print(label)
        EEG_list = EEG_segments[i]
        for EEG in EEG_list:
            if EEG != []:
                X.append(EEG)
                y.append(label)
                NE.append(named_entity)
    max_seq_length = max([len(x) for x in X])
    #paddding
    for i in range(len(X)):
        padding_count = max_seq_length - len(X[i])
        #print(padding_count)
        for j in range(padding_count):
            X[i].append(np.zeros((105,8)))

    return X, y, NE

def create_word_label_embeddings(Word_Labels_List):
    tokenized_words = []
    for i in range(len(Word_Labels_List)):
        tokenized_words.append([Word_Labels_List[i]])
    model = Word2Vec(sentences=tokenized_words, vector_size=50, window=5, min_count=1, workers=4)
    word_embeddings = {word: model.wv[word] for word in model.wv.index_to_key}
    print("Number of word embeddings:", len(word_embeddings))
    #word, embedding = list(word_embeddings.items())[10]
    #print(f"Word: {word}, Embedding: {embedding}")

    Embedded_Word_labels = []
    for word in EEG_word_level_labels:
        Embedded_Word_labels.append(word_embeddings[word])

    return Embedded_Word_labels, word_embeddings

def encode_labels(y):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(y)

    y_categorical = to_categorical(encoded_labels)

    return y_categorical


def reshape_data(X):
    #reshape the data to 840
    new_list = []
    for i in range(len(X)):
        array_list = X_train_numpy[i]
        arrays_list_reshaped = [arr.reshape(-1) for arr in array_list]
        new_list.append(arrays_list_reshaped)

    new_list = np.array(new_list)

    return new_list

if __name__ == '__main__':
    print(torch.__version__)
    print("GPU Available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = "cpu"

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str)
    parser.add_argument('--augmentation_size', type=int)

    args = parser.parse_args()
    model = args.model
    Generation_Size = args.augmentation_size

    # read in train and test data

    train_path = r"/users/gxb18167/Datasets/ZuCo/train_NER.pkl"

    test_path = r"/users/gxb18167/Datasets/ZuCo/test_NER.pkl"

    # To load the lists from the file:
    with open("/users/gxb18167/Datasets/ZuCo/EEG_Text_Pairs.pkl",
              'rb') as file:
        EEG_word_level_embeddings = pickle.load(file)
        EEG_word_level_labels = pickle.load(file)

    Embedded_Word_labels, word_embeddings = create_word_label_embeddings(EEG_word_level_labels)
    train_NE, train_EEG_segments, train_Classes = save_lists_to_file(train_path)
    test_NE, test_EEG_segments, test_Classes = save_lists_to_file(test_path)

    # padding
    X_train, y_train, NE_list = padding_x_y(train_EEG_segments, train_Classes, train_NE)
    X_train_numpy = np.array(X_train)
    X_train_numpy = reshape_data(X_train_numpy)
    y_train_categorical = encode_labels(y_train)

    X_test, y_test, NE_list_test = padding_x_y(test_EEG_segments, test_Classes, test_NE)
    X_test_numpy = np.array(X_test)
    X_test_numpy = reshape_data(X_test_numpy)
    y_test_categorical = encode_labels(y_test)


    # Convert numpy arrays to PyTorch tensors
    x_train_tensor = torch.tensor(X_train_numpy, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_categorical, dtype=torch.float32)  # Assuming your labels are integers

    # Create a custom dataset
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

    # Define batch size
    batch_size = 32  # Adjust according to your preference

    # Create the train loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Define model parameters
    input_size = 840
    hidden_size = 64
    num_layers = 2
    num_classes = 3

    # Instantiate the model
    model = Networks.BLSTMClassifier(input_size, hidden_size, num_layers, num_classes)
    model.to(device)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)