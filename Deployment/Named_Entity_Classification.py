import argparse
import pickle
from math import floor
import random
import numpy as np
import keras
import nltk
import torch
import os
import torch.nn as nn


import Networks
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import pickle

# Assuming x_train and y_train are your input data and target labels, respectively

nltk.download('punkt')
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
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


    return word_embeddings


def get_NE_embeddings(NE_list, word_embeddings):
    NE_list_embeddings = []
    for i in range(len(NE_list)):
        max_length = 7
        words = NE_list[i]
        words_embedding_list = []
        for word in words:
            if word in word_embeddings:
                embedding_word = word_embeddings[word]
                words_embedding_list.append(embedding_word)
            else:
                print(word)
        # padding
        NE_list_embeddings.append(words_embedding_list)

    return NE_list_embeddings

def encode_labels(y):
    label_encoder = {label: idx for idx, label in enumerate(set(y))}
    encoded_labels = [label_encoder[label] for label in y]

    # Step 2: Convert numerical labels to tensor
    encoded_labels_tensor = torch.tensor(encoded_labels)

    # Step 3: Convert numerical labels tensor to one-hot encoded tensor
    num_classes = len(label_encoder)
    y_onehot = F.one_hot(encoded_labels_tensor, num_classes=num_classes).float()

    return y_onehot


def reshape_data(X):
    #reshape the data to 840
    new_list = []
    for i in range(len(X)):
        array_list = X_train_numpy[i]
        arrays_list_reshaped = [arr.reshape(-1) for arr in array_list]
        new_list.append(arrays_list_reshaped)

    new_list = np.array(new_list)

    return new_list

def create_noise(batch_size, z_size, mode_z):
    if mode_z == 'uniform':
        input_z = torch.rand(batch_size, z_size)*2 - 1
    elif mode_z == 'normal':
        input_z = torch.randn(batch_size, z_size)
    return input_z

def generate_samples(generator_name, g_model, input_z, input_t):
    # Create random noise as input to the generator
    # Generate samples using the generator model

    if generator_name == "DCGAN_v1" or generator_name == "DCGAN_v2" or generator_name == "WGAN_v1" or generator_name == "WGAN_v2":
        with torch.no_grad():
            g_output = g_model(input_z)
    else:
        with torch.no_grad():
            g_output = g_model(input_z, input_t)

    return g_output


def augment_dataset(gen_model, generator_name, word_embeddings, EEG_word_level_embeddings, Named_Entity_List, max_length = 7):
    Named_Entity_Augmentation = []

    for word in Named_Entity_List:

        word_embedding = word_embeddings[word]
        input_z = create_noise(1, 100, "uniform").to(device)

        word_embedding_tensor = torch.tensor(word_embedding, dtype=torch.float32)
        word_embedding_tensor = word_embedding_tensor.unsqueeze(0)

        g_output = generate_samples(generator_name, gen_model, input_z, word_embedding_tensor)
        g_output = g_output.to('cpu')

        EEG_synthetic_denormalized = (g_output * np.max(np.abs(EEG_word_level_embeddings))) + np.mean(
            EEG_word_level_embeddings)

        print("D-type:", EEG_synthetic_denormalized[0][0].dtype)

        synthetic_sample = torch.tensor(EEG_synthetic_denormalized[0][0], dtype=torch.float32).to(device)
        synthetic_sample = synthetic_sample.resize(840).to('cpu')
        print("Synthetic Sample Shape inside: ", synthetic_sample.shape)

        Named_Entity_Augmentation.append(synthetic_sample)

    if len(Named_Entity_Augmentation) < max_length:
        padding_count = max_length - len(Named_Entity_Augmentation)
        for i in range(padding_count):
            Named_Entity_Augmentation.append(torch.zeros(840, dtype=torch.float32).to('cpu'))

    Named_Entity_Augmentation = np.array(Named_Entity_Augmentation)
    print("Synthetic Named Entity shape inside", Named_Entity_Augmentation.shape)

    return Named_Entity_Augmentation

def flatten_EEG_labels(NE_list, EEG_list):

    list_of_eeg_segments = []
    list_of_word_labels = []
    for i in range(len(NE_list)):
        Named_Entity = NE_list[i]
        Named_Entity_EEG_Segments = EEG_list[i]
        for j in range(len(Named_Entity)):
            word = Named_Entity[j]
            for EEG_Segments in range(len(Named_Entity_EEG_Segments)):
                EEG_Segment = Named_Entity_EEG_Segments[EEG_Segments][j]
                list_of_eeg_segments.append(EEG_Segment)
                list_of_word_labels.append(word)

    return list_of_eeg_segments, list_of_word_labels





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
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--aug_model', type=str)
    parser.add_argument('--generator_path', type=str)

    args = parser.parse_args()
    model_name = args.model
    augmentation_size = args.augmentation_size
    epochs = args.epochs
    aug_model = args.aug_model
    generator_path = args.generator_path

    # read in train and test data

    train_path = r"/users/gxb18167/Datasets/ZuCo/train_NER.pkl"

    test_path = r"/users/gxb18167/Datasets/ZuCo/test_NER.pkl"

    train_NE, train_EEG_segments, train_Classes = save_lists_to_file(train_path)
    test_NE, test_EEG_segments, test_Classes = save_lists_to_file(test_path)

    # padding
    X_train, y_train, NE_list = padding_x_y(train_EEG_segments, train_Classes, train_NE)
    X_train_numpy = np.array(X_train)
    X_train_numpy = reshape_data(X_train_numpy)
    #y_train_categorical = encode_labels(y_train)

    NE_List_Flat = [word for sublist in NE_list for word in sublist]
    word_embeddings = create_word_label_embeddings(NE_List_Flat)
    NE_embeddings = get_NE_embeddings(NE_list, word_embeddings)

    validation_size = int(0.2 * len(X_train_numpy))
    X_val = X_train_numpy[:validation_size]
    y_val = y_train[:validation_size]
    X_train_numpy = X_train_numpy[validation_size:]
    y_train = y_train[validation_size:]


    X_test, y_test, NE_list_test = padding_x_y(test_EEG_segments, test_Classes, test_NE)


    X_test_numpy = np.array(X_test)
    X_test_numpy = reshape_data(X_test_numpy)
    y_test_categorical = encode_labels(y_test)


    print("Length of Train before aug:", len(X_train_numpy))
    print("Length of Train Label before aug:", len(y_train))

    if augmentation_size > 0:
        print("Augmenting data")
        list_of_eeg_segments, list_of_word_labels = flatten_EEG_labels(train_NE, train_EEG_segments)
        if aug_model == "DCGAN_v2_Text":
            gen_model = Networks.GeneratorDCGAN_v2_Text(100, 50)


        checkpoint = torch.load(
            fr"/users/gxb18167/Datasets/Checkpoints/NER/{aug_model}/{generator_path}",
            map_location=device)

        print("Loading Generator Model from: ", fr"/users/gxb18167/Datasets/Checkpoints/NER/{aug_model}/{generator_path}")


        gen_model.load_state_dict(checkpoint['gen_model_state_dict'])
        gen_model.to(device)
        # Set the model to evaluation mode
        gen_model.eval()

        pairs = list(zip(NE_list, y_train))


        Augmentation_size = floor(int(len(NE_list) / 100 * augmentation_size))
        sampled_words = random.sample(pairs, Augmentation_size)

        print("Augmentation Size: ", Augmentation_size)

        sampled_words, sampled_labels = zip(*sampled_words)

        for i in range(len(sampled_words)):
            Named_Entity = sampled_words[i]
            label = sampled_labels[i]
            Synthetic_Named_Entity = augment_dataset(gen_model, model_name, word_embeddings,list_of_eeg_segments, Named_Entity)

            print("length of Synthetic Named Entity: ", len(Synthetic_Named_Entity))
            print("length of Synthetic Named Entity: ", len(Synthetic_Named_Entity[0]))
            print("Synthetic Named Entity shape", Synthetic_Named_Entity.shape)

            print("Train shape", X_train_numpy.shape)


            #X_train_numpy = np.append(X_train_numpy, Synthetic_Named_Entity, axis=0)
            #y_train = np.append(y_train, label)


    print("Length of Train after aug:", len(X_train_numpy))
    print("Length of Train Label after aug:", len(y_train))

    # Convert numpy arrays to PyTorch tensors

    y_train_categorical = encode_labels(y_train)
    y_val_categorical = encode_labels(y_val)

    x_train_tensor = torch.tensor(X_train_numpy, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_categorical, dtype=torch.float32)  # Assuming your labels are integers

    x_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_categorical, dtype=torch.float32)  # Assuming your labels are integers

    x_test_tensor = torch.tensor(X_test_numpy, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_categorical, dtype=torch.float32)


    # Create a custom dataset
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    # Define batch size
    batch_size = 64

    # Create the train loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # Define model parameters
    input_size = 840
    hidden_size = 64
    num_layers = 2
    num_classes = 3

    # Instantiate the model
    print(model_name)

    if model_name == 'BLSTM_v1':
        model = Networks.BLSTMClassifier(input_size, hidden_size, num_layers, num_classes)
        model.to(device)


    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = epochs

    best_valid_loss = float('inf')
    best_model_state = None
    patience = 50  # Number of epochs to wait for improvement
    counter = 0  # Counter for patience

    folder_path = f"/users/gxb18167/Datasets/NER/{model}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    save_path = folder_path + f'/Aug_size_{augmentation_size}_Epochs_{epochs}_Aug_Model_{aug_model}_best_model.pth'

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()

            outputs = model(batch_x)

            # Convert class probabilities to class indices
            _, predicted = torch.max(outputs, 1)

            loss = criterion(outputs, batch_y.squeeze())  # Ensure target tensor is Long type
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # Validation
        model.eval()
        with torch.no_grad():
            valid_loss = 0
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.squeeze())
                valid_loss += loss.item()

        avg_valid_loss = valid_loss / len(val_loader)
        print(f'Validation Loss: {avg_valid_loss:.4f}')

        # Early stopping and saving the best model
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1

        #if counter >= patience:
            #print("Early stopping!")
            #break

    # Save the best model state to a file
    if best_model_state is not None:
        torch.save(best_model_state, save_path)


    # Test the model
    model.load_state_dict(torch.load(save_path))
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == torch.argmax(batch_y, 1)).sum().item()

        accuracy = correct / total
        print(f'Test Accuracy: {accuracy:.4f}')

    #save test accuracy
    with open(f"/users/gxb18167/Datasets/NER/{model}/Aug_size_{augmentation_size}_Epochs_{epochs}_Aug_Model_{aug_model}_test_accuracy.txt", "w") as f:
        f.write(f"Test Accuracy: {accuracy:.4f}")



