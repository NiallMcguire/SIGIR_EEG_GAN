import argparse
import pickle
import re
import random
import os
import numpy as np
from keras.utils import to_categorical
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import Networks
import torch.nn as nn
import torch.optim as optim


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


    train_path = r"/users/gxb18167/Datasets/ZuCo/EEG_Text_Pairs_Sentence.pkl"
    test_path = r"/users/gxb18167/Datasets/ZuCo/Test_EEG_Text_Pairs_Sentence.pkl"

    # Load the EEG embeddings and labels
    EEG_word_level_embeddings, EEG_word_level_labels = read_EEG_embeddings_labels(train_path)
    Test_EEG_word_level_embeddings, Test_EEG_word_level_labels = read_EEG_embeddings_labels(test_path)

    EEG_word_level_sentences, EEG_sentence_embeddings = get_sentences_EEG(EEG_word_level_labels,
                                                                          EEG_word_level_embeddings)
    Test_EEG_word_level_sentences, Test_EEG_sentence_embeddings = get_sentences_EEG(Test_EEG_word_level_labels,
                                                                                    Test_EEG_word_level_embeddings)

    with open(
            r'/users/gxb18167/Datasets/ZuCo/EEG_Sentiment_Labels.pkl',
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

    # Split the data into training and validation sets
    validation_size = int(0.2 * len(X_train))
    X_val = X_train[:validation_size]
    y_val = train_labels_encoded[:validation_size]
    X_train_numpy = X_train[validation_size:]
    y_train = train_labels_encoded[validation_size:]



    x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(train_labels_encoded, dtype=torch.float32)

    x_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(test_labels_encoded, dtype=torch.float32)

    # Create a custom dataset
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

    # Define batch size
    batch_size = 32  # Adjust according to your preference

    # Create the train loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Create the validation loader
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    # Create the test loader
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)



    # Device configuration

    # Define model parameters
    input_size = 840
    hidden_size = 64
    num_layers = 2
    num_classes = 2

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

        folder_path = f"/users/gxb18167/Datasets/Sentiment/{model}"
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

            # if counter >= patience:
            # print("Early stopping!")
            # break

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

        # save test accuracy
        with open(
                f"/users/gxb18167/Datasets/Sentiment/{model}/Aug_size_{augmentation_size}_Epochs_{epochs}_Aug_Model_{aug_model}_test_accuracy.txt",
                "w") as f:
            f.write(f"Test Accuracy: {accuracy:.4f}")



