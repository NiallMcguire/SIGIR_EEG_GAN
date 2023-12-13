import pickle
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from torch import nn

import Generate_Samples as gs


def load_in_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

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

def get_search_x_y(SearchFeatures, label, augmentation_factor=0, gen_model=None):
    X_data = []
    Y_Data = []

    for index in range(len(SearchFeatures)):
        sentence_eeg = SearchFeatures[index]
        for eeg in sentence_eeg:
            #print(eeg.shape)
            X_data.append(eeg)
            Y_Data.append(label)

    if augmentation_factor > 0:
        augmentation_size = len(X_data)/100*augmentation_factor
        print("Augmentation Size:", augmentation_size)
        for i in range(int(augmentation_size)):
            synthetic_eeg = gs.generate_synthetic_samples(gen_model=gen_model, EEG_word_level_embeddings=X_data)
            X_data.append(synthetic_eeg)
            Y_Data.append(label)
    return X_data, Y_Data

def combine_data(NeedToSearch_X, CorrectSearch_X, IncorrectSearch_X, NeedToSearch_Y, CorrectSearch_Y, IncorrectSearch_Y):
    X_data = NeedToSearch_X + CorrectSearch_X[:len(NeedToSearch_X)] + IncorrectSearch_X[:len(NeedToSearch_X)]
    Y_data = NeedToSearch_Y + CorrectSearch_Y[:len(NeedToSearch_X)] + IncorrectSearch_Y[:len(NeedToSearch_X)]
    return X_data, Y_data

def get_all_subject_x_y(data, include_segments=2, augmentation_factor=0, gen_model=None):
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

        NeedToSearch_X, NeedToSearch_Y = get_search_x_y(Selected_NeedToSearchFeatures, label=0, augmentation_factor=augmentation_factor, gen_model=gen_model)
        CorrectSearch_X, CorrectSearch_Y = get_search_x_y(Selected_CorrectSearchFeatures, label=1)
        IncorrectSearch_X, IncorrectSearch_Y = get_search_x_y(Selected_IncorrectSearchFeatures, label=1)

        X_data, Y_data = combine_data(NeedToSearch_X, CorrectSearch_X, IncorrectSearch_X, NeedToSearch_Y, CorrectSearch_Y, IncorrectSearch_Y)
        X_data_all += X_data
        Y_data_all += Y_data
    return X_data_all, Y_data_all


def get_metrics(model, X_Test, Y_Test):
    y_pred = model.predict(X_Test)
    precision, recall, f1, support = precision_recall_fscore_support(Y_Test, y_pred, average='weighted')
    accuracy = accuracy_score(Y_Test, y_pred)
    print(precision, recall, f1, accuracy)
    metrics = [precision, recall, f1, accuracy]
    return accuracy

class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()

        self.noise_dim = noise_dim
        #self.word_embedding_dim = word_embedding_dim

        # Define the layers of your generator
        self.fc_noise = nn.Linear(noise_dim, 68*9)  # Increase the size for more complexity
        #self.fc_word_embedding = nn.Linear(word_embedding_dim, 105*8)  # Increase the size for more complexity
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, noise):
        # Process noise
        noise = self.fc_noise(noise)
        noise = noise.view(noise.size(0), 1, 68,9)  # Adjust the size to match conv1

        # Process word embedding
        #word_embedding = self.fc_word_embedding(word_embedding)
        #word_embedding = word_embedding.view(word_embedding.size(0), 1, 105, 8)  # Adjust the size to match conv1

        # Concatenate noise and word embedding
        #combined_input = torch.cat([noise, word_embedding], dim=1)

        # Upsample and generate the output
        z = self.conv1(noise)
        z = self.bn1(z)
        z = self.relu(z)

        z = self.conv2(z)
        z = self.bn2(z)
        z = self.relu(z)

        z = self.conv3(z)
        z = self.tanh(z)

        return z


if __name__ == "__main__":
    path = r"C:\Users\gxb18167\PycharmProjects\SIGIR_EEG_GAN\Development\Information-Need\Data\stat_features\Participant_Features.pkl"
    data = load_in_data(path)

    z_size = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    gen_model = Generator(z_size).to(device)
    checkpoint = torch.load(r"I:\Science\CIS-YASHMOSH\niallmcguire\WGAN_2.0\checkpoint_epoch_100.pt",
                            map_location=torch.device('cpu'))
    # Load the model's state_dict onto the CPU
    gen_model.load_state_dict(checkpoint['gen_model_state_dict'])
    # Set the model to evaluation mode
    gen_model.eval()

    X, Y = get_all_subject_x_y(data, include_segments=5, augmentation_factor=85, gen_model=gen_model)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    clf = RandomForestClassifier(max_depth=20, random_state=0)
    clf.fit(X_train, y_train)


    get_metrics(clf, X_test, y_test)