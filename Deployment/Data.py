import sys
import nltk
import torch.nn as nn
import numpy as np
nltk.download('punkt')
from gensim.models import Word2Vec
sys.path.insert(0, '..')
import pickle
import torch
import argparse
import Networks


class Data:
    def __init__(self):
        pass

    def create_word_label_embeddings(self, Word_Labels_List, word_embedding_dim=100):
        """
        This function takes in the list of words associated with EEG segments and returns the word embeddings for each word

        :param Word_Labels_List: List of the textual data associated with EEG segments
        :return Embedded_Word_labels: List of each words embeddings
        :return word_embeddings: Dictionary of word embeddings
        """
        tokenized_words = []
        for i in range(len(Word_Labels_List)):
            tokenized_words.append([Word_Labels_List[i]])
        model = Word2Vec(sentences=tokenized_words, vector_size=word_embedding_dim, window=5, min_count=1, workers=4)
        word_embeddings = {word: model.wv[word] for word in model.wv.index_to_key}
        print("Number of word embeddings:", len(word_embeddings))

        Embedded_Word_labels = []
        for word in Word_Labels_List:
            Embedded_Word_labels.append(word_embeddings[word])

        return Embedded_Word_labels, word_embeddings

    def create_dataloader(self, EEG_word_level_embeddings, Embedded_Word_labels):
        """
        This function takes in the EEG word level embeddings and the word labels and returns a dataloader

        :param EEG_word_level_embeddings: The EEG segments of the associated textual information
        :param Embedded_Word_labels: The word embeddings of the associated textual information
        :return trainloader: The dataloader for the EEG word level embeddings and the word labels
        """

        EEG_word_level_embeddings_normalize = (EEG_word_level_embeddings - np.mean(EEG_word_level_embeddings)) / np.max(
            np.abs(EEG_word_level_embeddings))

        float_tensor = torch.tensor(EEG_word_level_embeddings_normalize, dtype=torch.float)
        float_tensor = float_tensor.unsqueeze(1)

        # Sanity check
        print(torch.isnan(float_tensor).any())

        train_data = []
        for i in range(len(float_tensor)):
            train_data.append([float_tensor[i], Embedded_Word_labels[i]])
        trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=64)
        return trainloader

    def create_noise(self, batch_size, z_size, mode_z):
        """
        This function creates noise for the generator

        :param batch_size: Batch size
        :param z_size: Latent input size
        :param mode_z: Mode of the noise
        :return:
        """

        if mode_z == 'uniform':
            input_z = torch.rand(batch_size, z_size) * 2 - 1
        elif mode_z == 'normal':
            input_z = torch.randn(batch_size, z_size)
        return input_z



class GeneratorWGAN_v1(nn.Module):
    """
    This class is the generator for the WGAN
    """
    def __init__(self, noise_dim):
        """
        WGAN Generator constructor

        :param noise_dim: Size of the latent input
        """

        super(GeneratorWGAN_v1, self).__init__()

        self.noise_dim = noise_dim

        # Define the layers of your generator
        self.fc_noise = nn.Linear(noise_dim, 105 * 8)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)



    def forward(self, noise):
        """
        The forward pass for the generator

        :param noise: Takes in the noise for the generator
        :return: Returns the synthetic EEG data
        """

        # Process noise
        noise = self.fc_noise(noise)
        noise = noise.view(noise.size(0), 1, 105, 8)

        # Upsample and generate the output
        z = self.conv1(noise)
        z = self.bn1(z)
        z = self.relu(z)
        z = self.conv2(z)

        return z

class DiscriminatorWGAN_v1(nn.Module):
    """
    This class is the discriminator for the WGAN

    """

    def __init__(self, n_filters):
        """
        The constructor for the discriminator

        :param n_filters: Takes in the number of filters
        """
        super(DiscriminatorWGAN_v1, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters, n_filters*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters*2, n_filters*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(n_filters*4),
            nn.LeakyReLU(0.2),

            nn.Flatten(),  # Flatten spatial dimensions

            # Fully connected layer to reduce to a single value per sample
            nn.Linear(n_filters*4 * (105 // 8) * (8 // 8), 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        The forward pass for the discriminator
        :param input: Takes in the either real or fake EEG data
        :return: Returns the probability of the input being real or fake
        """


        output = self.network(input)
        return output

