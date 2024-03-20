import sys
import nltk
import torch.nn as nn
import numpy as np
nltk.download('punkt')
from gensim.models import Word2Vec
sys.path.insert(0, '..')
import torch



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

