import pickle
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

    def create_word_label_embeddings(self, Word_Labels_List, word_embedding_dim=50):
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


    def create_word_label_embeddings_bert(self, Word_Labels_List, word_embedding_dim=50):
        """
        This function takes in the list of words associated with EEG segments and returns the word embeddings for each word

        :param Word_Labels_List: List of the textual data associated with EEG segments
        :return Embedded_Word_labels: List of each words embeddings
        :return word_embeddings: Dictionary of word embeddings
        """

        with open("/users/gxb18167/Datasets/ZuCo/EEG_BERT_Embeddings.pkl",
                  'rb') as file:
            word_to_embedding = pickle.load(file)


        Embedded_Word_labels = []
        for word in Word_Labels_List:
            Embedded_Word_labels.append(word_to_embedding[word])

        return Embedded_Word_labels, word_to_embedding

    def create_word_label_embeddings_contextual(self, Word_Labels_List, word_embedding_dim=50):
        tokenized_words = []
        for i in range(len(Word_Labels_List)):
            tokenized_words.append([Word_Labels_List[i]])
        model = Word2Vec(sentences=tokenized_words, vector_size=word_embedding_dim, window=5, min_count=1, workers=4)
        word_embeddings = {word: model.wv[word] for word in model.wv.index_to_key}
        Embedded_Word_labels = []

        for words in range(0, len(Word_Labels_List)):
            current_word = Word_Labels_List[words]
            if current_word != "SOS" and words != len(Word_Labels_List) - 1:
                prior_word = Word_Labels_List[words - 1]

                current_word = Word_Labels_List[words]

                next_word = Word_Labels_List[words + 1]

                contextual_embedding = np.concatenate(
                    (word_embeddings[prior_word], word_embeddings[current_word], word_embeddings[next_word]), axis=-1)
                Embedded_Word_labels.append(contextual_embedding)
            elif words == len(Word_Labels_List) - 1:
                prior_word = Word_Labels_List[words - 1]
                next_word = "SOS"
                contextual_embedding = np.concatenate(
                    (word_embeddings[prior_word], word_embeddings[current_word], word_embeddings[next_word]), axis=-1)
                Embedded_Word_labels.append(contextual_embedding)

        return Embedded_Word_labels, word_embeddings

    def create_word_label_embeddings_sentence(self, Word_Labels_List, EEG_word_level_embeddings, word_embedding_dim=50):

        tokenized_words = []
        for i in range(len(Word_Labels_List)):
            tokenized_words.append([Word_Labels_List[i]])
        model = Word2Vec(sentences=tokenized_words, vector_size=word_embedding_dim, window=5, min_count=1, workers=4)
        word_embeddings = {word: model.wv[word] for word in model.wv.index_to_key}

        list_of_sentences = []
        sentence = []

        max_sentence_length = 0
        for words in range(0, len(Word_Labels_List)):

            current_word = Word_Labels_List[words]
            current_word_embedding = word_embeddings[current_word]

            if words == len(Word_Labels_List) - 1:
                if len(sentence) > max_sentence_length:
                    max_sentence_length = len(sentence)
                list_of_sentences.append(sentence)

            elif current_word == "SOS" and sentence != []:
                if len(sentence) > max_sentence_length:
                    max_sentence_length = len(sentence)

                list_of_sentences.append(sentence)
                sentence = []
            elif current_word != "SOS":
                sentence.append(current_word_embedding)

        EEG_sentence_list = []
        index_counter = 0
        for sentence in list_of_sentences:
            EEG_sentence = EEG_word_level_embeddings[index_counter:index_counter + len(sentence)]
            for i in range(max_sentence_length - len(EEG_sentence)):
                EEG_sentence.append(np.zeros((105, 8)))

            EEG_sentence_list.append(EEG_sentence)
            index_counter += len(sentence)
            if len(sentence) > max_sentence_length:
                max_sentence_length = len(sentence)

        for sentence in list_of_sentences:
            for i in range(max_sentence_length - len(sentence)):
                sentence.append(np.zeros((word_embedding_dim)))

        for i in range(len(EEG_sentence_list)):
            EEG_sentence = EEG_sentence_list[i]
            contact_EEG_sentence = np.concatenate(EEG_sentence, axis=1)
            EEG_sentence_list[i] = contact_EEG_sentence

            word_embedding_sentence = list_of_sentences[i]
            concat_word_embedding_sentence = np.concatenate(word_embedding_sentence, axis=0)
            list_of_sentences[i] = concat_word_embedding_sentence

        return EEG_sentence_list, list_of_sentences

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

    def create_dataloader_sentence(self, EEG_word_level_embeddings, Embedded_Word_labels):
        EEG_word_level_embeddings_normalize = (EEG_word_level_embeddings - np.mean(EEG_word_level_embeddings)) / np.max(
            np.abs(EEG_word_level_embeddings))

        float_tensor = torch.tensor(EEG_word_level_embeddings_normalize, dtype=torch.float32)
        float_tensor = float_tensor.unsqueeze(1)

        # Calculate mean and standard deviation
        print(torch.isnan(float_tensor).any())

        Embedded_Word_labels = torch.tensor(Embedded_Word_labels, dtype=torch.float32)

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

