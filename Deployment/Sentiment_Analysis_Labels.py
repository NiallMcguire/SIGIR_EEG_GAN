import pickle
import re

from transformers import pipeline
sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")


def read_EEG_embeddings_labels(path):
    with open(path, 'rb') as file:
        EEG_word_level_embeddings = pickle.load(file)
        EEG_word_level_labels = pickle.load(file)
    return EEG_word_level_embeddings, EEG_word_level_labels


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


def get_sentiment_label(list_of_sentence):
    sentiment_labels = []
    for sentence in list_of_sentence:
        sentence = " ".join(sentence)
        sentiment_results = sentiment_analysis(sentence)
        sentiment = sentiment_results[0]['label']

        sentiment_labels.append(sentiment)
    return sentiment_labels


if __name__ == "__main__":
    train_path = r"/users/gxb18167/Datasets/ZuCo/EEG_Text_Pairs_Sentence.pkl"
    test_path = r"/users/gxb18167/Datasets/ZuCo/Test_EEG_Text_Pairs_Sentence.pkl"

    EEG_word_level_embeddings, EEG_word_level_labels = read_EEG_embeddings_labels(train_path)
    Test_EEG_word_level_embeddings, Test_EEG_word_level_labels = read_EEG_embeddings_labels(test_path)

    EEG_word_level_sentences, EEG_sentence_embeddings = get_sentences_EEG(EEG_word_level_labels,
                                                                          EEG_word_level_embeddings)
    Test_EEG_word_level_sentences, Test_EEG_sentence_embeddings = get_sentences_EEG(Test_EEG_word_level_labels,
                                                                                    Test_EEG_word_level_embeddings)

    EEG_word_level_sentences_labels = get_sentiment_label(EEG_word_level_sentences)
    Test_EEG_word_level_sentences_labels = get_sentiment_label(Test_EEG_word_level_sentences)

    #write the labels to a file
    with open(r"C:\Users\gxb18167\PycharmProjects\EEG-To-Text\SIGIR_Development\EEG-GAN\EEG_Sentiment_Labels.pkl", 'wb') as file:
        pickle.dump(EEG_word_level_sentences_labels, file)
        pickle.dump(Test_EEG_word_level_sentences_labels, file)