import argparse
import pickle
import numpy as np
from scipy.spatial import distance
import nltk
nltk.download('punkt')
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import torch


if __name__ == '__main__':
    print(torch.__version__)
    print("GPU Available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = "cpu"

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str,
                        help='Select model by prefix: DCGAN_v1, DCGAN_v2, WGAN_v1, WGAN_v2, ACGAN_v1, ACGAN_v2, DCGAN_v1_Text, DCGAN_v2_Text, WGAN_v1_Text, WGAN_v2_Text')
    parser.add_argument('--type', type=str, help='normal, random')

    args = parser.parse_args()
    model = args.model
    generation_type = args.type

    # To load the lists from the file:
    with open("/users/gxb18167/Datasets/ZuCo/EEG_Text_Pairs.pkl",
              'rb') as file:
        EEG_word_level_embeddings = pickle.load(file)
        EEG_word_level_labels = pickle.load(file)




