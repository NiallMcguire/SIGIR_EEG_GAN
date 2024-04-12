import argparse
import os
import pickle
import numpy as np
from scipy.spatial import distance
import nltk

from Deployment import Networks

nltk.download('punkt')
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import torch
import Data


def average_eeg_segments(eeg_segments):
    """
    Compute the average EEG segment from a list of EEG segments.

    Parameters:
        eeg_segments (list of array-like): List of EEG segment data.

    Returns:
        array-like: Average EEG segment.
    """
    # Stack EEG segments along a new axis to compute the average
    stacked_segments = np.stack(eeg_segments, axis=0)

    # Compute the mean across segments
    avg_segment = np.mean(stacked_segments, axis=0)

    return avg_segment



def convert_to_probability_distribution(eeg_segment):
    """
    Convert EEG segment to a probability distribution.

    Parameters:
        eeg_segment (array-like): EEG segment data.

    Returns:
        array-like: Probability distribution representing the EEG segment.
    """
    # Resize EEG segment to a 1D array
    flattened_segment = eeg_segment.ravel()

    # Normalize the flattened segment
    normalized_segment = (flattened_segment - np.mean(flattened_segment)) / np.std(flattened_segment)

    # Convert normalized segment into probability values
    # For example, you can apply softmax function
    probabilities = np.exp(normalized_segment) / np.sum(np.exp(normalized_segment))

    return probabilities

def compute_js_distance(p, q):
    """
    Compute Jensen-Shannon distance between two probability distributions.

    Parameters:
        p (array-like): Probability distribution.
        q (array-like): Probability distribution.

    Returns:
        float: Jensen-Shannon distance between distributions.
    """
    # Normalize distributions
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Compute average distribution
    m = 0.5 * (p + q)

    # Compute Jensen-Shannon divergence
    js_divergence = 0.5 * (distance.jensenshannon(p, m) + distance.jensenshannon(q, m))

    return js_divergence

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

def create_noise(batch_size, z_size, mode_z):
    if mode_z == 'uniform':
        input_z = torch.rand(batch_size, z_size)*2 - 1
    elif mode_z == 'normal':
        input_z = torch.randn(batch_size, z_size)
    return input_z


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
    parser.add_argument('--generator_path', type=str, help='Path to generator checkpoint')

    args = parser.parse_args()
    model = args.model
    generation_type = args.type
    generator_path = args.generator_path

    z_size = 100
    word_embedding_dim = 50

    data = Data.Data()

    # To load the lists from the file:
    with open("/users/gxb18167/Datasets/ZuCo/EEG_Text_Pairs.pkl",
              'rb') as file:
        EEG_word_level_embeddings = pickle.load(file)
        EEG_word_level_labels = pickle.load(file)

    Embedded_Word_labels, word_embeddings = data.create_word_label_embeddings(EEG_word_level_labels, word_embedding_dim)

    # create dictionary with words a labels and the EEG embeddings in a list as the values
    EEG_word_level_dict = {}
    for i in range(len(EEG_word_level_labels)):
        if EEG_word_level_labels[i] in EEG_word_level_dict:
            EEG_word_level_dict[EEG_word_level_labels[i]].append(EEG_word_level_embeddings[i])
        else:
            EEG_word_level_dict[EEG_word_level_labels[i]] = [EEG_word_level_embeddings[i]]

    # Dictionary containing average EEG segment for each word
    average_segments_dict = {}
    for word, segments in EEG_word_level_dict.items():
        # Compute average EEG segment for the current word
        avg_segment = average_eeg_segments(segments)

        # Store average segment in dictionary
        average_segments_dict[word] = avg_segment

    probability_distribution_dict = {}

    for word, segment in average_segments_dict.items():
        probability_distribution_dict[word] = convert_to_probability_distribution(segment)

    # Random example
    # Compute Jensen-Shannon distance between two probability distributions

    output_all_results_path = f'/users/gxb18167/Datasets/Checkpoints/train_decoding/results/JS_Distance'
    if not os.path.exists(output_all_results_path):
        os.makedirs(output_all_results_path)

    output_file = f"{model}_{generation_type}_JS_Distance.csv"



    if model == "DCGAN_v1":
        gen_model = Networks.GeneratorDCGAN_v1(z_size).to(device)
    elif model == "DCGAN_v2":
        gen_model = Networks.GeneratorDCGAN_v2(z_size).to(device)
    elif model == "WGAN_v1":
        gen_model = Networks.GeneratorWGAN_v1(z_size).to(device)
    elif model == "WGAN_v2":
        gen_model = Networks.GeneratorWGAN_v2(z_size).to(device)
    elif model == "DCGAN_v1_Text":
        gen_model = Networks.GeneratorDCGAN_v1_Text(z_size, word_embedding_dim).to(device)
    elif model == "DCGAN_v2_Text":
        gen_model = Networks.GeneratorDCGAN_v2_Text(z_size, word_embedding_dim).to(device)
    elif model == "WGAN_v1_Text":
        gen_model = Networks.GeneratorWGAN_v1_Text(z_size, word_embedding_dim).to(device)
    elif model == "WGAN_v2_Text":
        gen_model = Networks.GeneratorWGAN_v2_Text(z_size, word_embedding_dim).to(device)

    random_distance_dict = {}
    normal_distance_dict = {}

    if generation_type == 'random':
        for word, segment in probability_distribution_dict.items():
            mean_eeg = np.mean(average_segments_dict[word])
            std_dev_eeg = np.std(average_segments_dict[word])

            random_value = np.random.normal(loc=mean_eeg, scale=std_dev_eeg, size=average_segments_dict[word].shape)
            random_value = convert_to_probability_distribution(random_value)

            js_distance = compute_js_distance(segment, random_value)
            random_distance_dict[word] = js_distance

            print(f"Word: {word}, JS Distance: {js_distance}")

        with open(f"{output_all_results_path}/{output_file}", 'w') as file:
            for word, js_distance in random_distance_dict.items():
                file.write(f"{word}, {js_distance}\n")


    elif generation_type == 'normal':
        checkpoint = torch.load(
            fr"/users/gxb18167/Datasets/Checkpoints/{model}/{generator_path}",
            map_location=device)

        gen_model.load_state_dict(checkpoint['gen_model_state_dict'])
        gen_model.to(device)
        # Set the model to evaluation mode
        gen_model.eval()

        for word, segment in probability_distribution_dict.items():
            word_embedding = word_embeddings[word]

            input_z = create_noise(1, 100, "uniform").to(device)

            word_embedding_tensor = torch.tensor(word_embedding, dtype=torch.float)
            word_embedding_tensor = word_embedding_tensor.unsqueeze(0)

            g_output = generate_samples(model, gen_model, input_z, word_embedding_tensor)
            g_output = g_output.to('cpu')

            EEG_synthetic_denormalized = (g_output * np.max(np.abs(EEG_word_level_embeddings))) + np.mean(
                EEG_word_level_embeddings)

            synthetic_sample = torch.tensor(EEG_synthetic_denormalized[0][0], dtype=torch.float).to(device)
            synthetic_sample = synthetic_sample.resize(840).to(device)

            synthetic_sample = convert_to_probability_distribution(synthetic_sample)

            js_distance = compute_js_distance(segment, synthetic_sample)

            print(f"Word: {word}, JS Distance: {js_distance}")

            normal_distance_dict[word] = js_distance

        with open(f"{output_all_results_path}/{output_file}", 'w') as file:
            for word, js_distance in normal_distance_dict.items():
                file.write(f"{word}, {js_distance}\n")








