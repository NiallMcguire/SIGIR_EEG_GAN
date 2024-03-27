import pickle
from transformers import BertTokenizer, BertModel
import torch



if __name__ == '__main__':
    # To load the lists from the file:
    with open("/users/gxb18167/Datasets/ZuCo/EEG_Text_Pairs.pkl",
              'rb') as file:
        EEG_word_level_embeddings = pickle.load(file)
        EEG_word_level_labels = pickle.load(file)


    print("Labels len:", len(EEG_word_level_labels))

    word_corpus = EEG_word_level_labels

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Move model to GPU
    model.to(device)

    # Split the word corpus into chunks
    chunk_size = 100  # You can adjust this based on your need
    word_chunks = [word_corpus[i:i + chunk_size] for i in range(0, len(word_corpus), chunk_size)]

    print("Word Corpus len:", len(word_corpus))

    # Initialize a list to store embeddings
    all_embeddings = []

    # Perform model inference for each chunk
    with torch.no_grad():
        for chunk in word_chunks:
            # Convert words to IDs (Tokenizing)
            word_ids = [tokenizer.convert_tokens_to_ids(word) for word in chunk]
            word_ids_tensor = torch.tensor([word_ids]).to(device)

            # Pass word IDs through BERT model to get word embeddings
            outputs = model(word_ids_tensor)

            # Extract embeddings from the last layer of BERT
            word_embeddings = outputs.last_hidden_state.squeeze(0)

            # Append embeddings to the list
            all_embeddings.append(word_embeddings)


    # Concatenate embeddings from all chunks along dimension 0
    final_embeddings = torch.cat(all_embeddings, dim=0)

    print("Final embeddings len:", len(final_embeddings))

    # Create a dictionary to map words to their embeddings
    word_to_embedding = {}
    current_idx = 0

    # Iterate through word corpus and embeddings
    for chunk in word_chunks:
        for word, embedding in zip(chunk, all_embeddings[current_idx]):
            word_to_embedding[word] = embedding.cpu().numpy()  # Convert to numpy array if needed
        current_idx += 1

    # Example: Lookup embedding for a given word
    #word = 'reason'
    #embedding = word_to_embedding[word]
    #print(embedding)

    # Save the embeddings to a file
    with open("/users/gxb18167/Datasets/ZuCo/EEG_BERT_Embeddings.pkl", 'wb') as file:
        pickle.dump(word_to_embedding, file)







