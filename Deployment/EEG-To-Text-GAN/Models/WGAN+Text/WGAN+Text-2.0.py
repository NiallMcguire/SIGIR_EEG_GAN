import sys
import nltk
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
nltk.download('punkt')
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
sys.path.insert(0, '..')
import pickle
from torch.autograd import grad as torch_grad



def create_word_label_embeddings(Word_Labels_List):
    tokenized_words = []
    for i in range(len(Word_Labels_List)):
        tokenized_words.append([Word_Labels_List[i]])
    model = Word2Vec(sentences=tokenized_words, vector_size=word_embedding_dim, window=5, min_count=1, workers=4)
    word_embeddings = {word: model.wv[word] for word in model.wv.index_to_key}
    print("Number of word embeddings:", len(word_embeddings))
    #word, embedding = list(word_embeddings.items())[10]
    #print(f"Word: {word}, Embedding: {embedding}")

    Embedded_Word_labels = []
    for word in EEG_word_level_labels:
        Embedded_Word_labels.append(word_embeddings[word])

    return Embedded_Word_labels, word_embeddings

def create_dataloader(EEG_word_level_embeddings, Embedded_Word_labels):
    #EEG_word_level_embeddings_normalize = (EEG_word_level_embeddings - np.mean(EEG_word_level_embeddings)) / np.std(EEG_word_level_embeddings)

    # Assuming EEG_synthetic is the generated synthetic EEG data
    #EEG_synthetic_denormalized = (EEG_synthetic * np.max(np.abs(EEG_word_level_embeddings))) + np.mean(EEG_word_level_embeddings)


    EEG_word_level_embeddings_normalize = (EEG_word_level_embeddings - np.mean(EEG_word_level_embeddings)) / np.max(np.abs(EEG_word_level_embeddings))


    float_tensor = torch.tensor(EEG_word_level_embeddings_normalize, dtype=torch.float)
    float_tensor = float_tensor.unsqueeze(1)

    #print(EEG_word_level_embeddings_normalize)
    # Calculate mean and standard deviation
    print(torch.isnan(float_tensor).any())

    train_data = []
    for i in range(len(float_tensor)):
       train_data.append([float_tensor[i], Embedded_Word_labels[i]])
    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=64)
    return trainloader

def create_noise(batch_size, z_size, mode_z):
    if mode_z == 'uniform':
        input_z = torch.rand(batch_size, z_size)*2 - 1
    elif mode_z == 'normal':
        input_z = torch.randn(batch_size, z_size)
    return input_z



def gradient_penalty(real_data, generated_data):
    batch_size = real_data.size(0)

    # Calculate interpolation
    alpha = torch.rand(real_data.shape[0], 1, 1, 1, requires_grad=True, device=device)
    #print("Gen:", generated_data.shape)
    interpolated = alpha * real_data + (1 - alpha) * generated_data

    # Calculate probability of interpolated examples
    proba_interpolated = disc_model(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=proba_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(proba_interpolated.size(), device=device),
                           create_graph=True, retain_graph=True)[0]

    gradients = gradients.view(batch_size, -1)
    gradients_norm = gradients.norm(2, dim=1)
    return lambda_gp * ((gradients_norm - 1)**2).mean()

## Train the discriminator
def d_train_wgan(x, input_t):
    disc_model.zero_grad()

    batch_size = x.size(0)
    x = x.to(device)
    #print("X:", x.shape)
    # Calculate probabilities on real and generated data
    d_real = disc_model(x)
    input_z = create_noise(batch_size, z_size, mode_z).to(device)
    g_output = gen_model(input_z, input_t)
    #print("D Real:", d_real.shape)

    d_generated = disc_model(g_output)
    #print("G output:", g_output.shape)

    d_loss = d_generated.mean() - d_real.mean() + gradient_penalty(x.data, g_output.data)

    d_loss.backward()
    d_optimizer.step()

    return d_loss.data.item()

## Train the generator
def g_train_wgan(x, input_t):
    gen_model.zero_grad()

    batch_size = x.size(0)
    input_z = create_noise(batch_size, z_size, mode_z).to(device)

    g_output = gen_model(input_z, input_t)

    d_generated = disc_model(g_output)
    g_loss = -d_generated.mean()
    print("G Loss:", g_loss)

    # gradient backprop & optimize ONLY G's parameters
    g_loss.backward()
    g_optimizer.step()

    return g_loss.data.item()

class Generator(nn.Module):
    def __init__(self, noise_dim, word_embedding_dim, output_shape):
        super(Generator, self).__init__()

        self.noise_dim = noise_dim
        self.word_embedding_dim = word_embedding_dim

        # Define the layers of your generator
        self.fc_noise = nn.Linear(noise_dim, 105*8)  # Increase the size for more complexity
        self.fc_word_embedding = nn.Linear(word_embedding_dim, 105*8)  # Increase the size for more complexity
        self.conv1 = nn.Conv2d(2, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, noise, word_embedding):
        # Process noise
        noise = self.fc_noise(noise)
        noise = noise.view(noise.size(0), 1, 105,8)  # Adjust the size to match conv1

        # Process word embedding
        word_embedding = self.fc_word_embedding(word_embedding.to(device))
        word_embedding = word_embedding.view(word_embedding.size(0), 1, 105, 8)  # Adjust the size to match conv1

        # Concatenate noise and word embedding
        combined_input = torch.cat([noise, word_embedding], dim=1)

        # Upsample and generate the output
        z = self.conv1(combined_input)
        z = self.bn1(z)
        z = self.relu(z)

        z = self.conv2(z)
        z = self.bn2(z)
        z = self.relu(z)

        z = self.conv3(z)
        z = self.tanh(z)

        return z

class DiscriminatorWGAN(nn.Module):
    def __init__(self, n_filters):
        super(DiscriminatorWGAN, self).__init__()
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
        output = self.network(input)
        return output




if __name__ == '__main__':
    print(torch.__version__)
    print("GPU Available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = "cpu"


    batch_size = 64
    word_embedding_dim = 50
    output_shape = (1, 105, 8)
    torch.manual_seed(1)
    np.random.seed(1)
    z_size = 100
    image_size = (105, 8)
    n_filters = 32

    # To load the lists from the file:
    with open("/users/gxb18167/Datasets/ZuCo/EEG_Text_Pairs.pkl",
              'rb') as file:
        EEG_word_level_embeddings = pickle.load(file)
        EEG_word_level_labels = pickle.load(file)

    Embedded_Word_labels, word_embeddings = create_word_label_embeddings(EEG_word_level_labels)
    trainloader = create_dataloader(EEG_word_level_embeddings, Embedded_Word_labels)

    mode_z = 'uniform'
    fixed_z = create_noise(batch_size, z_size, mode_z).to(device)

    noise = create_noise(batch_size, z_size, "uniform")

    gen_model = Generator(z_size, word_embedding_dim, output_shape).to(device)
    disc_model = DiscriminatorWGAN(n_filters).to(device)

    g_optimizer = torch.optim.Adam(gen_model.parameters(), 0.00002)
    d_optimizer = torch.optim.Adam(disc_model.parameters(), 0.00002)

    epoch_samples_wgan = []
    lambda_gp = 10.0
    num_epochs = 100
    torch.manual_seed(1)
    critic_iterations = 5
    save_interval = 5
    checkpoint_path = '/users/Datasets/Checkpoints/WGAN+Text_2.0/Textual_WGAN_GP_checkpoint_epoch_{}.pth'
    final_model_path = '/users/Datasets/Checkpoints/WGAN+Text_2.0/Textual_WGAN_GP_model_final.pth'

    for epoch in range(1, num_epochs + 1):
        gen_model.train()
        d_losses, g_losses = [], []
        for i, (x, t) in enumerate(trainloader):
            # print("T:", t)
            for _ in range(critic_iterations):
                d_loss = d_train_wgan(x, t)
                print("D Loss:", d_loss)
            d_losses.append(d_loss)
            g_losses.append(g_train_wgan(x, t))

        print(f'Epoch {epoch:03d} | D Loss >>'
              f' {torch.FloatTensor(d_losses).mean():.4f}')
        print(f'Epoch {epoch:03d} | G Loss >>'
              f' {torch.FloatTensor(g_losses).mean():.4f}')

        # Save checkpoints at regular intervals
        if epoch % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'gen_model_state_dict': gen_model.state_dict(),
                'optimizer_state_dict': g_optimizer.state_dict(),
                'd_losses': d_losses,
                'g_losses': g_losses,
            }, checkpoint_path.format(epoch))

        '''
        gen_model.eval()
        epoch_samples_wgan.append(
            create_samples(gen_model, fixed_z, t).detach().cpu().numpy())
        '''
    # Save the final model after training is complete
    torch.save({
        'epoch': num_epochs,
        'gen_model_state_dict': gen_model.state_dict(),
        'optimizer_state_dict': g_optimizer.state_dict(),
        'd_losses': d_losses,
        'g_losses': g_losses,
    }, final_model_path)
