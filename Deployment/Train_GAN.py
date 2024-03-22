import sys
import nltk
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder
nltk.download('punkt')
from torch.autograd import grad as torch_grad
sys.path.insert(0, '..')
import pickle
import torch
import argparse
import Networks
import Data
import os


def d_train(x):
    """
    This function trains the discriminator

    :param x: The real EEG data
    :return: The discriminator loss, the probability of the real data, and the probability of the fake data
    """
    disc_model.zero_grad()

    # Train discriminator with a real batch
    batch_size = x.size(0)
    x = x.to(device)
    d_labels_real = torch.ones(batch_size, 1, device=device)

    d_proba_real = disc_model(x)
    d_loss_real = loss_fn(d_proba_real, d_labels_real)

    # Train discriminator on a fake batch
    input_z = data.create_noise(batch_size, z_size, mode_z).to(device)
    g_output = gen_model(input_z)

    d_proba_fake = disc_model(g_output)
    d_labels_fake = torch.zeros(batch_size, 1, device=device)
    d_loss_fake = loss_fn(d_proba_fake, d_labels_fake)

    # gradient backprop & optimize ONLY D's parameters
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    d_optimizer.step()

    return d_loss.data.item(), d_proba_real.detach(), d_proba_fake.detach()

def g_train(x):
    """
    This function trains the generator

    :param x: The real EEG data
    :return: The generator loss
    """
    gen_model.zero_grad()

    batch_size = x.size(0)
    input_z = data.create_noise(batch_size, z_size, mode_z).to(device)
    g_labels_real = torch.ones((batch_size, 1), device=device)

    g_output = gen_model(input_z)
    d_proba_fake = disc_model(g_output)
    g_loss = loss_fn(d_proba_fake, g_labels_real)

    # gradient backprop & optimize ONLY G's parameters
    g_loss.backward()
    g_optimizer.step()

    return g_loss.data.item()

def d_train_text(x, T):
    disc_model.zero_grad()

    # Train discriminator with a real batch
    batch_size = x.size(0)
    x = x.to(device)
    d_labels_real = torch.ones(batch_size, 1, device=device)

    d_proba_real = disc_model(x)
    d_loss_real = loss_fn(d_proba_real, d_labels_real)

    # Train discriminator on a fake batch
    input_z = data.create_noise(batch_size, z_size, mode_z).to(device)
    g_output = gen_model(input_z, T)

    d_proba_fake = disc_model(g_output)
    d_labels_fake = torch.zeros(batch_size, 1, device=device)
    d_loss_fake = loss_fn(d_proba_fake, d_labels_fake)

    # gradient backprop & optimize ONLY D's parameters
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    d_optimizer.step()

    return d_loss.data.item(), d_proba_real.detach(), d_proba_fake.detach()

def g_train_text(x, T):
    gen_model.zero_grad()

    batch_size = x.size(0)
    input_z = data.create_noise(batch_size, z_size, mode_z).to(device)
    g_labels_real = torch.ones((batch_size, 1), device=device)

    g_output = gen_model(input_z, T)
    d_proba_fake = disc_model(g_output)
    g_loss = loss_fn(d_proba_fake, g_labels_real)

    # gradient backprop & optimize ONLY G's parameters
    g_loss.backward()
    g_optimizer.step()

    return g_loss.data.item()


def gradient_penalty(real_data, generated_data, lambda_gp=10.0):
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


def gradient_penalty_text(real_data, generated_data, input_t, lambda_gp=10.0):
    batch_size = real_data.size(0)

    # Calculate interpolation
    alpha = torch.rand(real_data.shape[0], 1, 1, 1, requires_grad=True, device=device)
    #print("Gen:", generated_data.shape)
    interpolated = alpha * real_data + (1 - alpha) * generated_data

    # Calculate probability of interpolated examples
    proba_interpolated = disc_model(interpolated, input_t)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=proba_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(proba_interpolated.size(), device=device),
                           create_graph=True, retain_graph=True)[0]

    gradients = gradients.view(batch_size, -1)
    gradients_norm = gradients.norm(2, dim=1)
    return lambda_gp * ((gradients_norm - 1)**2).mean()


def d_train_wgan(x):
    disc_model.zero_grad()

    batch_size = x.size(0)
    x = x.to(device)

    # Calculate probabilities on real and generated data
    d_real = disc_model(x)
    input_z = data.create_noise(batch_size, z_size, mode_z).to(device)
    g_output = gen_model(input_z)

    d_generated = disc_model(g_output)

    d_loss = d_generated.mean() - d_real.mean() + gradient_penalty(x.data, g_output.data)

    d_loss.backward()
    d_optimizer.step()

    return d_loss.data.item()

def g_train_wgan(x):
    gen_model.zero_grad()

    batch_size = x.size(0)
    input_z = data.create_noise(batch_size, z_size, mode_z).to(device)

    g_output = gen_model(input_z)

    d_generated = disc_model(g_output)
    g_loss = -d_generated.mean()

    # gradient backprop & optimize ONLY G's parameters
    g_loss.backward()
    g_optimizer.step()

    return g_loss.data.item()


## Train the discriminator
def d_train_wgan_text(x, input_t):

    #Init gradient
    disc_model.zero_grad()

    batch_size = x.size(0)
    x = x.to(device)

    #Discriminating Real Images
    d_real = disc_model(x, input_t)

    #Building Z
    input_z = data.create_noise(batch_size, z_size, mode_z).to(device)
    #Building Fake T
    fake_t = torch.zeros(batch_size, word_embedding_dim).to(device)

    #Generating Fake Images
    g_output = gen_model(input_z, fake_t)

    #Discriminating Fake Images
    d_generated = disc_model(g_output, input_t)

    d_loss = d_generated.mean() - d_real.mean() + gradient_penalty_text(x.data, g_output.data, input_t)

    d_loss.backward()
    d_optimizer.step()

    return d_loss.data.item()

def g_train_wgan_text(x):

    #Init gradient
    gen_model.zero_grad()

    #Building Z
    batch_size = x.size(0)
    input_z = data.create_noise(batch_size, z_size, mode_z).to(device)

    #Building Fake T
    fake_t = torch.zeros(batch_size, word_embedding_dim).to(device)

    #Generating Fake Images
    g_output = gen_model(input_z, fake_t)

    #Discriminating Fake Images
    d_generated = disc_model(g_output, fake_t)
    g_loss = -d_generated.mean()
    #print("G Loss:", g_loss)

    # gradient backprop & optimize ONLY G's parameters
    g_loss.backward()
    g_optimizer.step()

    return g_loss.data.item()



## Train the discriminator
def d_train_ACGAN(x, T):

    #init gradient
    disc_model.zero_grad()

    # Train discriminator with a real batch
    batch_size = x.size(0)
    x = x.to(device)
    d_proba_real, real_aux = disc_model(x)

    #Calculate loss (real images)
    d_labels_real = torch.ones(batch_size, 1, device=device)
    #d_loss_real = adversarial_loss(d_proba_real, d_labels_real)

    real_aux = real_aux.squeeze(dim=1)
    d_real_loss = (adversarial_loss(d_proba_real, d_labels_real) + auxiliary_loss(real_aux, T)) / 2

    #building Z
    input_z = data.create_noise(batch_size, z_size, mode_z).to(device)

    #building fake T
    gen_labels = torch.randint(0, n_classes, (batch_size,))

    # Train discriminator on a fake batch
    g_output = gen_model(input_z, gen_labels)

    #Discriminating fake images
    d_proba_fake, fake_aux = disc_model(g_output)

    #Calculate loss (fake images)
    d_labels_fake = torch.zeros(batch_size, 1, device=device)

    gen_labels = gen_labels.type(torch.float32).to(device)
    fake_aux = fake_aux.squeeze(dim=1)

    d_fake_loss = (adversarial_loss(d_proba_fake, d_labels_fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

    # gradient backprop & optimize ONLY D's parameters
    d_loss = (d_real_loss + d_fake_loss) / 2
    d_loss.backward()
    d_optimizer.step()

    return d_loss.data.item(), d_proba_real.detach(), d_proba_fake.detach()

## Train the generator
def g_train_ACGAN(x):

    #Init gradient
    gen_model.zero_grad()

    #Building Z
    batch_size = x.size(0)
    input_z = data.create_noise(batch_size, z_size, mode_z).to(device)

    #building real labels
    g_labels_real = torch.ones((batch_size, 1), device=device)

    #generate fake text embeddings
    gen_labels = torch.randint(0, n_classes, (batch_size,))

    g_output = gen_model(input_z, gen_labels)

    #in this case, d_proba_fake is their validity
    d_proba_fake, pred_label = disc_model(g_output)

    pred_label = pred_label.squeeze(dim=1)

    gen_labels = gen_labels.type(torch.float32).to(device)
    g_loss = 0.5 * (adversarial_loss(d_proba_fake, g_labels_real) + auxiliary_loss(pred_label, gen_labels))

    # gradient backprop & optimize ONLY G's parameters
    g_loss.backward()
    g_optimizer.step()

    return g_loss.data.item()



if __name__ == '__main__':
    print(torch.__version__)
    print("GPU Available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = "cpu"

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, help='Select model by prefix: DCGAN_v1, DCGAN_v2, WGAN_v1, WGAN_v2, ACGAN_v1, ACGAN_v2, DCGAN_v1_Text, DCGAN_v2_Text, WGAN_v1_Text, WGAN_v2_Text')
    parser.add_argument('--Generation_Size', type=str, help='Word_Level, Contextual, Sentence_Level')

    args = parser.parse_args()
    model = args.model
    Generation_Size = args.Generation_Size

    batch_size = 64

    if Generation_Size == "Word_Level":
        word_embedding_dim = 50
    elif Generation_Size == "Contextual":
        word_embedding_dim = 150
    #output_shape = (1, 105, 8)
    torch.manual_seed(1)
    np.random.seed(1)
    z_size = 100
    #image_size = (105, 8)
    n_filters = 32
    n_classes = 5860
    g_learning_rate = 0.00002
    d_learning_rate = 0.00002

    # Create the data object
    data = Data.Data()

    # To load the lists from the file:
    if Generation_Size == "Word_Level":
        with open("/users/gxb18167/Datasets/ZuCo/EEG_Text_Pairs.pkl",
                  'rb') as file:
            EEG_word_level_embeddings = pickle.load(file)
            EEG_word_level_labels = pickle.load(file)
    else:
        with open("/users/gxb18167/Datasets/ZuCo/EEG_Text_Pairs_Sentence.pkl",
                  'rb') as file:
            EEG_word_level_embeddings = pickle.load(file)
            EEG_word_level_labels = pickle.load(file)

    if Generation_Size == "Word_Level":
        Embedded_Word_labels, word_embeddings = data.create_word_label_embeddings(EEG_word_level_labels, word_embedding_dim=word_embedding_dim)
        trainloader = data.create_dataloader(EEG_word_level_embeddings, Embedded_Word_labels)
    elif Generation_Size == "Contextual":
        Embedded_Word_labels, word_embeddings = data.create_word_label_embeddings_contextual(EEG_word_level_labels, word_embedding_dim=word_embedding_dim)
        trainloader = data.create_dataloader(EEG_word_level_embeddings, Embedded_Word_labels)
    elif Generation_Size == "Sentence_Level":
        EEG_sentence_list, list_of_sentences = data.create_word_label_embeddings_sentence(EEG_word_level_embeddings, EEG_word_level_labels, word_embedding_dim=word_embedding_dim)
        trainloader = data.create_dataloader_sentence(EEG_sentence_list, list_of_sentences)
    elif model == "ACGAN_v1" or model == "ACGAN_v2":
        encoder = LabelEncoder()
        Embedded_Word_labels = encoder.fit_transform(np.array(EEG_word_level_labels).reshape(-1, 1))
        Embedded_Word_labels = torch.tensor(Embedded_Word_labels, dtype=torch.float)
        trainloader = data.create_dataloader(EEG_word_level_embeddings, Embedded_Word_labels)

    mode_z = 'uniform'
    fixed_z = data.create_noise(batch_size, z_size, mode_z).to(device)
    noise = data.create_noise(batch_size, z_size, "uniform")

    if model == "DCGAN_v1":
        gen_model = Networks.GeneratorDCGAN_v1(z_size).to(device)
        disc_model = Networks.DiscriminatorDCGAN_v1(n_filters).to(device)
    elif model == "DCGAN_v2":
        gen_model = Networks.GeneratorDCGAN_v2(z_size).to(device)
        disc_model = Networks.DiscriminatorDCGAN_v2(n_filters).to(device)
    elif model == "WGAN_v1":
        gen_model = Networks.GeneratorWGAN_v1(z_size).to(device)
        disc_model = Networks.DiscriminatorWGAN_v1(n_filters).to(device)
    elif model == "WGAN_v2":
        gen_model = Networks.GeneratorWGAN_v2(z_size).to(device)
        disc_model = Networks.DiscriminatorWGAN_v2(n_filters).to(device)
    elif model == "ACGAN_v1":
        gen_model = Networks.GeneratorACGAN_v1(z_size).to(device)
        disc_model = Networks.DiscriminatorACGAN_v1(n_filters).to(device)
    elif model == "ACGAN_v2":
        gen_model = Networks.GeneratorACGAN_v2(z_size).to(device)
        disc_model = Networks.DiscriminatorACGAN_v1(n_filters).to(device)

    elif model == "DCGAN_v1_Text":
        if Generation_Size == "Sentence_Level":
            gen_model = Networks.GeneratorDCGAN_v1_Sentence(z_size, word_embedding_dim).to(device)
            disc_model = Networks.DiscriminatorDCGAN_v1_Sentence(n_filters, word_embedding_dim).to(device)
        else:
            gen_model = Networks.GeneratorDCGAN_v1_Text(z_size, word_embedding_dim).to(device)
            disc_model = Networks.DiscriminatorDCGAN_v1_Text(n_filters, word_embedding_dim).to(device)
    elif model == "DCGAN_v2_Text":
        if Generation_Size == "Sentence_Level":
            gen_model = Networks.GeneratorDCGAN_v2_Sentence(z_size, word_embedding_dim).to(device)
            disc_model = Networks.DiscriminatorDCGAN_v2_Sentence(n_filters, word_embedding_dim).to(device)
        else:
            gen_model = Networks.GeneratorDCGAN_v2_Text(z_size, word_embedding_dim).to(device)
            disc_model = Networks.DiscriminatorDCGAN_v2_Text(n_filters, word_embedding_dim).to(device)
    elif model == "WGAN_v1_Text":
        if Generation_Size == "Sentence_Level":
            gen_model = Networks.GeneratorWGAN_v1_Sentence(z_size, word_embedding_dim).to(device)
            disc_model = Networks.DiscriminatorWGAN_v1_Sentence(n_filters, word_embedding_dim).to(device)
        else:
            gen_model = Networks.GeneratorWGAN_v1_Text(z_size, word_embedding_dim).to(device)
            disc_model = Networks.DiscriminatorWGAN_v1_Text(n_filters, word_embedding_dim).to(device)
    elif model == "WGAN_v2_Text":
        if Generation_Size == "Sentence_Level":
            gen_model = Networks.GeneratorWGAN_v2_Sentence(z_size, word_embedding_dim).to(device)
            disc_model = Networks.DiscriminatorWGAN_v2_Sentence(n_filters, word_embedding_dim).to(device)
        else:
            gen_model = Networks.GeneratorWGAN_v2_Text(z_size, word_embedding_dim).to(device)
            disc_model = Networks.DiscriminatorWGAN_v2_Text(n_filters, word_embedding_dim).to(device)


    loss_fn = nn.BCELoss()
    auxiliary_loss = nn.CrossEntropyLoss()
    adversarial_loss = nn.BCELoss()

    g_optimizer = torch.optim.Adam(gen_model.parameters(), g_learning_rate)
    d_optimizer = torch.optim.Adam(disc_model.parameters(), d_learning_rate)

    num_epochs = 100
    torch.manual_seed(1)
    critic_iterations = 5
    save_interval = 5

    model_parameters = f"Generation_size_{Generation_Size}_batch_size_{batch_size}_g_d_learning_rate{g_learning_rate}_{d_learning_rate}_word_embedding_dim_{word_embedding_dim}_z_size_{z_size}_num_epochs_{num_epochs}_device_{device}_"

    model_folder_path = f'/users/gxb18167/Datasets/Checkpoints/{model}'
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    checkpoint_path = model_folder_path+'/'+model_parameters+'checkpoint_epoch_{}.pt'
    final_model_path = model_folder_path+'/'+model_parameters+'model_final.pt'


    if model == "DCGAN_v1" or model == "DCGAN_v2" or model == "DCGAN_v1_Text" or model == "DCGAN_v2_Text":
        for epoch in range(1, num_epochs + 1):
            gen_model.train()
            d_losses, g_losses = [], []
            for i, (x, t) in enumerate(trainloader):
                if model == "DCGAN_v1" or model == "DCGAN_v2":
                    d_loss, d_proba_real, d_proba_fake = d_train(x)
                    d_losses.append(d_loss)
                    g_losses.append(g_train(x))
                else:
                    d_loss, d_proba_real, d_proba_fake = d_train_text(x, t)
                    d_losses.append(d_loss)
                    g_losses.append(g_train_text(x, t))
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
        # Save the final model after training is complete
        torch.save({
            'epoch': num_epochs,
            'gen_model_state_dict': gen_model.state_dict(),
            'optimizer_state_dict': g_optimizer.state_dict(),
            'd_losses': d_losses,
            'g_losses': g_losses,
        }, final_model_path)


    elif model == "WGAN_v1" or model == "WGAN_v2" or model == "WGAN_v1_Text" or model == "WGAN_v2_Text":
        for epoch in range(1, num_epochs + 1):
            gen_model.train()
            d_losses, g_losses = [], []
            for i, (x, t) in enumerate(trainloader):
                for _ in range(critic_iterations):
                    if model == "WGAN_v1" or model == "WGAN_v2":
                        d_loss = d_train_wgan(x)
                        d_losses.append(d_loss)
                    else:
                        d_loss = d_train_wgan_text(x, t)
                        d_losses.append(d_loss)
                if model == "WGAN_v1" or model == "WGAN_v2":
                    g_losses.append(g_train_wgan(x))
                else:
                    g_losses.append(g_train_wgan_text(x))

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
        # Save the final model after training is complete
        torch.save({
            'epoch': num_epochs,
            'gen_model_state_dict': gen_model.state_dict(),
            'optimizer_state_dict': g_optimizer.state_dict(),
            'd_losses': d_losses,
            'g_losses': g_losses,
        }, final_model_path)

    elif model == "ACGAN_v1" or model == "ACGAN_v2":
        for epoch in range(1, num_epochs + 1):
            gen_model.train()
            fixed_z = data.create_noise(batch_size, z_size, mode_z).to(device)
            d_losses, g_losses = [], []
            for i, (x, t) in enumerate(trainloader):
                g_losses.append(g_train_ACGAN(x))
                d_loss, d_proba_real, d_proba_fake = d_train_ACGAN(x, t)
                print("D Loss:", d_loss)
                d_losses.append(d_loss)

            print(f'Epoch {epoch:03d} | D Loss >>'
                  f' {torch.FloatTensor(d_losses).mean():.4f}')
            print(f'Epoch {epoch:03d} | G Loss >>'
                  f' {torch.FloatTensor(g_losses).mean():.4f}')

            if epoch % save_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'gen_model_state_dict': gen_model.state_dict(),
                    'optimizer_state_dict': g_optimizer.state_dict(),
                    'd_losses': d_losses,
                    'g_losses': g_losses,
                }, checkpoint_path.format(epoch))

