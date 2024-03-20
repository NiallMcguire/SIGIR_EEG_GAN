import sys
import nltk
import torch.nn as nn
import numpy as np
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


if __name__ == '__main__':
    print(torch.__version__)
    print("GPU Available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = "cpu"

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, help='Select model by prefix: DCGAN_v1, ')

    args = parser.parse_args()
    model = args.model

    batch_size = 64
    word_embedding_dim = 50
    #output_shape = (1, 105, 8)
    torch.manual_seed(1)
    np.random.seed(1)
    z_size = 100
    #image_size = (105, 8)
    n_filters = 32

    # Create the data object
    data = Data.Data()

    # To load the lists from the file:
    with open("/users/gxb18167/Datasets/ZuCo/EEG_Text_Pairs.pkl",
              'rb') as file:
        EEG_word_level_embeddings = pickle.load(file)
        EEG_word_level_labels = pickle.load(file)

    if model == "DCGAN_v1" or model == "DCGAN_v2" or model == "WGAN_v1" or model == "WGAN_v2":
        Embedded_Word_labels, word_embeddings = data.create_word_label_embeddings(EEG_word_level_labels, word_embedding_dim=word_embedding_dim)
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

    loss_fn = nn.BCELoss()

    g_optimizer = torch.optim.Adam(gen_model.parameters(), 0.00002)
    d_optimizer = torch.optim.Adam(disc_model.parameters(), 0.00002)

    epoch_samples_wgan = []
    num_epochs = 100
    torch.manual_seed(1)
    critic_iterations = 5
    save_interval = 5


    model_folder_path = f'/users/gxb18167/Datasets/Checkpoints/{model}'
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    checkpoint_path = model_folder_path+'/checkpoint_epoch_{}.pt'
    final_model_path = model_folder_path+'/model_final.pt'


    if model == "DCGAN_v1" or model == "DCGAN_v2":
        for epoch in range(1, num_epochs + 1):
            gen_model.train()
            d_losses, g_losses = [], []
            for i, (x, _) in enumerate(trainloader):
                d_loss, d_proba_real, d_proba_fake = d_train(x)
                print("D Loss:", d_loss)
                d_losses.append(d_loss)
                g_losses.append(g_train(x))

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

