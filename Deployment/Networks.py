import torch.nn as nn

class GeneratorDCGAN_v1(nn.Module):
    """
    This class is the generator for the GAN
    """

    def __init__(self, noise_dim):
        """
        The constructor for the generator

        :param noise_dim: The latent input size
        """
        super(GeneratorDCGAN_v1, self).__init__()

        self.noise_dim = noise_dim
        self.fc_noise = nn.Linear(noise_dim, 105 * 8)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, noise):
        """
        This function is the forward pass for the generator
        :param noise: The noise for the generator
        :return: Returns the synthetic EEG data
        """

        # Process noise
        noise = self.fc_noise(noise)
        noise = noise.view(noise.size(0), 1, 105, 8)

        z = self.conv1(noise)
        z = self.bn1(z)
        z = self.relu(z)
        z = self.conv2(z)

        return z

class DiscriminatorDCGAN_v1(nn.Module):
    """
    The class for the discriminator
    """

    def __init__(self, n_filters):
        """
        The constructor for the discriminator
        :param n_filters:
        """
        super(DiscriminatorDCGAN_v1, self).__init__()
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
        :param input:
        :return: returns the probability of the input being real or fake
        """
        output = self.network(input)
        return output



class GeneratorDCGAN_v2(nn.Module):
    """
    This class is the generator for the GAN
    """
    def __init__(self, noise_dim):
        """
        The constructor for the generator

        :param noise_dim: The latent input size
        """
        super(GeneratorDCGAN_v2, self).__init__()

        self.noise_dim = noise_dim

        # Define the layers of your generator
        self.fc_noise = nn.Linear(noise_dim, 105*8)  # Increase the size for more complexity
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, noise):
        """
        This function is the forward pass for the generator
        :param noise: The noise for the generator
        :return: Returns the synthetic EEG data
        """
        # Process noise
        noise = self.fc_noise(noise)
        noise = noise.view(noise.size(0), 1, 105,8)  # Adjust the size to match conv1

        # Upsample and generate the output
        z = self.conv1(noise)
        z = self.bn1(z)
        z = self.relu(z)

        z = self.conv2(z)
        z = self.bn2(z)
        z = self.relu(z)

        z = self.conv3(z)
        z = self.tanh(z)

        return z

class DiscriminatorDCGAN_v2(nn.Module):
    """
    The class for the discriminator
    """
    def __init__(self, n_filters):
        super(DiscriminatorDCGAN_v2, self).__init__()
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
        :param input:
        :return: returns the probability of the input being real or fake
        """
        output = self.network(input)
        return output


