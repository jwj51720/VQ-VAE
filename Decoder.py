import torch.nn as nn

class Decoder(nn.Module):
    """
    Decoder: p(x|z): z_q -> x_hat
    - input_channel: the channel of image data (ex. R,G,B -> 3)
    - hidden_channel: the channel of hidden variable (hidden unit = 256)
    """
    def __init__(self, input_channel, hidden_channel, B=128):
        super(Decoder, self).__init__()
        self.b = B  # Batch size
        self.transposed_conv1 = nn.ConvTranspose2d(hidden_channel, hidden_channel, kernel_size=4, stride=2, padding=1)
        self.transposed_conv2 = nn.ConvTranspose2d(hidden_channel, input_channel, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channel, hidden_channel, kernel_size=1)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(hidden_channel)
        self.residual_block = nn.Sequential(
            self.relu, self.conv1, self.batch_norm, self.relu, self.conv2, self.batch_norm
        )

    def forward(self, input): # [128, 256, 32, 32]
        output = output + self.residual_block(input)  # [128, 256, 32, 32]
        output = output + self.residual_block(input)  # [128, 256, 32, 32]
        output = self.relu(output)
        output = self.transposed_conv1(output)  # [128, 256, 64, 64]
        output = self.relu(output)
        output = self.transposed_conv2(output) # [128, 3, 128, 128]
        output = self.relu(output)
        return output