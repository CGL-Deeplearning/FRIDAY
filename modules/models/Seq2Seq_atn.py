import torch
import torch.nn.functional as F
import torch.nn as nn
from modules.models.inception import Inception3


class EncoderCNN(nn.Module):
    def __init__(self, image_channels, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        self.inception = Inception3(image_channels)
        self.linear = nn.Linear(self.inception.in_features, embed_size)

    def forward(self, images):
        """Extract feature vectors from input images."""
        features = self.inception(images)
        features = self.linear(features)
        return features


class EncoderCRNN(nn.Module):
    def __init__(self, image_channels, hidden_size):
        super(EncoderCRNN, self).__init__()
        self.cnn_encoder = EncoderCNN(image_channels, hidden_size)
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, 6)
        # self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, x):
        features = self.cnn_encoder(x)
        output = self.linear(features)
        # output = features.view(1, features.size(0), -1)

        # output, hidden = self.gru(output, hidden)
        return output

    def init_hidden(self, batch_size, num_layers=1, num_directions=1):
        return torch.zeros(num_directions * num_layers, batch_size, self.hidden_size)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, num_classes, max_length, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.num_classes, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x, hidden, encoder_outputs):
        embedded = self.embedding(x)

        embedded = self.dropout(embedded)
        encoder_outputs = encoder_outputs.view(encoder_outputs.size(1), -1)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs.unsqueeze(1))
        attn_applied = attn_applied.transpose(0, 1)

        output = torch.cat((embedded[0], attn_applied[0]), 1)

        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)

        output, hidden = self.gru(output, hidden)

        output = self.out(output[0])
        return output, hidden, attn_weights