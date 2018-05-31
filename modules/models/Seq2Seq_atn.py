import torch
import torch.nn.functional as F
import torch.nn as nn
from modules.models.inception import Inception3


class EncoderCNN(nn.Module):
    def __init__(self, image_channels, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        self.inception_alt1 = Inception3(image_channels)
        self.inception_alt2 = Inception3(image_channels)
        self.linear_alt1 = nn.Linear(self.inception_alt1.in_features, embed_size)
        self.linear_alt2 = nn.Linear(self.inception_alt2.in_features, embed_size)

    def forward(self, images):
        """Extract feature vectors from input images."""
        allele_1_image = images[:, 0:8, :, :]
        allele_2_image = torch.cat((images[:, 0:6, :, :], images[:, 7:9, :, :]), dim=1)
        features_alt1 = self.inception_alt1(allele_1_image)
        features_alt1 = self.linear_alt1(features_alt1)
        features_alt2 = self.inception_alt2(allele_2_image)
        features_alt2 = self.linear_alt2(features_alt2)
        features = torch.cat((features_alt1, features_alt2), dim=1)
        return features


class EncoderCRNN(nn.Module):
    def __init__(self, image_channels, hidden_size):
        super(EncoderCRNN, self).__init__()
        # self.cnn_encoder = EncoderCNN(image_channels, hidden_size)
        self.hidden_size = hidden_size
        self.input_size = image_channels * 100
        self.linear = nn.Linear(21 * hidden_size * 4, 512)
        self.classify = nn.Linear(512, 6)
        self.gru_alt1 = nn.GRU(self.input_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.gru_alt2 = nn.GRU(self.input_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)

    def forward(self, x, hidden_a, hidden_b):
        # features = self.cnn_encoder(x)
        # output = self.linear(features)
        # output = features.view(1, features.size(0), -1)
        print("IMAGE SIZE IN FORWARD: ", x.size())
        batch_size = x.size(0)
        allele_1_image = x[:, 0:8, :, :].contiguous()
        allele_1_image = allele_1_image.view(batch_size, allele_1_image.size(2), -1)
        print(allele_1_image.size())
        allele_2_image = torch.cat((x[:, 0:6, :, :], x[:, 7:9, :, :]), dim=1)
        allele_2_image = allele_2_image.view(batch_size, allele_2_image.size(2), -1)
        print(allele_2_image.size())
        alt1_x, hidden_alt1 = self.gru_alt1(allele_1_image, hidden_a)
        alt2_x, hidden_alt2 = self.gru_alt2(allele_2_image, hidden_b)
        print(alt1_x.size(), alt2_x.size())
        combined_logits = torch.cat((alt1_x, alt2_x), dim=2)
        print(combined_logits.size())
        features = self.linear(combined_logits.view(batch_size, -1))
        print(features.size())
        logits = self.classify(features)
        print(logits.size())
        # output, hidden = self.gru(output, hidden)
        return logits

    def init_hidden(self, batch_size, num_layers=1, num_directions=2):
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
