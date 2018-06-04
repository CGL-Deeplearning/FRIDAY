import torch
import random
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from modules.models.inception import Inception3


class Attention(nn.Module):
    r"""
    Applies an attention mechanism on the output features from the decoder.
    Args:
        dim(int): The number of expected features in the output
    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.
    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.
    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.
    """
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked
        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)

        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn


class EncoderCNN(nn.Module):
    def __init__(self, image_channels):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        self.inception_alt1 = Inception3(image_channels)
        self.inception_alt2 = Inception3(image_channels)
        self.bn = nn.BatchNorm2d(736, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        allele_1_image = images[:, 0:8, :, :]
        allele_2_image = torch.cat((images[:, 0:6, :, :], images[:, 7:9, :, :]), dim=1)
        features_alt1 = self.bn(self.inception_alt1(allele_1_image))
        features_alt2 = self.bn(self.inception_alt2(allele_2_image))
        return features_alt1, features_alt2


class EncoderCRNN(nn.Module):
    def __init__(self, image_channels, hidden_size, seq_len=1, bidirectional=True):
        super(EncoderCRNN, self).__init__()
        self.cnn_encoder = EncoderCNN(image_channels)
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = 1
        self.gru_alt1 = nn.GRU(736 * 5, hidden_size, num_layers=self.num_layers, bidirectional=bidirectional,
                               batch_first=True)
        self.gru_alt2 = nn.GRU(736 * 5, hidden_size, num_layers=self.num_layers, bidirectional=bidirectional,
                               batch_first=True)

    def forward(self, x):
        features_alt1, features_alt2 = self.cnn_encoder(x)
        batch_size = features_alt1.size(0)
        seq_len = features_alt1.size(2)
        features_alt1 = features_alt1.view(batch_size, seq_len, -1)
        features_alt2 = features_alt2.view(batch_size, seq_len, -1)

        self.gru_alt1.flatten_parameters()
        self.gru_alt2.flatten_parameters()
        alt1_x, hidden_alt1 = self.gru_alt1(features_alt1)
        alt2_x, hidden_alt2 = self.gru_alt2(features_alt2)

        if self.bidirectional:
            alt1_x = alt1_x.contiguous()
            alt2_x = alt2_x.contiguous()
            alt1_x = alt1_x.view(alt1_x.size(0), alt1_x.size(1), 2, -1).sum(2).view(alt1_x.size(0), alt1_x.size(1), -1)
            alt2_x = alt2_x.view(alt2_x.size(0), alt2_x.size(1), 2, -1).sum(2).view(alt2_x.size(0), alt2_x.size(1), -1)
            # (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
            # hidden_alt1 = torch.cat([hidden_alt1[0:hidden_alt1.size(0):2], hidden_alt1[1:hidden_alt1.size(0):2]], 2)
            # hidden_alt2 = torch.cat([hidden_alt2[0:hidden_alt2.size(0):2], hidden_alt2[1:hidden_alt2.size(0):2]], 2)

        encoder_output = torch.cat((alt1_x, alt2_x), dim=2).contiguous()
        encoder_hidden = torch.cat((hidden_alt1, hidden_alt2), dim=2).contiguous()
        return encoder_output, encoder_hidden

    def init_hidden(self, batch_size, num_layers=3, num_directions=2):
        return torch.zeros(batch_size, num_directions * num_layers, self.hidden_size)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, num_classes, max_length, dropout_p=0.1, bidirectional=True):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(self.num_classes, self.hidden_size)
        self.attention = Attention(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.out = nn.Linear(self.hidden_size, self.num_classes)

    def forward_step(self, x, hidden, encoder_outputs):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        output, hidden = self.gru(embedded, hidden)
        if self.bidirectional:
            output = output.contiguous()
            output = output.view(output.size(0), output.size(1), 2, -1).sum(2).view(output.size(0), output.size(1), -1)

        output, attn = self.attention(output, encoder_outputs)

        logits = self.out(output.contiguous().view(-1, self.hidden_size))

        return logits, hidden, attn

    def forward(self, inputs, encoder_hidden, encoder_outputs):
        print("In Forward: ", inputs.size(), encoder_hidden.size(), encoder_outputs.size())
        encoder_hidden = encoder_hidden.transpose(0, 1)

        decoder_input = inputs[:, 0].unsqueeze(1)
        decoder_output, decoder_hidden, attn = self.forward_step(decoder_input, encoder_hidden, encoder_outputs)

        return decoder_output, decoder_hidden
