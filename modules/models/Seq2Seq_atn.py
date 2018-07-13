import torch
import torch.nn.functional as F
import torch.nn as nn
from modules.models.resnet import resnet18_custom


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
        self.resnet = resnet18_custom(image_channels)

    def forward(self, images):
        """Extract feature vectors from input images."""
        features = self.resnet(images)

        return features


class EncoderCRNN(nn.Module):
    def __init__(self, image_channels, hidden_size, bidirectional=True):
        super(EncoderCRNN, self).__init__()
        self.cnn_encoder = EncoderCNN(image_channels)
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = 1
        self.gru = nn.GRU(2048, hidden_size, num_layers=self.num_layers, bidirectional=bidirectional, batch_first=True)

    def forward(self, x, hidden):
        hidden = hidden.transpose(0, 1).contiguous()

        features_cnn = self.cnn_encoder(x)

        batch_size = features_cnn.size(0)
        seq_len = features_cnn.size(2)
        features_cnn = features_cnn.view(batch_size, seq_len, -1)

        # self.gru.flatten_parameters()
        output_rnn, hidden_rnn = self.gru(features_cnn, hidden)

        if self.bidirectional:
            output_rnn = output_rnn.contiguous()
            output_rnn = output_rnn.view(output_rnn.size(0), output_rnn.size(1), 2, -1)\
                .sum(2).view(output_rnn.size(0), output_rnn.size(1), -1)

        hidden_rnn = hidden_rnn.transpose(0, 1).contiguous()

        return output_rnn, hidden_rnn

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

    def forward_step(self, x, encoder_output, encoder_hidden):
        embedded = self.embedding(x).view(x.size(0), 1, -1)
        embedded = self.dropout(embedded)

        # self.gru.flatten_parameters()
        output_gru, hidden_gru = self.gru(embedded, encoder_hidden)

        if self.bidirectional:
            output_gru = output_gru.contiguous()
            output_gru = output_gru.view(output_gru.size(0), output_gru.size(1), 2, -1).sum(2)\
                .view(output_gru.size(0), output_gru.size(1), -1)

        output, attn = self.attention(output_gru, encoder_output)

        class_probabilities = self.out(output.contiguous().view(-1, self.hidden_size))

        return class_probabilities, hidden_gru, attn

    def forward(self, decoder_input, encoder_output, encoder_hidden):
        encoder_hidden = encoder_hidden.transpose(0, 1).contiguous()
        class_probabilities, hidden, attn = self.forward_step(decoder_input, encoder_output, encoder_hidden)

        hidden = hidden.transpose(0, 1).contiguous()

        return class_probabilities, hidden, attn
