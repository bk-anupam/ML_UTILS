import torch
import torch.nn as nn
from typing import Optional

# https://github.com/affjljoo3581/CommonLit-Readability-Prize/blob/c6b44a330e6cd37d9310e3ef0f530f9f429ffb05/src/modeling/miscellaneous.py
class AttentionHead(nn.Module):
    '''
     An attention-based classification head.
    This class is used to pool the hidden states from transformer model and computes the
    logits. In order to calculate the worthful representations from the transformer
    outputs, this class adopts time-based attention gating. Precisely, this class
    computes the importance of each word by using features and then apply
    weight-averaging to the features across the time axis.
    Since original classification models (i.e. `*ForSequenceClassification`) use simple
    feed-forward layers, the attention-based classification head can learn better
    generality.
    Args:
        hidden_size: The dimensionality of hidden units.
        num_labels: The number of labels to predict.
        dropout_prob: The dropout probability used in both attention and projection
            layers. Default is `0.1`.
    '''
    def __init__(self, hidden_size: int, dropout_prob: float = 0.1):    
        super().__init__()
        self.attention = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            # hidden_state which is the 3rd dimension in input tensor is condensed to a vector of size 1 from a vector of 
            # hidden_size
            nn.Linear(hidden_size, 1)
        )

    def forward(self, features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Calculate the attention scores 
        # features is hidden state from the last layer [batch_size, seq_len, hidden_size]
        attention_score = self.attention(features)
        # Apply the attention mask so that the features of the padding tokens are not attended to the representation.
        # At positions corresponding to padding tokens we subtract a big number from the corresponding
        # attention score so as to set the weight hidden state of that particular time step close to 0
        if attention_mask is not None:
            # attention_mask = [batch_size, seq_len]
            # for a specific input row attention_mask = [1, 1, 1, 0, 0 ....]
            # where 1 indicates that particular token is to be attended to and 0 otherwise            
            attention_mask = attention_mask.unsqueeze(-1)
            # attention_mask = [batch_size, seq_len, 1]
            attention_mask_value = (1 - attention_mask) * -10000.0
            # the positions with padding tokens would be set to -10000.0, while the rest would be 0.0
            attention_score += attention_mask_value
        # Calculate the weight of hidden state for each time step
        attention_weights = attention_score.softmax(dim=1)
        # attention_weights = [batch_size, seq_len, 1]
        # context_vector = weighted sum of input feature hidden states
        # * is element wise multiplication of two tensors, here we are multiplying element wise
        # hidden_states with attention weights along dim 1 (time steps) and taking the sum
        context_vector = torch.sum(attention_weights * features, dim=1)
        # The hidden states across the time steps (seq_length) are condensed into one single hidden state
        # of hidden_size, context_vector = [batch_size, hidden_size]
        return context_vector

