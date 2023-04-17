# *************************************************************************** #
#                                                                              #
#    attention.py                                                              #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2023/04/17 08:30:50 by Widium                                    #
#    Updated: 2023/04/17 08:30:50 by Widium                                    #
#                                                                              #
# **************************************************************************** #

from torch import Tensor
from torch.nn import Module
from torch.nn import LayerNorm
from torch.nn import MultiheadAttention

from .residual import ResidualConnection

class MultiHeadSelfAttentionBlock(Module):
    """
    Normalize input feature and apply multi-head self-attention followed by a residual connection.
    The input and output tensor shapes are expected to be identical.

    Attributes:
        `layer_norm` (LayerNorm): Layer normalization applied to the input tensor.
        `multi_head_attention` (MultiheadAttention): Multi-head self-attention mechanism.

    Args:
        `embedding_size` (int, optional): the input tensor's last dimension, which corresponds
            to the number of features or embedding dimensions. Default is 768.
        `num_heads` (int, optional): The number of attention heads in the multi-head self-attention mechanism.
            Default is 12.
        `dropout_rate` (float, optional): The dropout rate on attention.
            Default is 0.1.

    Returns:
        Tensor: The output tensor after applying multi-head self-attention and the residual connection.
    """
    def __init__(
        self, 
        embedding_size : int = 768, 
        nbr_heads : int = 12, 
        dropout_rate : float = 0.1,
    )->None:
        
        super().__init__()
        
        self.layer_norm = LayerNorm(normalized_shape=embedding_size)
        
        self.multi_head_attention = MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=nbr_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.residual_connection = ResidualConnection()
        
    
    def forward(self, x : Tensor)->Tensor:
        
        x_normalized = self.layer_norm(x)
        
        attention, _ = self.multi_head_attention(# duplicate the input tensor `x_normalized` for Self Attention
            query=x_normalized,
            key=x_normalized,
            value=x_normalized,
            need_weights=False
        )
        
        
        residual_output = self.residual_connection(input=x, output=attention)
        
        return (residual_output)

