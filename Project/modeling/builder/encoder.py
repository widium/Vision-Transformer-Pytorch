# *************************************************************************** #
#                                                                              #
#    encoder.py                                                                #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2023/04/17 08:33:30 by Widium                                    #
#    Updated: 2023/04/17 08:33:30 by Widium                                    #
#                                                                              #
# **************************************************************************** #

from torch import Tensor
from torch.nn import Module
from torch.nn import Sequential

from .attention import MultiHeadSelfAttentionBlock
from .mlp import MultiLayerPerceptronBlock

# **************************************************************************** #

class TransformerEncoder(Module):  
    """
    Create a Sequential Module of stacked `TransformerEncoderBlock` with same parameters.

    Args:
        `embedding_size` (int, optional): The size of the input embeddings. Default is 768.
        `nbr_heads` (int, optional): The number of attention heads in the `MultiHeadSelfAttentionBlock`. Default is 12.
        `dropout_attention` (float, optional): The dropout rate for the `MultiHeadSelfAttentionBlock`. Default is 0.0.
        `mlp_units` (int, optional): The number of units in the `MultiLayerPerceptronBlock`. Default is 3072.
        `dropout_mlp` (float, optional): The dropout rate for the `MultiLayerPerceptronBlock`. Default is 0.1.
    
    Forward method input:
        x (Tensor): a sequence of Tokens Embedding like [batch_size, nbr_tokens, embedding_size].
    
    Return
        Tensor: a tensor with same input size after passing through the stack of `TransformerEncoderBlock`.
    """
    def __init__(
        self, 
        nbr_encoder_blocks : int = 12,
        embedding_size : int = 768, 
        nbr_heads : int = 12,
        dropout_attention : float = 0.0,
        mlp_units : int = 3072,
        dropout_mlp : float = 0.1,
    )->None:
        
        super().__init__()
        
        self.encoder_blocks = Sequential()
        
        for _ in range(nbr_encoder_blocks):
            
            block = TransformerEncoderBlock(
                embedding_size=embedding_size,
                nbr_heads=nbr_heads,
                dropout_attention=dropout_attention,
                mlp_units=mlp_units,
                dropout_mlp=dropout_mlp
            )
            
            self.encoder_blocks.append(module=block)
        
    def forward(self, x : Tensor)->Tensor:
        
        output = self.encoder_blocks(x)
        
        return (output)
        
# **************************************************************************** #        
        
class TransformerEncoderBlock(Module):   
    """
    building block that combines multi-head self-attention and a multi-layer perceptron, 
    applying residual connections and layer normalization. 

    Args:
        `embedding_size` (int, optional): The size of the input embeddings. Default is 768.
        `nbr_heads` (int, optional): The number of attention heads in the `MultiHeadSelfAttentionBlock`. Default is 12.
        `dropout_attention` (float, optional): The dropout rate for the `MultiHeadSelfAttentionBlock`. Default is 0.0.
        `mlp_units` (int, optional): The number of units in the `MultiLayerPerceptronBlock`. Default is 3072.
        `dropout_mlp` (float, optional): The dropout rate for the `MultiLayerPerceptronBlock`. Default is 0.1.
    
    Forward method input:
        x (Tensor): a sequence of Tokens Embedding like [batch_size, nbr_tokens, embedding_size].
    
    Return
        Tensor: a tensor with same input size after passing through the `MultiHeadSelfAttentionBlock` and` MultiLayerPerceptronBlock`.
    """
    def __init__(
        self, 
        embedding_size : int = 768, 
        nbr_heads : int = 12,
        dropout_attention : float = 0.0,
        mlp_units : int = 3072,
        dropout_mlp : float = 0.1,
    )->None:
        
        super().__init__()
        
        self.attention_block = MultiHeadSelfAttentionBlock(
            embedding_size=embedding_size,
            nbr_heads=nbr_heads,
            dropout_rate=dropout_attention
        )
        
        self.mlp_block = MultiLayerPerceptronBlock(
            embedding_size=embedding_size,
            units=mlp_units,
            dropout_rate=dropout_mlp,
        )
        
    
    def forward(self, x : Tensor)->Tensor:
        
        attention = self.attention_block(x)
        new_features = self.mlp_block(attention)
    
        return (new_features)