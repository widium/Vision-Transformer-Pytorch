# *************************************************************************** #
#                                                                              #
#    mlp.py                                                                    #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2023/04/17 08:32:03 by Widium                                    #
#    Updated: 2023/04/17 08:32:03 by Widium                                    #
#                                                                              #
# **************************************************************************** #

from torch import Tensor
from torch.nn import Module
from torch.nn import Linear
from torch.nn import GELU
from torch.nn import Dropout
from torch.nn import LayerNorm

from .residual import ResidualConnection

class MultiLayerPerceptronBlock(Module):  
    """
    A Multi-Layer Perceptron (MLP) block with residual connections, used in the Transformer Encoder.
    
    The MLP block consists of two fully connected layers with a GELU activation function in between.
    Layer normalization is applied on input features before the block, 
    Dropout is applied after each feature creation (like Linear Layer)
    Residual connections are used on output of the second Layer with input tensor
    
    Args:
        `embedding_size` (int, optional): Input and output tensor size. Default is 768.
        `units` (int, optional): Number of hidden units in the intermediate layer. Default is 3072.
        `dropout_rate` (float, optional): Dropout probability. Default is 0.1.
        
    Returns:
        Tensor: Output tensor with the same shape as the input tensor (batch_size, nbr_tokens, embedding_size).
    """
    def __init__(
        self, 
        embedding_size: int = 768, 
        units: int = 3072, 
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.layer_norm = LayerNorm(normalized_shape=embedding_size)
        self.gelu = GELU()
        self.dropout = Dropout(p=dropout_rate)
        
        self.fc1 = Linear(
            in_features=embedding_size,
            out_features=units
        )
        
        self.fc2 = Linear(
            in_features=units,
            out_features=embedding_size
        )
        
        self.residual_connection = ResidualConnection()
        
    def forward(self, x: Tensor) -> Tensor:
        
        x_norm = self.layer_norm(x)
        
        hidden = self.fc1(x_norm)
        hidden = self.gelu(hidden)
        hidden = self.dropout(hidden)
        
        output = self.fc2(hidden)
        output = self.dropout(output)
        
        residual_output = self.residual_connection(input=x, output=output)
        
        return (residual_output)
        
