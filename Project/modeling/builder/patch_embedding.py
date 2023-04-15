# *************************************************************************** #
#                                                                              #
#    patch_embedding.py                                                        #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2023/04/15 12:00:51 by Widium                                    #
#    Updated: 2023/04/15 12:00:51 by Widium                                    #
#                                                                              #
# **************************************************************************** #

from torch import Tensor
from torch.nn import Module
from torch.nn import Linear

class PatchTokenEmbedding(Module):
    """
    Linear Projection on color channels value of tokenized patches

    Args:
        `color_channels` (int): nbr of color channels of tokenized patches
        `embedding_size` (int): size of fixed-lenght vector for represent tokenized patches 
    """
    def __init__(self, color_channels : int, embedding_size : int):
        
        super().__init__()
        
        self.embedding = Linear(
            in_features=color_channels, 
            out_features=embedding_size
        )
    
    def forward(self, x : Tensor)->Tensor:
        """
        Apply linear projection on color channels value

        Args:
            `x` (Tensor): patches tokenized with shape (batch_size, nbr_tokens, color)

        Returns:
            Tensor: embedding tokens with shape (batch_size, nbr_tokens, embedding_size
        """
        token_embedding = self.embedding(x)
        
        return (token_embedding) 