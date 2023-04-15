# *************************************************************************** #
#                                                                              #
#    patch_tokenization.py                                                     #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2023/04/15 12:00:04 by Widium                                    #
#    Updated: 2023/04/15 12:00:04 by Widium                                    #
#                                                                              #
# **************************************************************************** #

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import Flatten

class PatchTokenizer(Module): 
    """
    Convert 3D patches Tensor to sequence of patches with extracted pixels 
    Flatten and Reshape input tensor to create a sequence of feature vectors
    """
    def __init__(self):
        
        super().__init__()
        
        self.flatten = Flatten(start_dim=2, end_dim=3)
    
    def forward(self, x : Tensor)->Tensor:
        """
        Flatten spatial dimension in input tensor (batch_size, color, height * width)
        reshape it to sequence of feature vector (nbr_tokens, color)

        Args:
            `x` (Tensor): extracted pixels on patches (batch_size, color, height, width)

        Returns:
            Tensor: sequence of feature vectors (batch_size, nbr_tokens, color)
        """
        batch_size = x.shape[0]
        color = x.shape[1]
        number_patches = x.shape[2] * x.shape[3] 
        
        patches_flatten = self.flatten(x)
        
        patches_tokenized = torch.reshape(
            input=patches_flatten,
            shape=(batch_size, number_patches, color)
        ) 
        
        return (patches_tokenized)