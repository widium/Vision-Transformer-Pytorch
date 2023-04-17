# *************************************************************************** #
#                                                                              #
#    residual.py                                                               #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2023/04/17 08:30:20 by Widium                                    #
#    Updated: 2023/04/17 08:30:20 by Widium                                    #
#                                                                              #
# **************************************************************************** #

import torch
from torch import Tensor
from torch.nn import Module

class ResidualConnection(Module):
    """
    Apply residual connection with element-wise addition between input tensor and output tensor
    
    Return :
        Tensor : same shape of input and output tensor
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, input : Tensor, output : Tensor)->Tensor:
        
        residual_connection = torch.add(input, output)
        
        return (residual_connection)
        