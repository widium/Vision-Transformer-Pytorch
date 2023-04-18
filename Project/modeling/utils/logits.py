# *************************************************************************** #
#                                                                              #
#    logits.py                                                                 #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2023/04/17 18:30:16 by Widium                                    #
#    Updated: 2023/04/17 18:30:16 by Widium                                    #
#                                                                              #
# **************************************************************************** #

import torch
from torch import Tensor

def logits_to_class_integer(logits : Tensor)->Tensor:
    """Convert Logits to Class Integers for Binary or Multi Classification

    Check the shape of Logits fo define type of Classification 
    
    Args:
        logits (Tensor): 2 dimensional Tensor with logits

    Returns:
        Tensor: Scalar of class prediction
    """
    if (len(logits.shape) == 1 or logits.shape[1] == 1):
        # BinaryClass
        y_preds = torch.sigmoid(logits)
        y_preds = torch.round(y_preds)

    elif (logits.shape[1] > 1):
        # MultiClass
        y_preds = torch.softmax(logits, dim=1)
        y_preds = torch.argmax(y_preds, dim=1)
    
    return (y_preds)

# **************************************************************************** #

def logits_to_probs(logits : Tensor)->Tensor:
    """Convert logits to Probability of Class Prediction

    Args:
        logits (Tensor): 2 dimensional Tensor with logits

    Returns:
        Tensor: Float Scalar of class Probability
    """
    if (len(logits.shape) == 1 or logits.shape[1] == 1):
        # BinaryClass
        y_probs = torch.sigmoid(logits)
        if y_probs < 0.5:
            y_probs = 1 - y_probs

    elif (logits.shape[1] > 1):
        # MultiClass
        y_preds = torch.softmax(logits, dim=1)
        y_probs = y_preds.max()
    
    return (y_probs * 100)