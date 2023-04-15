# *************************************************************************** #
#                                                                              #
#    tokens_processing.py                                                      #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2023/04/15 12:31:22 by Widium                                    #
#    Updated: 2023/04/15 12:31:22 by Widium                                    #
#                                                                              #
# **************************************************************************** #

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import Parameter

# ******************************************************************************************************************************

class ClassTokenPrepender(Module):
    """
    Create a learnable class token vector and prepend it to the sequence of patch embedding vectors.
    The class token serves as a summary of global information extracted from patches during encoding and
    provides additional information for classification tasks.

    Args:
        embedding_size (int): embedding size of other embedding vectors 
    """
    def __init__(self, embedding_size : int):
        
        super().__init__()
        
        self.embedding_size = embedding_size
        
        class_token_vector = torch.randn(
            size=(1, 1, self.embedding_size)
        )

        self.class_token_vector = Parameter(
            data=class_token_vector,
            requires_grad=True #  convert to learnable parameters
        )
    
    def forward(self, x : Tensor)->Tensor:
        """
        Prepend the learnable class token vector to the input tensor. 
        If the input tensor has more than one batch, 
        the class token vector is expanded to match the batch size.

        Args:
            x (Tensor): input tensor of size [batch_size, nbr_token, embedding_size]

        Returns:
            Tensor: input tensor with the class token prepended, 
                    resulting in size [batch_size, nbr_token + 1, embedding_size]

        """
        batch_size = x.shape[0]
        
        if (batch_size > 1):
            class_token_vector = self.class_token_vector.expand(batch_size, 1, self.embedding_size)
        else :
            # don't repeat the token
             class_token_vector = self.class_token_vector
        
        x = torch.cat(tensors=[class_token_vector, x], dim=1) # Preprend the class token in input tensor
        
        return (x)
        
# ******************************************************************************************************************************
   
from torch.nn import Module
from torch.nn import Parameter

class PositionalEmbedding(Module):
    """
    Create learnable positional embeddings representing spatial locations of input tokens.
    The positional embeddings have the same size as the input embeddings for effectuate element-wise addition.
    This helping the self-attention mecanism to considering tokens position.

    Args:
        nbr_token (int): nbr of input Embedding Vector (token)
        embedding_size (int) : size of input Embedding Vector
    """
    def __init__(self, nbr_token : int , embedding_size : int):
        
        super().__init__()
        
        self.nbr_token = nbr_token
        self.embedding_size = embedding_size
        
        positional_embedding = torch.randn(
            size=(1, self.nbr_token, self.embedding_size)
        ) 

        self.positional_embedding = Parameter(
            data=positional_embedding,
            requires_grad=True #  convert to learnable parameters
        )
    
    def forward(self, x : Tensor)->Tensor:
        """
        Apply positional embeddings to the input tensor by adding them element-wise. 
        If the input tensor has more than one batch, the positional embeddings are expanded 
        to match the batch size.

        Args:
            x (Tensor): input tensor of size [batch_size, nbr_token, embedding_size]

        Returns:
            Tensor: tensor with the same shape as the input tensor, with positional embeddings added element-wise
        """
        batch_size = x.shape[0]
        
        if (batch_size > 1):
            positional_embedding = self.positional_embedding.expand(
                batch_size,
                self.nbr_token, 
                self.embedding_size
            )

        else :
            # # don't expand the tensor
            positional_embedding = self.positional_embedding
        
        # apply element_wise addition
        embedding_vectors = torch.add(x, positional_embedding)

        return (embedding_vectors)
    
# ******************************************************************************************************************************