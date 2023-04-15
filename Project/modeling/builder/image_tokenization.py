# *************************************************************************** #
#                                                                              #
#    image_tokenization.py                                                     #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2023/04/15 12:01:23 by Widium                                    #
#    Updated: 2023/04/15 12:01:23 by Widium                                    #
#                                                                              #
# **************************************************************************** #

from torch import Tensor
from torch.nn import Module

from .patch_extraction import PatchExtractor
from .patch_tokenization import PatchTokenizer
from .patch_embedding import PatchTokenEmbedding

class ImageTokenizer(Module):
    """
    Create patches, extract value on it, flatten it to sequence of tokens and embedding tokens

    Args:
        `patch_size` (int): size of created patches 
        `color_channels` (int): nbr of color channels in input image
        `embedding_size` (int): size of fixed-lenght vector of embedding tokens
    """
    def __init__(self, patch_size : int, color_channels : int, embedding_size : int):
        
        super().__init__()
        
        self.patch_extractor = PatchExtractor(
            patch_size=patch_size,
            color_channels=color_channels
        )
        
        self.patch_tokenizer = PatchTokenizer()
        
        self.token_embedding = PatchTokenEmbedding(
            color_channels=3, 
            embedding_size=embedding_size
        )
        
    def forward(self, x : Tensor)->Tensor:
        """
        Extract patch with `PatchExtractor`
        Tokenize patch into sequence with `PatchTokenizer`
        Embedding tokenized patch with `PatchTokenEmbedding`

        Args:
            x (Tensor): input image with shape (batch_size, color, height, width)

        Returns:
            Tensor: sequence of embedding tokens with shape (batch_size, nbr_tokens, embedding_size)
        """
        patches = self.patch_extractor(x)
        patches_tokenized = self.patch_tokenizer(patches)
        tokens_embedding = self.token_embedding(patches_tokenized)
        
        return (tokens_embedding)
