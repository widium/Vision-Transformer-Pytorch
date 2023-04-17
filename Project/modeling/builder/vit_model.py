# *************************************************************************** #
#                                                                              #
#    vit_model.py                                                              #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2023/04/17 16:56:18 by Widium                                    #
#    Updated: 2023/04/17 16:56:18 by Widium                                    #
#                                                                              #
# **************************************************************************** #

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import Linear
from torch.nn import Dropout
from torch.nn import Sequential
from torch.nn import LayerNorm

from .image_tokenization import ImageTokenizer
from .tokens_processing import ClassTokenPrepender
from .tokens_processing import PositionalEmbedding
from .encoder import TransformerEncoder


class VisionTransformerClassifier(Module): 
    """
    Vision Transformer model from Paper "https://arxiv.org/abs/2010.11929" for image Classification

    Args:
        `nbr_classes` (int): Number of classes for classification.
        `height` (int, optional): Height of the input image. Defaults to 224.
        `width` (int, optional): Width of the input image. Defaults to 224.
        `color_channels` (int, optional): Number of color channels in the input image. Defaults to 3.
        `patch_size` (int, optional): Size of the patches to divide the input image into. Defaults to 16.
        `embedding_size` (int, optional): Size of the token embeddings. Defaults to 768.
        `nbr_encoder_blocks` (int, optional): Number of transformer encoder blocks. Defaults to 12.
        `nbr_heads` (int, optional): Number of attention heads in the transformer. Defaults to 12.
        `mlp_units` (int, optional): Number of units in the mlp block inside the transformer. Defaults to 3072.
        `dropout_embedding` (float, optional): Dropout rate for the embeddings. Defaults to 0.1.
        `dropout_attention` (float, optional): Dropout rate for the attention block. Defaults to 0.0.
        `dropout_mlp` (float, optional): Dropout rate for the mlp block. Defaults to 0.1.
    """
    def __init__(
        self,
        nbr_classes : int,
        height : int = 224,
        width : int = 224,
        color_channels : int = 3,
        patch_size : int = 16,
        embedding_size : int = 768,
        nbr_encoder_blocks : int = 12,
        nbr_heads : int = 12,
        mlp_units : int = 3072,
        dropout_embedding : int = 0.1,
        dropout_attention : float = 0.0,
        dropout_mlp : float = 0.1,
        ):
        
        super().__init__()
        
        self.nbr_patches = int((height * width) / (patch_size**2))
        
        self.image_tokenizer = ImageTokenizer(
            patch_size=patch_size,
            color_channels=color_channels,
            embedding_size=embedding_size,
        )
        
        self.class_token_prepender = ClassTokenPrepender(
            embedding_size=embedding_size,
        )
        
        self.positional_embedding = PositionalEmbedding(
            nbr_token=self.nbr_patches + 1, # +1 for add class token
            embedding_size=embedding_size,
        )
        
        self.embedding_dropout = Dropout(p=dropout_embedding)
        
        self.transformer_encoder = TransformerEncoder(
            nbr_encoder_blocks=nbr_encoder_blocks,
            embedding_size=embedding_size,
            nbr_heads=nbr_heads,
            dropout_attention=dropout_attention,
            mlp_units=mlp_units,
            dropout_mlp=dropout_mlp,
        )
        
        self.classifier = Sequential(
            LayerNorm(normalized_shape=embedding_size),
            Linear(in_features=embedding_size, out_features=nbr_classes)
        )
        
    
    def forward(self, x : Tensor)->Tensor:
        """
        1. Tokenize the input image using the `ImageTokenizer`, which extracts patches and converts them into tokens.
        2. Prepend the class token to the sequence of patch tokens using the `ClassTokenPrepender`.
        3. Add positional embeddings to the token sequence using the `PositionalEmbedding`.
        4. Apply dropout to the embeddings of tokens using `Dropout` layer.
        5. Encoding the sequence of tokens (including the class token) using the `TransformerEncoder`.
        6. Extract the class token from the encoded sequence of tokens.
        7. Produce the output classification logits using the class token and the classifier.

        Args:
            x (Tensor): Input image tensor with shape (batch_size, color_channels, height, width).

        Returns:
            Tensor: Output classification logits with shape (batch_size, num_classes).
        """
        tokens = self.image_tokenizer(x)
        
        tokens = self.class_token_prepender(tokens)
        tokens = self.positional_embedding(tokens)
        tokens = self.embedding_dropout(tokens)
        
        tokens_encoded = self.transformer_encoder(tokens)
        class_token_encoded = tokens_encoded[:, 0] # Extract Class Token
        
        classification = self.classifier(class_token_encoded) # Use Class token for the Classification
        
        return (classification)