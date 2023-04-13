## Vision Transformer

~~~python
import torch
import numpy as np
~~~

### Image to Patch
![](https://i.imgur.com/NHh1Mug.png)
- $\Large N = \frac{H\times{W}}{P^2}$

~~~python
height = 224
width = 224
color_channels = 3
patch_size = 16

number_patches = int((height * width) / (patch_size**2))
~~~

### Visualize Image to Patch


~~~python
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose, Resize, ToTensor
import wget
~~~

Import image from web and pre-processing it


~~~python
preprocessing = Compose([
    ToTensor(),
    Resize(size=(224, 224)),
])

img_path = wget.download("https://i.imgur.com/o0uWjQz.png")
image = Image.open(img_path)

img = preprocessing(image).unsqueeze(dim=0)
img_permuted = img.permute(0, 2, 3, 1)

plt.imshow(img_permuted[0])
~~~
![png](patch_files/patch_6_1.png)
    


### Create Visualization for patch creation on image 


~~~python
from torch import Tensor

def visualize_img_to_patch(
    img : Tensor,
    height : int,
    width : int,
    patch_size : int
)->None:
    """
    Visualize img sliced in patch 

    Args:
        `img` (Tensor): img in shape (H x W x C)
        `height` (int): height size
        `width` (int): width size
        `patch_size` (int): patch size like (16 x 16)
    """
    
    assert height == width, "Image must have same height and width"
    assert height % patch_size == 0, "Image size must be divisible by patch size"
    
    plt.imshow(img)
    plt.show()

    ## -------------- Compute Number of Patches -------------- ##
    
    height_patches = height // patch_size
    width_patches = width // patch_size
    all_patches = height_patches * width_patches

    print(f"Number of Patches : {all_patches}\
        \nPatch Size : {(patch_size, patch_size)}\
        \nNumber of Patches per Column : {height_patches}\
        \nNumber of Patches per Row : {width_patches}")

    ## -------------- Setup Indexing for Image -------------- ##
    
    indexes_height_patch = range(0, height, patch_size)
    indexes_width_patch = range(0, width, patch_size)

    _, ax = plt.subplots(
        figsize=(height_patches, width_patches),
        nrows=width_patches,
        ncols=height_patches,
        sharex=True,
        sharey=True
    )

    ## -------------- Create Patch and display them  -------------- ##
    
    for h, index_height in enumerate(indexes_height_patch):
        
        for w, index_width in enumerate(indexes_width_patch):
            
            current_patch = img[
                index_height : index_height + patch_size,
                index_width : index_width + patch_size,
                :,
            ]
            
            ax[h, w].imshow(current_patch)
            ax[h, w].set_xticks([])
            ax[h, w].set_yticks([])
            
            
    plt.show()

~~~


~~~python
visualize_img_to_patch(
    img=img_permuted[0],
    height=224,
    width=224,
    patch_size=16
)
~~~ 
![png](patch_files/patch_9_0.png)
> Number of Patches : 196        
> Patch Size : (16, 16)        
> Number of Patches per Column : 14        
> Number of Patches per Row : 14
![png](patch_files/patch_9_2.png)
    


### Create Non-Overlapping Patch with `Conv2d` !
![](https://i.imgur.com/PqGZSjV.png)
![](https://i.imgur.com/nvNNtpE.png)


~~~python

from torch.nn import Module
from torch.nn import Conv2d

class PatchExtractor(Module):
    """
    Create and learn to extract valuable information in input image

    Create non-overlapping patch with Conv2d and extract 1 pixel per patch
    
    Args:
        `patch_size` (int): size of patch like 16x16
        `color_channels` (int) : number of color channels in input Image
    """
    def __init__(self, patch_size : int, color_channels : int):
        
        super().__init__()
        
        self.patch_extractor = Conv2d(
            in_channels=color_channels,
            out_channels=color_channels,
            kernel_size=(patch_size, patch_size),
            stride=patch_size
        )
    
    def forward(self, x : Tensor)->Tensor:
        """
        Extract patches information on input image

        Args:
            `x` (Tensor): input image with size (color, height, width)

        Returns:
            Tensor: Patches information of input information with shape (color, height // patch_size, width // patch_size)
        """
        patches = self.patch_extractor(x)
        
        return (patches)
        
    
~~~


~~~python
patches_extractor = PatchExtractor(patch_size=16, color_channels=3)

patches = patches_extractor(img)
print(patches.shape)

patch_permuted = patches.permute(0, 2, 3, 1)
patch_numpy = patch_permuted.detach().numpy()
patch_numpy = (patch_numpy - np.min(patch_numpy)) / (np.max(patch_numpy) - np.min(patch_numpy))

plt.imshow(patch_numpy[0], cmap="viridis")

~~~
> torch.Size([1, 3, 14, 14])
![png](patch_files/patch_12_2.png)
    


#### Visualize Patch Extraction


~~~python
from torch import Tensor

def visualize_patch_extraction(input_image : Tensor, patches : Tensor)->None:
    """
    Visualize an input image with its extracted patches

    Args:
        `input_image` (Tensor): input image with shape (color, height, width)
        `patches` (Tensor): patches tensor with shape (color, height, width)
    """
    
    ## -------------- Convert Tensor to Numpy Array -------------- ##
    
    input_image_permuted = input_image.permute(1, 2, 0)
    
    patch_permuted = patches.permute(1, 2, 0)
    patch_numpy = patch_permuted.detach().numpy()
    patch_numpy = (patch_numpy - np.min(patch_numpy)) / (np.max(patch_numpy) - np.min(patch_numpy))

    ## -------------- Compute Number of Patches -------------- ##
    
    height = patch_numpy.shape[0]
    width = patch_numpy.shape[1]
    
    number_patches = height * width
    print(f"Number of Patches : {number_patches}")

    ## -------------- Display Input Image -------------- ##
    
    plt.figure(figsize=(5, 5))
    plt.title(f"Input Image with Shape {input_image.shape}")
    plt.imshow(input_image_permuted)
    plt.show()
    
    ## -------------- Display Patches -------------- ##
    
    indexes_height_patch = range(0, height, 1)
    indexes_width_patch = range(0, width, 1)

    _, ax = plt.subplots(
        figsize=(5, 5),
        nrows=width,
        ncols=height,
        sharex=True,
        sharey=True
    )
    plt.suptitle(f"Patches Extracted with Shape : {patches.shape}")

    ## -------------- Indexing Image with Patches Index  -------------- ##
    
    for h, index_height in enumerate(indexes_height_patch):
        
        for w, index_width in enumerate(indexes_width_patch):
            
            current_patch = patch_numpy[
                index_height : index_height + patch_size,
                index_width : index_width + patch_size,
                :,
            ]
            
            ax[h, w].imshow(current_patch)
            ax[h, w].set_xticks([])
            ax[h, w].set_yticks([])
~~~


~~~python
visualize_patch_extraction(input_image=img[0], patches=patches[0])
~~~
> Number of Patches : 196
![png](patch_files/patch_15_1.png)
![png](patch_files/patch_15_2.png)
    


### PatchTokenizer
![](https://i.imgur.com/jZsYT1I.png)


~~~python
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
~~~


~~~python
tokenizer = PatchTokenizer()

patches_tokenized = tokenizer(patches)

patches_tokenized.shape
~~~
> torch.Size([1, 196, 3])



#### Visualize Flatten Patch to Sequence of Tokens


~~~python
def visualize_patch_tokenized(patches_tokenized : Tensor):
    """
    Visualize sequence of tokenized patch in line 

    Args:
        `tokens` (Tensor): tokenized patch with shape (batch_size, nbr_token, color)
    """
    nbr_tokens = patches_tokenized.shape[1]
    color = patches_tokenized.shape[2]
    
    tokens_numpy = patches_tokenized.detach().numpy()
    tokens_numpy = (tokens_numpy - np.min(tokens_numpy)) / (np.max(tokens_numpy) - np.min(tokens_numpy))
    
    plt.figure(figsize=(100, 7))
    plt.subplots_adjust(wspace=0.5)
    plt.suptitle(f"{nbr_tokens} Tokens (Patch) with {color} Color Channel(s) Value", fontsize=100)
    
    for t in range(nbr_tokens):
        current_token = tokens_numpy[:, t, :]
        current_token = current_token.reshape((1, 1, color))
        plt.subplot(1, nbr_tokens, t + 1)
        plt.imshow(current_token)
        plt.axis(False)
~~~


~~~python
visualize_patch_tokenized(patches_tokenized)
~~~
![png](patch_files/patch_21_0.png)
    


### Token Embedding
![](https://i.imgur.com/xaf3tvT.png)


~~~python
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
~~~


~~~python
embedder = PatchTokenEmbedding(color_channels=3, embedding_size=768)

tokens_embedding = embedder(patches_tokenized)

tokens_embedding.shape 
~~~
> torch.Size([1, 196, 768])



#### Visualize Token Embedding 


~~~python
def visualize_tokens_embedding(tokens_embedding : Tensor):
    """
    Visualize 100 first value in Embedding Vector for each tokens

    Args:
        `tokens_embedding` (Tensor): sequence of tokens with shape (batch_size, nbr_tokens, embedding_size)
    """
    nbr_tokens = tokens_embedding.shape[1]

    plt.figure(figsize=(20, 7))
    plt.subplots_adjust(wspace=0.5)
    plt.suptitle(f"{nbr_tokens} Token Embedding Vector with 100 first Value")

    for index, token in enumerate(tokens_embedding[0]):
        token_numpy = token.detach().numpy()
        token_numpy = token_numpy[:100, np.newaxis, np.newaxis]
        plt.subplot(1, nbr_tokens, index + 1)
        plt.imshow(token_numpy)
        plt.axis(False)
~~~


~~~python
visualize_tokens_embedding(tokens_embedding)
~~~
![png](patch_files/patch_27_0.png)
    


### Put all layers together to Convert Image to Embedding Tokens !
![](https://i.imgur.com/4MXVwO5.png)


~~~python
from torch.nn import Module

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

~~~


~~~python
image_tokenizer = ImageTokenizer(
    patch_size=16, 
    color_channels=3, 
    embedding_size=768
)

tokens = image_tokenizer(img)

tokens.shape
~~~
> torch.Size([1, 196, 768])

~~~python
import torch
~~~

### Create and Append Class Token to Patch Token 
![](https://i.imgur.com/Vwk2hbe.png)

- The class token serves as a **summary of global information extracted from patches during encoding** and
provides additional information for classification tasks.
- Create a learnable class token vector with `Parameter` and prepend it to the sequence of patch embedding vectors with `torch.cat()`


~~~python
from torch import Tensor
from torch.nn import Module
from torch.nn import Parameter

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
        
~~~


~~~python
token_prepender = ClassTokenPrepender(embedding_size=768)
dummy_tokens = torch.randn(size=(1, 196, 768))

tokens_embedding = token_prepender(dummy_tokens)
tokens_embedding.shape
~~~
> torch.Size([1, 197, 768])



### Positional Embedding
![](https://i.imgur.com/9Mv4UVO.png)
- Positional embeddings are learnable vectors,
- initialized randomly and updated during training,
- that represent the **spatial locations** of patch tokens in an image,
- **Help the Self Attention mechanism to considering patch positions.**
- The Positional Embedding must be apply after class token creation this ensure that the model treats the class token as an **integral part of the input sequence and accounts for its position**
***
- Create Tensor with same size of tokens with Learnable random values
- Wrapped into `Parameters` with Gradient Tracking
- Use **Element-Wise Addition** between embedding_vectors and positional_embedding_vectors


~~~python
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
~~~


~~~python
positional_embedder = PositionalEmbedding(nbr_token=197, embedding_size=768)

dummy = torch.randn(size=(1, 197, 768))

positional_embedder(dummy).shape
~~~
> torch.Size([1, 197, 768])





