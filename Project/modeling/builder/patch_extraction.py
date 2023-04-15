# *************************************************************************** #
#                                                                              #
#    patch_extraction.py                                                       #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2023/04/15 11:58:59 by Widium                                    #
#    Updated: 2023/04/15 11:58:59 by Widium                                    #
#                                                                              #
# **************************************************************************** #

from torch import Tensor
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
        
    