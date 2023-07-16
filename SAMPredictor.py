
import numpy as np
import torch
from torch import nn
from SAM import SAM_predict

from typing import Optional, Tuple

from SAM_transform import ResizeLongestSide
from utils.model import GPUManager


class SamPredictor:
    def __init__(
        self,
        sam_model: SAM_predict,
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__()
        self.model = sam_model
        self.reset_image()

    @torch.no_grad()
    def set_image(
        self,
        image: np.ndarray,
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """

        self.reset_image()
        # Apply transform
        self.original_size = image.shape[:2]
        image, x = self.model.transform.apply_image(image)
        self.input_size = tuple(x.shape[-2:])
        # Encode image
        return image, self.model.encode_image(x)

    @torch.no_grad()
    def predict(
        self,
        image_embeddings: torch.Tensor,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          image_embeddings (torch.Tensor): The image embeddings for the image
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        
        if not self.original_size:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")
        # Prompt Encoding
        dense_embeddings, sparse_embeddings = self.model.encode_prompts(self.original_size, point_coords, point_labels, box, mask_input)
        # Mask Decoding
        low_res_masks, iou_predictions = self.model.decode_masks(image_embeddings, dense_embeddings, sparse_embeddings, multimask_output)
        # Mask Upsampling
        masks = nn.functional.interpolate(
            low_res_masks,
            (self.model.image_encoder.img_size, self.model.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : self.input_size[0], : self.input_size[1]]
        masks = nn.functional.interpolate(masks, self.original_size, mode="bilinear", align_corners=False)
        # Mask thresholding
        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks[0].numpy(), iou_predictions[0].numpy(), low_res_masks[0].numpy()
    
    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.original_size = None
        self.input_size = None
