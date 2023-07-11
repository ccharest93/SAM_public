
import numpy as np
import torch

from SAM import SAM

from typing import Optional, Tuple

from SAM_Transforms import ResizeLongestSide
from utils.model import GPUManager


class SamPredictor:
    def __init__(
        self,
        sam_model: SAM,
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
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

        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_image = self.model.preprocess(input_image_torch)

        with GPUManager(tensors={"input_image":input_image, "model.image_encoder":self.model.image_encoder}, device=torch.device('cuda'),task="Embedding image",verbose=True) as tensors:
          self.features = tensors['model.image_encoder'](tensors['input_image']).to(torch.device('cpu'))
          self.model.image_encoder = tensors['model.image_encoder'].to(torch.device('cpu'))
        # Set various attributes
        self.original_size = image.shape[:2]
        self.input_size = tuple(input_image_torch.shape[-2:])
        self.is_image_set = True

    @torch.no_grad()
    def predict(
        self,
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
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Transform input prompts
        point_coords, point_labels, box, mask_input = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            point_coords = torch.as_tensor(point_coords[None, :, :], dtype=torch.float)
            point_labels = torch.as_tensor(point_labels[None, :], dtype=torch.int)
            points = (point_coords, point_labels)
        else:
            points = None

        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box = torch.as_tensor(box[None, :], dtype=torch.float)

        if mask_input is not None:
            mask_input = torch.as_tensor(mask_input[None, :, :, :], dtype=torch.float)


        with GPUManager(tensors={"model.prompt_encoder":self.model.prompt_encoder, "points":points, "box":box, "mask_input":mask_input}, device=torch.device('cuda'),task="Embedding prompt",verbose=True) as tensors:
          #Prompt encoder
          tensors['sparse_embeddings'], tensors['dense_embeddings'] = tensors['model.prompt_encoder'](
            points=tensors['points'],
            boxes=tensors['box'],
            masks=tensors['mask_input'],
          )
          del tensors['points'], tensors['box'], tensors['mask_input']
          tensors['model.mask_decoder'] = self.model.mask_decoder.to(torch.device('cuda'))
          tensors['features'] = self.features.to(torch.device('cuda'))
          #Mask decoder
          tensors['low_res_masks'], tensors['iou_predictions'] = tensors['model.mask_decoder'](
              image_embeddings=tensors['features'],
              image_pe=self.model.prompt_encoder.get_dense_pe(),
              sparse_prompt_embeddings= tensors['sparse_embeddings'],
              dense_prompt_embeddings= tensors['dense_embeddings'],
              multimask_output=multimask_output,
          )
          self.model.mask_decoder = tensors['model.mask_decoder'].to(torch.device('cpu'))
          self.model.prompt_encoder = tensors['model.prompt_encoder'].to(torch.device('cpu'))
          low_res_masks = tensors['low_res_masks'].to(torch.device('cpu'))
          iou_predictions = tensors['iou_predictions'].to(torch.device('cpu'))
        
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

        #threshold masks
        if not return_logits:
            masks = masks > self.model.mask_threshold

        masks_np = masks[0].detach().cpu().numpy()
        iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
        low_res_masks_np = low_res_masks[0].detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None
