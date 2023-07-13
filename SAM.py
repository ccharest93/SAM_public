CONFIGS = {
    "vit_h": {
        "image_size": 1024,
        "patch_size": 16,
        "embed_dim": 1280,
        "depth": 32,
        "num_heads": 16,
        "encoder_global_attn_indexes":[7, 15, 23, 31],
        "prompt_embed_dim": 256,
    },
    "vit_l": {
        "image_size": 1024,
        "patch_size": 16,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "encoder_global_attn_indexes":[5, 11, 17, 23],
        "prompt_embed_dim": 256,
    },
    "vit_b": {
        "image_size": 1024,
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "encoder_global_attn_indexes":[2, 5, 8, 11],
        "prompt_embed_dim": 256,
    },
}
CHECKPOINTS = {"vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
               "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
               "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"}
import torch
from torch import nn
import os
import numpy as np
from ImageEncoderVIT import ImageEncoderViT
from PromptEncoder import PromptEncoder
from MaskDecoder import MaskDecoder
from layers.transformer import TwoWayTransformer
from SAM_transform import ResizeLongestSide
from typing import Optional, Tuple
from utils.model import GPUManager

from typing import Tuple, List, Dict, Any
class SAM_predict(nn.Module):
    mask_threshold: float = 0.0

    def __init__(self, version: str):
        super().__init__()
        self.version = version
        args = CONFIGS[version]
        image_embedding_size = args["image_size"] // args["patch_size"]
        
        self.image_encoder = ImageEncoderViT(
            depth=args["depth"],
            embed_dim=args["embed_dim"],
            img_size=args["image_size"],
            num_heads=args['num_heads'],
            patch_size=args['patch_size'],
            global_attn_indexes=args['encoder_global_attn_indexes'],
            out_chans=args["prompt_embed_dim"],
        )
        self.prompt_encoder = PromptEncoder(
            embed_dim=args["prompt_embed_dim"],
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(args["image_size"], args["image_size"]),
            mask_in_chans=16,
        )
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=args["prompt_embed_dim"],
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=args["prompt_embed_dim"],
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        
        self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), False)
        self.load_state_dict(self.load_weights(), strict=True)
        self.eval()
        
        self.transform =ResizeLongestSide(self.image_encoder.img_size)

    def load_weights(self, output_dir = "./pretrain_pth"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(os.path.join(output_dir, f"{self.version}.pth")):
            print(f"Downloading weights for {self.version}...")
            url = CHECKPOINTS[self.version]
            state_dict = torch.hub.load_state_dict_from_url(url, output_dir, map_location=torch.device('cpu'),file_name=f"{self.version}.pth")
        else:
            state_dict = torch.load(os.path.join(output_dir, f"{self.version}.pth"), map_location=torch.device('cpu'))
        return state_dict
    
    @torch.no_grad()
    def encode_image(
        self,
        image: np.ndarray,
    ) -> None:
        """
        Arguments:
          image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
        """

        # Resize
        image, x = self.transform.apply_image(image)
        # Normalize
        x = (x - self.pixel_mean) / self.pixel_std
        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = nn.functional.pad(x, (0, padw, 0, padh))
        # Encode
        with GPUManager(task="encode_image", device=torch.device('cuda'), verbose=True, tensors={
        "image_encoder":self.image_encoder, 
        "x": x })  as tensors:
            tensors['out'] = tensors['image_encoder'](tensors['x'])
            self.image_encoder = tensors['image_encoder'].to(torch.device('cpu'))
            return image, tensors['out'].to(torch.device('cpu'))
    
    @torch.no_grad()
    def encode_prompt(self,
        original_size,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        ):
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, original_size)
            point_coords = torch.as_tensor(point_coords[None, :, :], dtype=torch.float)
            point_labels = torch.as_tensor(point_labels[None, :], dtype=torch.int)
            points = (point_coords, point_labels)
        else:
            points = None
        if box is not None:
            box = self.transform.apply_boxes(box, original_size)
            box = torch.as_tensor(box[None, :], dtype=torch.float)
        if mask_input is not None:
            mask_input = torch.as_tensor(mask_input[None, :, :, :], dtype=torch.float)
        
        with GPUManager(task="encode_prompt", device=torch.device('cuda'), verbose=True ,tensors={
        "prompt_encoder":self.prompt_encoder, 
        "points":points, 
        "box":box, 
        "mask_input":mask_input }) as tensors:
            tensors['sparse_embeddings'], tensors['dense_embeddings'] = tensors['prompt_encoder'](
                points=tensors['points'],
                boxes=tensors['box'],
                masks=tensors['mask_input'],
                )
            self.prompt_encoder = tensors['prompt_encoder'].to(torch.device('cpu'))
            return  tensors['dense_embeddings'].to(torch.device('cpu')), tensors['sparse_embeddings'].to(torch.device('cpu'))
        
    @torch.no_grad()
    def decode_masks(
        self,       
        image_embeddings,
        dense_prompt_embeddings,
        sparse_prompt_embeddings,
        multimask_output,):
        with GPUManager(task="decode_masks", device=torch.device('cuda'), verbose=True ,tensors={
        "mask_decoder":self.mask_decoder,
        "image_embeddings":image_embeddings,
        "dense_prompt_embeddings":dense_prompt_embeddings,
        "sparse_prompt_embeddings":sparse_prompt_embeddings,
        "dense_pe":self.prompt_encoder.get_dense_pe() }) as tensors:
            tensors['low_res_masks'], tensors['iou_predictions'] = tensors['mask_decoder'](
                image_embeddings= tensors['image_embeddings'],
                image_pe= tensors['dense_pe'],
                dense_prompt_embeddings= tensors['dense_prompt_embeddings'],
                sparse_prompt_embeddings= tensors['sparse_prompt_embeddings'],
                multimask_output= multimask_output,
            )
            self.mask_decoder = tensors['mask_decoder'].to(torch.device('cpu'))
            low_res_masks = tensors['low_res_masks'].to(torch.device('cpu'))
            iou_predictions = tensors['iou_predictions'].to(torch.device('cpu'))

        return low_res_masks, iou_predictions


    @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs
    
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = nn.functional.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = nn.functional.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    
if __name__ == "__main__":
    model = SAM("vit_l")
