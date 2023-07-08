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
from ImageEncoderVIT import ImageEncoderViT
from PromptEncoder import PromptEncoder
from MaskDecoder import MaskDecoder
from layers.transformer import TwoWayTransformer

from typing import Tuple, List, Dict, Any
class SAM(nn.Module):
    mask_threshold: float = 0.0
    pixel_mean= torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std= torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.load_state_dict(self.load_weights(), strict=True)
        self.eval()

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

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = nn.functional.pad(x, (0, padw, 0, padh))
        return x
    
if __name__ == "__main__":
    model = SAM("vit_l")
