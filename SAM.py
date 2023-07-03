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
from ImageEncoderVIT import ImageEncoderViT
from functools import partial
from torch import nn
from PromptEncoder import PromptEncoder
from MaskDecoder import MaskDecoder
from layers.transformer import TwoWayTransformer
class SAM(nn.Module):
    def __init__(version: str):
        super().__init__()
        args = CONFIGS[version]
        pixel_mean=[123.675, 116.28, 103.53]
        pixel_std=[58.395, 57.12, 57.375]

        image_embedding_size = args["image_size"] // args["patch_size"]
        image_encoder = ImageEncoderViT(
            depth=args["depth"],
            embed_dim=args["embed_dim"],
            img_size=args["image_size"],
            num_heads=args['num_heads'],
            patch_size=args['patch_size'],
            global_attn_indexes=args['encoder_global_attn_indexes'],
            out_chans=args["prompt_embed_dim"],
        )
        prompt_encoder = PromptEncoder(
            embed_dim=args["prompt_embed_dim"],
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(args["image_size"], args["image_size"]),
            mask_in_chans=16,
        )
        mask_decoder = MaskDecoder(
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
        ),
