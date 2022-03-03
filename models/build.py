# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
from .swin_mlp import SwinMLP


def parse_bool(b):
    if b == 'True':
        return True


def build_model(config):
    model_type = config.model.type
    if model_type == 'swin':
        model = SwinTransformer(img_size=config.data.img_size,
                                patch_size=config.model.swin.patch_size,
                                in_chans=config.model.swin.in_chans,
                                num_classes=config.model.num_classes,
                                embed_dim=config.model.swin.embed_dim,
                                depths=config.model.swin.depths,
                                num_heads=config.model.swin.num_heads,
                                window_size=config.model.swin.window_size,
                                mlp_ratio=config.model.swin.mlp_ratio,
                                qkv_bias=None if config.model.swin.qkv_bias == 'None' else config.model.swin.qkv_bias,
                                qk_scale=None if config.model.swin.qk_scale == 'None' else config.model.swin.qk_scale,
                                drop_rate=config.model.drop_rate,
                                drop_path_rate=config.model.drop_path_rate,
                                ape=parse_bool(config.model.swin.ape),
                                patch_norm=parse_bool(config.model.swin.patch_norm),
                                use_checkpoint=parse_bool(config.train.use_checkpoint))
    elif model_type == 'swin_mlp':
        model = SwinMLP(img_size=config.data.img_size,
                        patch_size=config.model.swin_mlp.patch_size,
                        in_chans=config.model.swin_mlp.in_chans,
                        num_classes=config.model.num_classes,
                        embed_dim=config.model.swin_mlp.embed_dim,
                        depths=config.model.swin_mlp.depths,
                        num_heads=config.model.swin_mlp.num_heads,
                        window_size=config.model.swin_mlp.window_size,
                        mlp_ratio=config.model.swin_mlp.mlp_ratio,
                        drop_rate=config.model.drop_rate,
                        drop_path_rate=config.model.drop_path_rate,
                        ape=config.model.swin_mlp.ape,
                        patch_norm=config.model.swin_mlp.patch_norm,
                        use_checkpoint=config.train.use_checkpoint)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
