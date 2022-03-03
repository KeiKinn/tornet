from models.swin_transformer import SwinTransformer
from models.swin_mlp import SwinMLP

if __name__ == '__main__':
    from mmcv import Config
    from models import build_model

    path = './configs/swin_base_patch4_window7_224.yaml'
    cfg = Config.fromfile(path)
    # build the model method 1
    model = build_model(cfg)
    # build the model method 2
    model2 = SwinTransformer(img_size=cfg.DATA.IMG_SIZE)
    print(model)

    path = './configs/swin_mlp_base_patch4_window7_224.yaml'
    cfg = Config.fromfile(path)
    model = build_model(cfg)
    # build the model method 2
    model2 = SwinMLP(img_size=cfg.DATA.IMG_SIZE)
    print(model)
