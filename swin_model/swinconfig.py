class SwinConfig:

    # base용도 파라미터
    # img_size = (252, 234)   patch_size = (3, 3)  # (18, 13) #(14, 18)
    img_size = (252, 234)
    patch_size = (18, 13)
    in_chans = 3  # 1
    num_classes = 18  # 액션 클래스
    embed_dim = 32  # 64 , 96(default)
    depths = [3, 3]
    num_heads = [4, 4]
    window_size = 3
    mlp_ratio = 4
    qkv_bias = True
    qk_scale = None
    drop_rate = 0.0 # 0.1 (3) ,0.2 (+ epoch 200 해야될차례)
    attn_drop_rate = 0.  # 0.
    drop_path_rate = 0.5
    ape = False
    patch_norm = True
    fused_window_process = False
    interpolation = 'bicubic'
    double_check = True

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def get_config(cls):
        return {name: val for name, val in vars(cls).items() if not name.startswith('__')}
