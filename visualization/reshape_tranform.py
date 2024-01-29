'''ViT 计算grad-CAM要用到的函数，其他模型不需要'''
class ReshapeTransform:
    def __init__(self, model):
        # input_size = model.patch_embed.img_size
        # patch_size = model.patch_embed.patch_size
        # self.h = input_size[0] // patch_size[0]
        # self.w = input_size[1] // patch_size[1]
        input_size = model.image_size
        patch_size = model.patch_size
        self.h = input_size // patch_size
        self.w = input_size // patch_size

    def __call__(self, x):
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        result = x[:, 1:, :].reshape(x.size(0),
                                    self.h,
                                    self.w,
                                    x.size(2))

        # Bring the channels to the first dimension,
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)
        return result