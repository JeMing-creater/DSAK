import torch
import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block


class Encoder(nn.Module):
    def __init__(self,
                 img_size=(32, 32),
                 patch_size=4,
                 modality=3,
                 embed_dim=1024,
                 mlp_ratio=4.,
                 num_heads=16,
                 norm_layer=nn.LayerNorm,
                 depth=24) -> None:
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, modality, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim))  # requires_grad=False 即为不可训练；num_patches + 1 是因为cls_token也占了一个位置
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, seg_length, dim
        len_keep = int(L * (1 - mask_ratio))  # 需要保留的patch数目

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1] noise.shape: [N, L]

        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove; dim=1按照seq_length这个维度排序
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # 通过ids_restore来将序列还原

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]  # 要保留下来的
        # Gathers values along an axis specified by dim. https://zhuanlan.zhihu.com/p/352877584
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  # mask之后保留下来的token序列

        # generate the binary mask for decoder: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0  # nomask的值为0，mask的值为1
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)  # 原始img得到的token序列的mask顺序
        return x_masked, mask, ids_restore

    def forward(self, x, mask_ratio=0.75):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]  # 1, 1, 1024
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)  # cls_token准备与x进行拼接 64, 1, 1024
        x = torch.cat((cls_tokens, x), dim=1)  # 拼接起来，可以作为vit的输入了 bs, 17, 1024
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore


class Decoder(nn.Module):
    def __init__(self,
                 decoder_embed_dim=512,
                 embed_dim=1024,
                 patch_size=16,
                 modality=3,
                 num_patches=None,
                 decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 ) -> None:
        super().__init__()
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))  # mask ratio 75%
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * modality, bias=True)

    def forward(self, x, ids_restore):
        x = self.decoder_embed(x)  # Linear,此时传进来的x是latent（保留下来的图像） bs, 65, 512
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)  # 64, 147,512
        x_ = torch.cat([x[:, 1:, :], mask_tokens],
                       dim=1)  # （latent + mask）no cls token，因为下一行作unsheffle时，ids_restore没有cls_token，所以此时先不考虑cls_token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[
            2]))  # unshuffle，还原成patch的原始顺序 64, 196, 512
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token，至此x = cls_token+ latent + mask 64, 197, 512
        # add pos embed
        x = x + self.decoder_pos_embed
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)  # 64, 197, 512
        # predictor projection
        x = self.decoder_pred(x)  # 64, 197, 768
        # remove cls token
        x = x[:, 1:, :]  # x = latent + mask 64, 196, 768
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=(32, 32),
                 patch_size=4,
                 modality=3,
                 embed_dim=1024,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 mlp_ratio=4.,
                 num_heads=16,
                 decoder_num_heads=16,
                 norm_layer=nn.LayerNorm,
                 depth=24):
        super(VisionTransformer, self).__init__()
        self.encoder = Encoder(img_size=img_size,
                               patch_size=patch_size,
                               modality=modality,
                               embed_dim=embed_dim,
                               mlp_ratio=mlp_ratio,
                               num_heads=num_heads,
                               norm_layer=norm_layer,
                               depth=depth)
        self.decoder = Decoder(decoder_embed_dim=decoder_embed_dim,
                               embed_dim=embed_dim,
                               patch_size=patch_size,
                               modality=modality,
                               decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads,
                               mlp_ratio=mlp_ratio, norm_layer=norm_layer,
                               num_patches=self.encoder.patch_embed.num_patches)


    def forward(self, imgs, mask_ratio=0):
        x, mask, ids_restore = self.encoder(imgs, mask_ratio)
        pred = self.decoder(x, ids_restore)
        return pred, mask


def unpatchify(x,patch_size):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = patch_size
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))  # [N, patch_h, patch_w, patch_size, patch_size, 3]
    x = torch.einsum('nhwpqc->nchpwq', x)  # [N, 3, patch_h, patch_size, patch_w, patch_size]
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))  # [N, 3, img_h, img_w]
    return imgs

def patchify(imgs,patch_size):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p  # patch_h 和patch_w是patch在img的height和width上的数目
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))  # [N, 3, patch_h, patch_size, patch_w, patch_size]
    x = torch.einsum('nchpwq->nhwpqc', x)  # [N, patch_h, patch_w, patch_size, patch_size, 3]
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))  # [N, patch_h * patch_w, patch_size * patch_size * 3]
    return x

class MAELoss(object):
    def __init__(self, patch_size=4, norm_pix_loss=True):
        super().__init__()
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss

    def __call__(self, pred, imgs, mask):
        """
        imgs: [N, H, W, D]
        pred: [N, L, p*p*d]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = patchify(imgs,self.patch_size)  # [N, patch_h * patch_w, patch_size * patch_size * 3] 即batch，patch_num, patch大小 64, 196, 768
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)  # 对patch求mean和var
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5  # 归一化，1.e-6防止方差为0.

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

if __name__ == '__main__':
    input = torch.rand(1, 3, 32, 32)
    model = VisionTransformer()
    y, mask = model(input.float(), mask_ratio=0.75)
    # loss
    loss = MAELoss()
    l = loss(y, input, mask)
    # 重构
    y = unpatchify(y,4)
    print(l)
    print(y.shape)
