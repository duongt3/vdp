import vdp
import torch
from torch import nn
import wandb
from einops import rearrange
from einops.layers.torch import Rearrange

# fork of https://github.com/lucidrains/vit-pytorch

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

def mult_var(mu_a, sigma_a, mu_b, sigma_b):
    sigma_ab = ((mu_a ** 2) @ sigma_a) + \
               ((mu_b ** 2) @ sigma_b) + \
               (sigma_a.mT @ sigma_b)
               
    return sigma_ab

# classes

class LinearHead(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.ln = vdp.LayerNorm(dim)
        self.lin = vdp.Linear(dim, num_classes)
    def forward(self, mu, sigma):
        mu, sigma = self.ln(mu, sigma)
        mu, sigma = self.lin(mu, sigma)
        return mu, sigma
            

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.ln = vdp.LayerNorm(dim)
        self.lin1 = vdp.Linear(dim, hidden_dim)
        self.gelu = vdp.GELU()
        self.lin2 = vdp.Linear(hidden_dim, dim)
        
    def forward(self, mu, sigma):
        mu, sigma = self.ln(mu, sigma)
        mu, sigma = self.lin1(mu, sigma)
        mu, sigma = self.gelu(mu, sigma)
        mu, sigma = self.lin2(mu, sigma)
        return mu, sigma

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = vdp.LayerNorm(dim)

        self.attend = vdp.Softmax(dim=-1)

        self.to_qkv = vdp.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = vdp.Linear(inner_dim, dim, bias = False)

    def forward(self, mu, sigma):
        mu, sigma = self.norm(mu, sigma)

        mu_qkv, sigma_qkv = self.to_qkv(mu, sigma)
        mu_qkv, sigma_qkv = mu_qkv.chunk(3, dim = -1), sigma_qkv.chunk(3, dim = -1)
        
        mu_q, mu_k, mu_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), mu_qkv)
        sigma_q, sigma_k, sigma_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), sigma_qkv)

        mu_dots = torch.matmul(mu_q, mu_k.transpose(-1, -2)) * self.scale
        # sigma_dots = mult_var(mu_q, sigma_q.transpose(-1, -2), mu_k, sigma_k.transpose(-1, -2)) * self.scale
        sigma_dots = torch.matmul(sigma_q, sigma_k.transpose(-1, -2)) * self.scale

        mu_attn, sigma_attn = self.attend(mu_dots, sigma_dots)

        mu_out = torch.matmul(mu_attn, mu_v)
        # sigma_out = mult_var(mu_attn, sigma_attn, mu_v, sigma_v)
        sigma_out = torch.matmul(sigma_attn, sigma_v)
        mu_out = rearrange(mu_out, 'b h n d -> b n (h d)')
        sigma_out = rearrange(sigma_out, 'b h n d -> b n (h d)')
        return self.to_out(mu_out, sigma_out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, mu, sigma):
        for attn, ff in self.layers:
            mu_attn, sigma_attn = attn(mu, sigma)
            mu_attn += mu
            mu_ff, sigma_ff = ff(mu_attn, sigma_attn)
            mu_ff += mu_attn
        return mu_ff, sigma # sigma by-passes inner residuals

class SimpleViT_vdp(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            vdp.Linear(patch_dim, dim, input_flag=True),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()
        self.linear_head = LinearHead(dim, num_classes)

    def forward(self, img):
        *_, h, w, dtype = *img.shape, img.dtype

        mu, sigma = self.to_patch_embedding(img)
        mu_pe = posemb_sincos_2d(mu)
        sigma_pe = posemb_sincos_2d(sigma)
        
        
        mu_re = rearrange(mu, 'b ... d -> b (...) d') + mu_pe
        sigma_re = rearrange(sigma, 'b ... d -> b (...) d') + sigma_pe

        mu_trans, sigma_trans = self.transformer(mu_re, sigma_re)
        mu_trans = mu_trans.mean(dim = 1)
        sigma_trans = sigma_trans.sum(dim=1) / (sigma_trans.shape[1]**2)    # expectation of a variance? 1/N^2 sum (var)

        # x = self.to_latent(x)
        mu, sigma = self.linear_head(mu_trans, sigma_trans)
        return mu, sigma
    
    def score(self, logits, y):
        score = torch.sum(torch.argmax(logits, dim=1) == y)/len(logits)
        return score.cpu().numpy()
    
    def get_loss(self, mu, sigma, y):
        log_det, nll = vdp.ELBOLoss(mu, sigma, y)
        kl = vdp.gather_kl(self)
        # if self.alpha is None:
        #     self.alpha, self.tau = vdp.scale_hyperp(log_det, nll, kl)
        # loss = self.alpha * log_det + nll + self.tau * sum(kl)
        loss = log_det + 100 * nll + sum(kl)
        return loss