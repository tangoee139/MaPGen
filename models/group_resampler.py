import torch
import torch.nn as nn
import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum



class PerceiverResampler(nn.Module):
    def __init__(self, dim=2048, depth=3, num_latents=256):
        super().__init__()

        self.group_attn = GroupAttention(dim=dim, dim_head=64, heads=8)

        self.segment_embeddings = nn.Embedding(24, 2048)
        
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=64, heads=8),
                        FeedForward(dim=dim, mult=4),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, v = x.shape[:2]
 
        x = self.group_attn(x, (24, 49))

        segment_ids = torch.arange(24).repeat_interleave(49).to(x.device)
        segment_embeddings = self.segment_embeddings(segment_ids)
        x = x + segment_embeddings

        #x_global_S = x_local.reshape(-1, 24, 49, 2048).mean(dim=-2)
        #x_global_T = x_local.reshape(-1, 24, 49, 2048).mean(dim=-3)
        #x = torch.cat([x_local, x_global_S, x_global_T], dim = -2)

        latents = repeat(self.latents, "n d -> b T n d", b=b, T=1)
        x = x.unsqueeze(1)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm(latents).squeeze(1)

    

   
# =================================resampler related =================================
def exists(val):
    return val is not None


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class GroupAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads
        self.scale = self.dim_head**-0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.norm = nn.LayerNorm(dim)


    def forward(self, x, inputs_size):
        t, n = inputs_size #t=24, n=49
        h = self.heads

        x = self.norm(x)
        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)
        q, k, v = rearrange_many((q, k, v), "b h (t n) d -> b h t n d", t=t)
        q = q * self.scale
        

        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b (t n) (h d)", h=h)

        return self.to_out(out)


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latent (torch.Tensor): latent features
                shape (b, T, n2, D)
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale

        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out)
    
