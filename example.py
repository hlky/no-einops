import torch
import no_einops

x = torch.randn([2, 4, 8, 16, 16])
no_einops.rearrange(x, "b c t h w -> (b t) c h w")
