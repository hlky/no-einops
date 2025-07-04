import torch
import einops

x = torch.randn([2, 4, 8, 16, 16])
x_einops = einops.rearrange(x, "b c t h w -> (b t) c h w")

x = torch.permute(x, [0, 2, 1, 3, 4])
x = torch.reshape(
    x,
    [
        x.shape[0] * x.shape[1],
        x.shape[2],
        x.shape[3],
        x.shape[4],
    ],
)

torch.testing.assert_close(x, x_einops)
