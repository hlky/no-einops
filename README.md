# no-einops
Helper to unravel `einops.rearrange` into the underlying PyTorch operations

## Usage

### einops.rearrange

```python
import torch
import einops

x = torch.randn([2, 4, 8, 16, 16])
x_einops = einops.rearrange(x, "b c t h w -> (b t) c h w")
```


```
import torch
import no_einops

x = torch.randn([2, 4, 8, 16, 16])
no_einops.rearrange(x, "b c t h w -> (b t) c h w")
```

Output:

```
torch.permute(tensor, [0, 2, 1, 3, 4])
torch.reshape(tensor, [tensor.shape[0] * tensor.shape[1], tensor.shape[2], tensor.shape[3], tensor.shape[4]])
```

->

```python
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

```
