import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2DManual(nn.Module):
    """Manual Conv2D implemented using unfold/im2col with PyTorch ops so autograd works.

    Supports: padding, stride, bias. Does not support dilation/groups.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kh, kw = kernel_size, kernel_size
        else:
            kh, kw = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kh, kw)
        self.stride = stride
        self.padding = padding

        weight_shape = (out_channels, in_channels, kh, kw)
        self.weight = nn.Parameter(torch.randn(weight_shape) * 0.01)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        # x: (N, C, H, W)
        N, C, H, W = x.shape
        kh, kw = self.kernel_size
        # extract sliding local blocks -> (N, C*kh*kw, L)
        patches = F.unfold(x, kernel_size=(kh, kw), padding=self.padding, stride=self.stride)
        # patches_permute: (N, L, K) where K = C*kh*kw
        patches_permute = patches.permute(0, 2, 1)
        K = patches_permute.shape[2]
        # weight matrix shape (out_ch, K)
        weight_mat = self.weight.view(self.out_channels, -1)  # (out_ch, K)
        weight_t = weight_mat.t()  # (K, out_ch)
        # batch matmul: (N, L, K) @ (K, out_ch) -> (N, L, out_ch)
        out = patches_permute @ weight_t
        # add bias
        if self.bias is not None:
            out = out + self.bias.view(1, 1, -1)
        # reshape to (N, out_ch, L)
        out = out.permute(0, 2, 1)
        # compute output H_out and W_out
        H_out = (H + 2 * self.padding - kh) // self.stride + 1
        W_out = (W + 2 * self.padding - kw) // self.stride + 1
        out = out.contiguous().view(N, self.out_channels, H_out, W_out)
        return out


if __name__ == "__main__":
    # Quick smoke test
    conv = Conv2DManual(3, 8, 3, stride=1, padding=1)
    x = torch.randn(2, 3, 16, 16)
    y = conv(x)
    print('output shape', y.shape)
