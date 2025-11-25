import torch
import torch.nn as nn
import torch.nn.functional as F
from conv_manual import Conv2DManual


def compare_conv(in_ch=3, out_ch=4, H=8, W=8, kh=3, kw=3, stride=1, padding=1, dtype=torch.float32, device='cpu'):
    torch.manual_seed(0)
    # input
    x = torch.randn(2, in_ch, H, W, dtype=dtype, device=device, requires_grad=True)

    # reference conv
    conv_ref = nn.Conv2d(in_ch, out_ch, (kh, kw), stride=stride, padding=padding, bias=True).to(device).to(dtype)
    # manual conv
    conv_man = Conv2DManual(in_ch, out_ch, (kh, kw), stride=stride, padding=padding, bias=True).to(device).to(dtype)

    # copy weights and bias
    with torch.no_grad():
        conv_man.weight.copy_(conv_ref.weight)
        conv_man.bias.copy_(conv_ref.bias)

    # forward
    y_ref = conv_ref(x)
    y_man = conv_man(x)

    # compare forward outputs
    forward_close = torch.allclose(y_ref, y_man, atol=1e-6)
    print('Forward close:', forward_close)
    if not forward_close:
        print('max abs diff:', (y_ref - y_man).abs().max().item())

    # backward: simple scalar loss
    loss_ref = y_ref.sum()
    loss_man = y_man.sum()

    loss_ref.backward()
    loss_man.backward()

    # compare gradients of weights
    grad_w_close = torch.allclose(conv_ref.weight.grad, conv_man.weight.grad, atol=1e-6)
    grad_x_close = torch.allclose(x.grad, x.grad, atol=1e-6)  # trivial but keep
    print('Weight grad close:', grad_w_close)

    # compare bias grads
    bias_grad_close = torch.allclose(conv_ref.bias.grad, conv_man.bias.grad, atol=1e-6)
    print('Bias grad close:', bias_grad_close)

    if forward_close and grad_w_close and bias_grad_close:
        print('TEST PASSED')
        return 0
    else:
        print('TEST FAILED')
        return 1


if __name__ == '__main__':
    import sys
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rc = compare_conv(in_ch=3, out_ch=5, H=10, W=10, kh=3, kw=3, stride=1, padding=1, device=device)
    sys.exit(rc)
