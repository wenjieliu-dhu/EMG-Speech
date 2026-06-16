import torch
import torch.nn as nn
import torch.nn.functional as F

class get_mask_conv(nn.Module):
    def __init__(self, input_conv, gpu_ids):
        super().__init__()
        self.ori_weight = input_conv.weight.detach()
        shape = torch.ones((self.ori_weight.size(0), self.ori_weight.size(1), 1)).cuda(gpu_ids)
        self.weight_mask = nn.Parameter(data=torch.ones_like(shape) * 6, requires_grad=True).cuda(gpu_ids)
        if input_conv.bias is not None:
            self.ori_bias = input_conv.bias.detach()
            self.bias_mask = nn.Parameter(data=torch.ones_like(self.ori_bias) * 6, requires_grad=True).cuda(gpu_ids)
        else:
            self.ori_bias = None
            self.bias_mask = None
        self.gpu_ids = gpu_ids
        self.sig = nn.ReLU6()
        self.padding = input_conv.padding
        self.stride = input_conv.stride
        self.dilation = input_conv.dilation

    def forward(self, x):
        res = list()
        for i in range(x.size(0)):
            new_weight = self.ori_weight * self.sig(
                self.weight_mask.repeat(1, 1, self.ori_weight.size(2))) / 6

            if self.ori_bias is not None:
                new_bias = self.ori_bias * self.sig(self.bias_mask) / 6

                res.append(F.conv1d(
                    x[i].unsqueeze(0), new_weight, new_bias,
                    padding=self.padding, stride=self.stride, dilation=self.dilation
                ).squeeze(0))
            else:
                res.append(F.conv1d(
                    x[i].unsqueeze(0), new_weight,
                    padding=self.padding, stride=self.stride, dilation=self.dilation
                ).squeeze(0))

        middle_output = torch.stack(res, dim=0)
        return middle_output

    def get_weight_mask_loss(self, x):
        loss = self.sig(x) * self.sig(x) / 36
        return loss.sum() / torch.numel(x)

    def get_rate(self):
        m = self.sig(self.weight_mask) / 6
        return m.sum() / torch.numel(m)


class get_fingerprint(nn.Module):
    def __init__(self, input_masked_conv, finger_dim):
        super().__init__()
        self.sig = nn.ReLU6()
        self.gpu_ids = input_masked_conv.gpu_ids
        self.finger_dim = finger_dim
        self.w_mask = input_masked_conv.weight_mask.detach()
        self.ori_weight = input_masked_conv.ori_weight.detach()
        if input_masked_conv.bias_mask is None:
            self.b_mask = None
            self.ori_bias = None
            self.f_bias = None
        else:
            self.b_mask = input_masked_conv.bias_mask.detach()
            self.ori_bias = input_masked_conv.ori_bias.detach()
        weight_shape = self.ori_weight.size()
        self.style_weight = ApplyStyle(weight_shape[1] * weight_shape[2], finger_dim)

        self.f_weight = nn.Parameter(
            data=torch.ones(1, weight_shape[0], weight_shape[1], weight_shape[2]).cuda(self.gpu_ids) * self.ori_weight,
            requires_grad=True
        ).cuda(self.gpu_ids)

        self.f_w_weight = (6 - self.sig(self.w_mask.repeat(1, 1, self.ori_weight.size(2)))) / 6

        if self.ori_bias is not None:
            self.f_bias = nn.Parameter(
                data=torch.ones(1, self.ori_bias.size(0)).cuda(self.gpu_ids) * self.ori_bias,
                requires_grad=True
            ).cuda(self.gpu_ids)
            self.f_w_bias = (6 - self.sig(self.b_mask)) / 6

        self.padding = input_masked_conv.padding
        self.stride = input_masked_conv.stride
        self.dilation = input_masked_conv.dilation
        self.mean_ori_weight = self.ori_weight.mean()
        self.std_ori_weight = self.ori_weight.std()

    def set_mask(self):
        self.f_w_weight = (6 - self.sig(self.w_mask.repeat(1, 1, self.ori_weight.size(2)))) / 6
        if self.b_mask is not None:
            self.f_w_bias = (6 - self.sig(self.b_mask)) / 6

    def forward(self, x):
        if isinstance(x, (list, tuple)) and len(x) == 2:
            y, fingerprint = x
        else:
            raise ValueError(
                f"Expected input to be a list or tuple of length 2, but got {type(x)} with length {len(x)}")

        if isinstance(y, (list, tuple)):
            y, fingerprint = y

        res = []
        s_weight = self.style_weight(self.f_weight.repeat(fingerprint.size(0), 1, 1, 1), fingerprint)

        if self.ori_bias is not None:
            s_bias = self.f_bias.repeat(fingerprint.size(0), 1)
            for i in range(y.size(0)):
                new_weight = (1 - self.f_w_weight) * self.ori_weight + self.f_w_weight * s_weight[i]
                std_weight = torch.std(new_weight)
                mean_weight = torch.mean(new_weight)
                new_weight = ((new_weight - mean_weight) / std_weight * self.std_ori_weight + self.mean_ori_weight)
                new_weight = (1 - self.f_w_weight) * self.ori_weight + self.f_w_weight * new_weight
                new_bias = (1 - self.f_w_bias) * self.ori_bias + self.f_w_bias * s_bias[i]
                res.append(F.conv1d(y[i].unsqueeze(0), new_weight, new_bias,
                                    padding=self.padding, stride=self.stride,
                                    dilation=self.dilation).squeeze(0))
        else:
            for i in range(y.size(0)):
                new_weight = (1 - self.f_w_weight) * self.ori_weight + self.f_w_weight * s_weight[i]
                std_weight = torch.std(new_weight)
                mean_weight = torch.mean(new_weight)
                new_weight = ((new_weight - mean_weight) / std_weight * self.std_ori_weight + self.mean_ori_weight)
                new_weight = (1 - self.f_w_weight) * self.ori_weight + self.f_w_weight * new_weight
                res.append(F.conv1d(y[i].unsqueeze(0), new_weight,
                                    padding=self.padding, stride=self.stride,
                                    dilation=self.dilation).squeeze(0))

        middle_output = torch.stack(res, dim=0)
        return middle_output

    def get_weight_mask_loss(self, x):
        loss = self.sig(x) * self.sig(x)
        return loss.sum() / torch.numel(x)

    def get_rate(self):
        m = (6 - self.sig(self.w_mask)) / 6
        return m.sum() / torch.numel(m)

    def hook_fn(self, module, input, fingerprint):
        input_tuple = tuple(input, fingerprint)
        return input_tuple



class ApplyStyle(nn.Module):
    def __init__(self, channels, fin_dim):
        super(ApplyStyle, self).__init__()
        self.channels = channels
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=fin_dim, out_features=fin_dim, bias=False),
            nn.LayerNorm(fin_dim),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=fin_dim, out_features=channels, bias=False),
            nn.LayerNorm(channels),
            nn.LeakyReLU()
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=channels, out_features=channels * 2, bias=False),
        )

    def forward(self, x, latent):
        kernel_size = x.size(3)
        x = x.contiguous().view(x.size(0),x.size(1),-1)
        style = self.fc1(latent)
        style = self.fc2(style)
        style = self.linear(style)
        shape = [-1, 2, 1, x.size(2)]
        style = style.view(shape)
        x = x * (style[:, 0] * 1.+1) + style[:, 1] * 1
        x = x.view(x.size(0),x.size(1),-1,kernel_size)
        return x


