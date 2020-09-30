from torch import nn
from mean_field import *
from functools import partial

__all__ = ['MobileNetV2', 'mobilenet_v2']

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, conv_layer=None, norm_layer=None):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            conv_layer(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, conv_layer=None, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, conv_layer=conv_layer, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, conv_layer=conv_layer, norm_layer=norm_layer),
            # pw-linear
            conv_layer(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self,
                 args,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(MobileNetV2, self).__init__()
        self.bayes = args.bayes
        self.fc_bayes = args.fc_bayes
        self.num_classes = num_classes

        if block is None:
            block = InvertedResidual

        if self.bayes is None:
            conv_layer, linear_layer, norm_layer = nn.Conv2d, nn.Linear, nn.BatchNorm2d
        elif self.bayes == 'mf':
            conv_layer, linear_layer, norm_layer = partial(BayesConv2dMF, args.single_eps, args.local_reparam), partial(BayesLinearMF, args.single_eps, args.local_reparam) if self.fc_bayes else nn.Linear, partial(BayesBatchNorm2dMF, args.single_eps)
        else:
            raise NotImplementedError

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, conv_layer=conv_layer, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, conv_layer=conv_layer, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, conv_layer=conv_layer, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(args.dropout_rate),
            linear_layer(self.last_channel, num_classes),
        )


        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                assert m.bias is None # nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, BayesConv2dMF):
                nn.init.kaiming_normal_(m.weight_mu, mode='fan_out')
                m.weight_log_sigma.data.uniform_(args.log_sigma_init_range[0], args.log_sigma_init_range[1])
            elif isinstance(m, BayesBatchNorm2dMF):
                nn.init.constant_(m.weight_mu, 1)
                nn.init.constant_(m.bias_mu, 0)
                m.weight_log_sigma.data.uniform_(args.log_sigma_init_range[0], args.log_sigma_init_range[1])
                m.bias_log_sigma.data.uniform_(args.log_sigma_init_range[0], args.log_sigma_init_range[1])
            elif isinstance(m, BayesLinearMF):
                nn.init.normal_(m.weight_mu, 0, 0.01)
                nn.init.zeros_(m.bias_mu)
                m.weight_log_sigma.data.uniform_(args.log_sigma_init_range[0], args.log_sigma_init_range[1])
                m.bias_log_sigma.data.uniform_(args.log_sigma_init_range[0], args.log_sigma_init_range[1])

    def _forward_impl(self, x, return_f=False, return_both=False):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        if return_f:
            return x
        if return_both:
            return x, self.classifier(x)
        else:
            return self.classifier(x)

    def forward(self, x, return_f=False, return_both=False):
        return self._forward_impl(x, return_f, return_both)


def mobilenet_v2(args, pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    assert(not pretrained)
    model = MobileNetV2(args, **kwargs)
    return model