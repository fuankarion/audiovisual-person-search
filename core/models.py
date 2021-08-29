
import torch
import torch.nn as nn
import torch.nn.parameter
from core.network_blocks import conv3x3, conv1x1, BasicBlock, Bottleneck

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def _load_weights_into_two_stream_resnet(model, rgb_stack_size, arch, progress):
    resnet_state_dict = load_state_dict_from_url(model_urls[arch],
                                                 progress=progress)

    own_state = model.state_dict()
    for name, param in resnet_state_dict.items():
        # this cases are not mutually exlcusive
        if 'v_'+name in own_state:
            own_state['v_'+name].copy_(param)
        if 'a_'+name in own_state:
            own_state['a_'+name].copy_(param)
        if 'v_'+name not in own_state and 'a_'+name not in own_state:
            print('No assignation for ', name)

    conv1_weights = resnet_state_dict['conv1.weight']
    tupleWs = tuple(conv1_weights for i in range(rgb_stack_size))
    concatWs = torch.cat(tupleWs, dim=1)
    own_state['video_conv1.weight'].copy_(concatWs)

    avgWs = torch.mean(conv1_weights, dim=1, keepdim=True)
    own_state['audio_conv1.weight'].copy_(avgWs)

    print('loaded ws from resnet')
    return model


class ThreeStreamResNetAudio(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ThreeStreamResNetAudio, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # Shared Stream
        self.inplanes = 64
        self.conv1_audio = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_stream(self, x):
        x = self.conv1_audio(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        return x

    def forward(self, a, p, n):
        a = self._forward_stream(a)
        p = self._forward_stream(p)
        n = self._forward_stream(n)

        return a, p, n


class ThreeStreamResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ThreeStreamResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # Shared Stream
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_stream(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        return x

    def forward(self, a, p, n):
        a = self._forward_stream(a)
        p = self._forward_stream(p)
        n = self._forward_stream(n)

        return a, p, n


class ThreeStreamResNetAudioVideo(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ThreeStreamResNetAudioVideo, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # Audio Stream
        self.inplanes = 64
        self.audio_conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.a_bn1 = norm_layer(self.inplanes)

        self.a_layer1 = self._make_layer(block, 64, layers[0])
        self.a_layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.a_layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.a_layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # Video Stream
        self.inplanes = 64
        self.video_conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.v_bn1 = norm_layer(self.inplanes)

        self.v_layer1 = self._make_layer(block, 64, layers[0])
        self.v_layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.v_layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.v_layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # Shared
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        self.fc_shared = nn.Linear(512 * 2, 512)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_audio_stream(self, x):
        x = self.audio_conv1(x)
        x = self.a_bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.a_layer1(x)
        x = self.a_layer2(x)
        x = self.a_layer3(x)
        x = self.a_layer4(x)
        x = self.avgpool(x)

        return x

    def _forward_video_stream(self, x):
        x = self.video_conv1(x)
        x = self.v_bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.v_layer1(x)
        x = self.v_layer2(x)
        x = self.v_layer3(x)
        x = self.v_layer4(x)
        x = self.avgpool(x)

        return x

    def forward(self, anchor_audio_data, positive_audio_data, negative_audio_data,
                anchor_video_data, positive_video_data, negative_video_data):
        a_a = self._forward_audio_stream(anchor_audio_data)
        p_a = self._forward_audio_stream(positive_audio_data)
        n_a = self._forward_audio_stream(negative_audio_data)

        a_v = self._forward_video_stream(anchor_video_data)
        p_v = self._forward_video_stream(positive_video_data)
        n_v = self._forward_video_stream(negative_video_data)

        a_a = a_a.reshape(a_a.size(0), -1)
        a_v = a_v.reshape(a_v.size(0), -1)

        p_a = p_a.reshape(p_a.size(0), -1)
        p_v = p_v.reshape(p_v.size(0), -1)

        n_a = n_a.reshape(n_a.size(0), -1)
        n_v = n_v.reshape(n_v.size(0), -1)

        a = self.fc_shared(torch.cat([a_a, a_v], dim=1))
        p = self.fc_shared(torch.cat([p_a, p_v], dim=1))
        n = self.fc_shared(torch.cat([n_a, n_v], dim=1))

        return a, p, n


def _triplet_reid_resnet_audio(arch, block, layers, pretrained, progress, **kwargs):
    model = ThreeStreamResNetAudio(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)

        own_state = model.state_dict()
        conv1_weights = state_dict['conv1.weight']
        avgWs = torch.mean(conv1_weights, dim=1, keepdim=True)
        own_state['conv1_audio.weight'].copy_(avgWs)
    return model


def _triplet_reid_resnet_visual(arch, block, layers, pretrained, progress, **kwargs):
    model = ThreeStreamResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def _triplet_reid_resnet_audio_video(arch, block, layers, pretrained, progress, **kwargs):
    model = ThreeStreamResNetAudioVideo(block, layers, **kwargs)
    if pretrained:
        model = _load_weights_into_two_stream_resnet(model, 1, arch, progress)
    return model


# Callable Methods
def triplet_reid_resnet_18_audio(pretrained=True, progress=True, **kwargs):
    return _triplet_reid_resnet_audio('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                                      **kwargs)


def triplet_reid_resnet_18_visual(pretrained=True, progress=True, **kwargs):
    return _triplet_reid_resnet_visual('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                                       **kwargs)


def triplet_reid_resnet_18_audio_video(pretrained=True, progress=True, **kwargs):
    return _triplet_reid_resnet_audio_video('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                                            **kwargs)
