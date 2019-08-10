from torchvision import models
import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    '''
    The main modules architecture. Based on a resnet34 backbone with Nearest neighbour upsampling
    layers. Input is a (b, c, h, w) FloatTensor. output is (b, embedding_dim, h, w) FloatTensor
    with the embedded pixels. To get equal input and output size, the input dimensions (h,w) should
    be a multiple of 2^5=32.

    :param embedding_dim - the number of output channels. i.e. the length of the embedded
    pixel vector.
    :param context - boolean, If True, a context layer is added to the model. See ContextModule for more.
    '''
    def __init__(self, embedding_dim):
        super(FeatureExtractor, self).__init__()
        self.embedding_dim = embedding_dim
        self.resnet = models.resnet101(True)  # can be resnet34 or 50
        for param in self.resnet.parameters():   # Freeze resnet layers
            param.requires_grad = False

        self.upsample1 = UpsamplingBlock(2048, 1024, skip=True)
        self.upsample2 = UpsamplingBlock(1024, 512, skip=True)
        self.upsample3 = UpsamplingBlock(512, 256, skip=True)
        self.upsample4 = UpsamplingBlock(256, 64, skip=True)
        self.upsample5 = UpsamplingBlock(64, 64, skip=False)
        self.finalConv = nn.Sequential(nn.Conv2d(64, self.embedding_dim, 1, 1),
                                       nn.ReLU(),
                                       nn.BatchNorm2d(self.embedding_dim))

    def forward(self, x):
        outputs = {}
        for name, module in list(self.resnet.named_children())[:-2]:
            x = module(x)
            outputs[name] = x
        features = outputs['layer4']  # Resnet output before final avgpool and fc layer
        # outputs['layer4'] is 2048x7x7
        # outputs['layer3'] is 1024x14x14
        # outputs['layer2'] is 512x28x28
        # outputs['layer1'] is 256x56x56
        # outputs['relu'] is 64x112x112
        # features here is 2048x7x7
        features = self.upsample1(features, outputs['layer3'])
        # features here is 1024x14x14
        features = self.upsample2(features, outputs['layer2'])
        # features here 512x28x28
        features = self.upsample3(features, outputs['layer1'])
        # features here 256x56x56
        features = self.upsample4(features, outputs['relu'])
        # features here 64x112x112
        features = self.upsample5(features)
        # features here 64x224x224
        features = self.finalConv(features)
        # features here embeddingDimx224x224

        return features


class UpsamplingBlock(nn.Module):
    '''
    For the up-sampling I chose Nearest neighbour over deconvolution, to avoid artifacts in the output.
    If skip is set to True, the input to the forward pass must include skip input - i.e. the equivalent sized output
    of the downsampling backbone (here resnet).

    :param channels_in - number of filters/channels in the input.
    :param channels_in - number of filters/channels in the output.
    :param skip - whether or not to use skip input. recommended to set to true.

    '''
    def __init__(self, channels_in, channels_out, skip=False):
        super(UpsamplingBlock, self).__init__()
        #self.upsamplingLayer = nn.ConvTranspose2d(channels_in, channels_out, kernel_size=2, stride=2)
        self.upsamplingLayer = nn.Sequential(nn.Upsample(scale_factor=2),
                                             nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=1),
                                             nn.ReLU(),
                                             nn.BatchNorm2d(channels_out))

        if skip:
            self.conv1 = nn.Conv2d(2*channels_out, channels_out, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1)

        self.convLayer1 = nn.Sequential(self.conv1,
                                        nn.ReLU(),
                                        nn.BatchNorm2d(channels_out))

        self.convLayer2 = nn.Sequential(self.conv2,
                                        nn.ReLU(),
                                        nn.BatchNorm2d(channels_out))

    def forward(self, x, skip_input=None):
        x = self.upsamplingLayer(x)
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)
        x = self.convLayer1(x)
        x = self.convLayer2(x)
        return x
