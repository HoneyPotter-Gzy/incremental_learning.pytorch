from collections import OrderedDict
from torch import nn

class IntermediateLayerGetter(nn.ModuleDict):
    """ get the output of certain layers """

    def __init__ (self, model, return_layers):
        # 判断传入的return_layers是否存在于model中
        if not set(return_layers).issubset(
                [name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = { k: v for k, v in return_layers.items() }  # 构造dict
        layers = OrderedDict()
        # 将要从model中获取信息的最后一层之前的模块全部复制下来
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(
            layers)  # 将所需的网络层通过继承的方式保存下来
        self.return_layers = orig_return_layers

    def forward (self, x):
        out = OrderedDict()
        # 将所需的值以k,v的形式保存到out中
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out

import torchvision
import torch

model = torchvision.models.resnet18()
return_layers = { 'avgpool': 'feature_1', 'layer2': 'feature_2' }
backbone = IntermediateLayerGetter(model, return_layers)

backbone.eval()
x = torch.randn(1, 3, 224, 224)
out = backbone(x)
print(out['feature_1'].shape, out['feature_2'].shape)
