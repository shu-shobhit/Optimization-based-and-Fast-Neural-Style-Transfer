import torch
import torch.nn as nn
import torchvision.models as models

class VGG16(nn.Module):
    def __init__(self, pool_type="avg"):
        super().__init__()
        self.layer_list = []
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        if pool_type == "avg":
            for layer in self.vgg16:
                if isinstance(layer, torch.nn.modules.pooling.MaxPool2d):
                    self.layer_list.append(
                        nn.AvgPool2d(
                            kernel_size=2, stride=2, padding=0, ceil_mode=False
                        )
                    )
                else:
                    self.layer_list.append(layer)
            self.final_model = nn.Sequential(*self.layer_list)
        else:
            self.final_model = models.vgg16(
                weights=models.VGG16_Weights.IMAGENET1K_V1
            ).features

        for param in self.final_model.parameters():
            param.requires_grad = False

    def forward(self, input, layer_keys):
        out = {}
        last_layer_key = str(max([int(key) for key in layer_keys]))
        for name, layer in self.final_model.named_children():
            out[name] = layer(input)
            input = out[name]
            if name == last_layer_key:
                break

        final_output = {k: out[k] for k in layer_keys}
        return final_output