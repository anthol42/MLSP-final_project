import torchvision.models
from torch import nn

def from_name(config):
    name = config["model"]["name"]
    if name == "EfficientNetV2_s":
        model = torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        out_feat = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=config["model"]["dropout"], inplace=True),
            nn.Linear(out_feat, 3),
        )
    elif name == "EfficientNetV2_m":
        model = torchvision.models.efficientnet_v2_m(weights=torchvision.models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        out_feat = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=config["model"]["dropout"], inplace=True),
            nn.Linear(out_feat, 3),
        )

    elif name == "VIT_b_16":
        model: torchvision.models.VisionTransformer = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads[0] = nn.Linear(768, 3)
    elif name == "VIT_b_32":
        model: torchvision.models.VisionTransformer = torchvision.models.vit_b_32(weights=torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1)
        model.heads[0] = nn.Linear(768, 3)
    elif name == "ConvNeXt_tiny":
        model: torchvision.models.ConvNeXt = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        lastconv_output_channels = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(lastconv_output_channels, 3)
    elif name == "ConvNeXt_small":
        model: torchvision.models.ConvNeXt = torchvision.models.convnext_small(weights=torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
        lastconv_output_channels = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(lastconv_output_channels, 3)

    else:
        raise ValueError(f"Invalid model name: {name}!")

    return model
