import torchvision.models
from torch import nn


class PaperModel(nn.Module):
    def __init__(self, dropout: float = 0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=dropout),
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=dropout),
            nn.Conv2d(96, 256, kernel_size=3, padding=1),
        )
        self.projector = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        """
        :param x: Batch of images (B, C, H, W)
        :return: The predicted logits
        """
        x = self.encoder(x)
        x = x.mean(dim=[2, 3])
        return self.projector(x)

class PaperModel1D(nn.Module):
    def __init__(self, dropout: float = 0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=dropout),
            nn.Conv1d(48, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=dropout),
            nn.Conv1d(96, 256, kernel_size=3, padding=1),
        )
        self.projector = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        """
        :param x: Batch of TS (B, T, 5)
        :return: The predicted logits
        """
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = x.mean(dim=[2])
        return self.projector(x)

class ResLSTM(nn.Module):
    def __init__(self, dropout: float = 0.25):
        super().__init__()
        hidden_size = 64
        self.l1 = nn.LSTM(5, hidden_size, batch_first=True)
        self.l2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.l3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.d = nn.Dropout(p=dropout)
        self.p = nn.Linear(hidden_size, 256)
        self.relu = nn.ReLU()
        self.projector = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 2),
        )
    def forward(self, x):
        z, _ = self.l1(x)
        z2, _ = self.l2(z)
        z = z + self.d(z2)
        z3, _ = self.l3(z)
        z = self.d(z3) + z
        z = self.p(z)
        z = z.mean(dim=[1])
        return self.projector(z)

def freeze_weights(model: torchvision.models.VisionTransformer):
    for p in model.parameters():
        p.requires_grad = False

def from_name(config, annotation_type: str = "default"):
    name = config["model"]["name"]
    n_class = 3 if annotation_type == "default" else 2
    if name == "EfficientNetV2_s":
        model = torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        out_feat = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=config["model"]["dropout"], inplace=True),
            nn.Linear(out_feat, n_class),
        )
    elif name == "EfficientNetV2_m":
        model = torchvision.models.efficientnet_v2_m(weights=torchvision.models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        out_feat = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=config["model"]["dropout"], inplace=True),
            nn.Linear(out_feat, n_class),
        )
    elif name == "paper":
        model = PaperModel(dropout=config["model"]["dropout"])
    elif name == "paper1D":
        model = PaperModel1D(dropout=config["model"]["dropout"])
    elif name == "ResLSTM":
        model = ResLSTM(dropout=config["model"]["dropout"])
    elif name == "VIT_b_16":
        model: torchvision.models.VisionTransformer = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
        freeze_weights(model)
        model.heads[0] = nn.Linear(768, n_class)
    elif name == "VIT_b_32":
        model: torchvision.models.VisionTransformer = torchvision.models.vit_b_32(weights=torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1)
        freeze_weights(model)
        model.heads[0] = nn.Linear(768, n_class)
    elif name == "ConvNeXt_tiny":
        model: torchvision.models.ConvNeXt = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        lastconv_output_channels = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(lastconv_output_channels, n_class)
    elif name == "ConvNeXt_small":
        model: torchvision.models.ConvNeXt = torchvision.models.convnext_small(weights=torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
        lastconv_output_channels = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(lastconv_output_channels, n_class)

    else:
        raise ValueError(f"Invalid model name: {name}!")

    return model

if __name__ == "__main__":
    from torchinfo import summary
    model = ResLSTM()

    summary(model, input_size=(1024, 20, 5))