import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm

norm_layer = nn.InstanceNorm2d


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      norm_layer(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      norm_layer(in_features)
                      ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(Generator, self).__init__()

        # Initial convolution block
        model0 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, 64, 7),
                  norm_layer(64),
                  nn.ReLU(inplace=True)]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model1 += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                       norm_layer(out_features),
                       nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2
        self.model1 = nn.Sequential(*model1)

        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)

        # Upsampling
        model3 = []
        out_features = in_features // 2
        for _ in range(2):
            model3 += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                       norm_layer(out_features),
                       nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        model4 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]

        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out


def main():
    config = OmegaConf.load('./configs/preprocess.yaml')
    data_dir = Path(config.data_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Generator(3, 1, 3)
    model.load_state_dict(torch.load(config.model_path, map_location=torch.device('cpu')))
    model.eval().to(device)
    transform = transforms.Compose([transforms.ToTensor()])

    def predict(input_img):
        input_img = transform(input_img)
        input_img = torch.unsqueeze(input_img, 0)
        input_img = input_img.to(device)
        b, c, h, w = input_img.size()
        h, w = (h // 4) * 4, (w // 4) * 4
        input_img = input_img[:, :, :h, :w]

        with torch.no_grad():
            drawing = model(input_img)[0].detach().cpu()
            drawing = drawing.repeat_interleave(3, dim=0)

        out = torch.cat([input_img.cpu()[0], drawing], dim=-1)

        out = transforms.ToPILImage()(out)
        return out

    files = list(sorted(data_dir.glob('*')))
    for p in tqdm(files, total=len(files)):
        input_img = Image.open(p)
        out = predict(input_img)
        out.save(output_dir / f'gray_{p.name}')


if __name__ == '__main__':
    main()
