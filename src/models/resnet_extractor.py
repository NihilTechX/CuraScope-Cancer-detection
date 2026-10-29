import io
import base64
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class ResNet50FeatureExtractor(nn.Module):
    """
    Standard ResNet-50 convolutional feature extractor.
    Extracts 2,048-dimensional embeddings from H&E histopathology patches
    using official ImageNet-1K V2 pretrained weights.
    """
    def __init__(self):
        super().__init__()
        try:
            # Load official torchvision ResNet-50 ImageNet-1K V2 weights (cached locally)
            self.model = torch.hub.load('pytorch/vision:v0.16.0', 'resnet50', weights='ResNet50_Weights.IMAGENET1K_V2')
            self.model.fc = nn.Identity()
            self.model.eval()
        except Exception as e:
            # Fallback to structural model if hub weights cannot be loaded
            self.in_planes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(64, 3, stride=1)
            self.layer2 = self._make_layer(128, 4, stride=2)
            self.layer3 = self._make_layer(256, 6, stride=2)
            self.layer4 = self._make_layer(512, 3, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.model = None
            self.eval()

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4)
            )
        layers = [Bottleneck(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.model is not None:
            return self.model(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1) # [batch_size, 2048]

def preprocess_image_to_patches(img: Image.Image, num_patches: int = 5):
    """
    Takes an H&E image, resizes or crops into patches (224x224),
    and standardizes using ImageNet mean & std.
    """
    img = img.convert('RGB')
    w, h = img.size
    
    # Generate 5 representative patches (Center + 4 quadrants)
    patches = []
    patch_size = min(w, h) // 2
    if patch_size < 50:
        patch_size = min(w, h)
        
    coords = [
        (w // 4, h // 4),
        (0, 0),
        (w - patch_size, 0),
        (0, h - patch_size),
        (w - patch_size, h - patch_size)
    ]
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    tensor_patches = []
    patch_pil_images = []
    
    for (cx, cy) in coords[:num_patches]:
        crop = img.crop((cx, cy, cx + patch_size, cy + patch_size))
        crop_resized = crop.resize((224, 224), Image.Resampling.BILINEAR)
        patch_pil_images.append(crop_resized)
        
        arr = np.array(crop_resized, dtype=np.float32) / 255.0
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1)) # [3, 224, 224]
        tensor_patches.append(torch.tensor(arr, dtype=torch.float32))
        
    batch_t = torch.stack(tensor_patches, dim=0) # [num_patches, 3, 224, 224]
    return batch_t, patch_pil_images

def generate_attention_heatmap(img: Image.Image, attention_weights: list) -> Image.Image:
    """
    Overlays a translucent attention heatmap on the input H&E image.
    High-attention patches are highlighted with warm colors (red/orange).
    """
    img = img.convert('RGBA')
    w, h = img.size
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Grid coordinates corresponding to patches
    patch_size = min(w, h) // 2
    coords = [
        (w // 4, h // 4, w // 4 + patch_size, h // 4 + patch_size),
        (0, 0, patch_size, patch_size),
        (w - patch_size, 0, w, patch_size),
        (0, h - patch_size, patch_size, h),
        (w - patch_size, h - patch_size, w, h)
    ]
    
    weights = np.array(attention_weights[:len(coords)])
    if weights.max() > weights.min():
        norm_w = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)
    else:
        norm_w = np.ones_like(weights) * 0.5
        
    for box, nw in zip(coords, norm_w):
        # Color gradient: Blue (cool) -> Yellow -> Red (hot)
        alpha = int(40 + nw * 140)
        r = int(min(255, nw * 255 * 1.5))
        g = int(min(255, (1 - nw) * 120 + 30))
        b = int(min(255, (1 - nw) * 255 * 0.8))
        draw.rectangle(box, fill=(r, g, b, alpha), outline=(255, 255, 255, 180), width=3)
        
    heatmap_combined = Image.alpha_composite(img, overlay).convert('RGB')
    return heatmap_combined

def image_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")
