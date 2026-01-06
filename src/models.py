import torch.nn as nn
import timm
import torch.nn.functional as F

from .attention import (
    CBAM, SEBlock, ECABlock,
    ChannelAttention, SpatialAttention, SelfAttention
)

# ====================== MODEL DEFINITIONS ======================

class ConvNeXtOnly(nn.Module):
    """Single ConvNeXt baseline without attention"""
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = timm.create_model('convnext_small', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)

class ConvNeXt_CBAM(nn.Module):
    """Single branch with CBAM attention"""
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = timm.create_model('convnext_small', pretrained=True, features_only=True)
        self.cbam = CBAM(768)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.4), nn.Linear(768, num_classes)
        )

    def forward(self, x):
        feat = self.cbam(self.backbone(x)[-1])
        return self.head(feat)

class ConvNeXt_SE(nn.Module):
    """Single branch with SE attention"""
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = timm.create_model('convnext_small', pretrained=True, features_only=True)
        self.se = SEBlock(768)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.4), nn.Linear(768, num_classes)
        )

    def forward(self, x):
        feat = self.se(self.backbone(x)[-1])
        return self.head(feat)

class DualAttention_CBAM_SE(nn.Module):
    """Dual-branch with complementary attentions (CBAM + SE)"""
    def __init__(self, num_classes=3):
        super().__init__()

        self.branch_a = timm.create_model('convnext_small', pretrained=True, features_only=True)
        self.cbam = CBAM(768)

        self.branch_b = timm.create_model('convnext_small', pretrained=True, features_only=True)
        self.se = SEBlock(768)

        self.fusion = nn.Sequential(
            nn.Conv2d(768 * 2, 768, 1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.4), nn.Linear(768, num_classes)
        )

    def forward(self, x):
        feat_a = self.cbam(self.branch_a(x)[-1])
        feat_b = self.se(self.branch_b(x)[-1])

        fused = torch.cat([feat_a, feat_b], dim=1)
        fused = self.fusion(fused)

        return self.head(fused)

class DualAttention_CBAM_ECA(nn.Module):
    """Dual-branch with CBAM + ECA"""
    def __init__(self, num_classes=3):
        super().__init__()

        self.branch_a = timm.create_model('convnext_small', pretrained=True, features_only=True)
        self.cbam = CBAM(768)

        self.branch_b = timm.create_model('convnext_small', pretrained=True, features_only=True)
        self.eca = ECABlock(768)

        self.fusion = nn.Sequential(
            nn.Conv2d(768 * 2, 768, 1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.4), nn.Linear(768, num_classes)
        )

    def forward(self, x):
        feat_a = self.cbam(self.branch_a(x)[-1])
        feat_b = self.eca(self.branch_b(x)[-1])

        fused = torch.cat([feat_a, feat_b], dim=1)
        fused = self.fusion(fused)

        return self.head(fused)

class DualAttention_Channel_Spatial(nn.Module):
    """Dual-branch with pure Channel + pure Spatial attention"""
    def __init__(self, num_classes=3):
        super().__init__()

        self.branch_a = timm.create_model('convnext_small', pretrained=True, features_only=True)
        self.channel_attn = ChannelAttention(768)

        self.branch_b = timm.create_model('convnext_small', pretrained=True, features_only=True)
        self.spatial_attn = SpatialAttention()

        self.fusion = nn.Sequential(
            nn.Conv2d(768 * 2, 768, 1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.4), nn.Linear(768, num_classes)
        )

    def forward(self, x):
        feat_a = self.channel_attn(self.branch_a(x)[-1])
        feat_b = self.spatial_attn(self.branch_b(x)[-1])

        fused = torch.cat([feat_a, feat_b], dim=1)
        fused = self.fusion(fused)

        return self.head(fused)

class DualAttention_Self_CBAM(nn.Module):
    """Dual-branch with Self-Attention + CBAM"""
    def __init__(self, num_classes=3):
        super().__init__()

        self.branch_a = timm.create_model('convnext_small', pretrained=True, features_only=True)
        self.self_attn = SelfAttention(768)

        self.branch_b = timm.create_model('convnext_small', pretrained=True, features_only=True)
        self.cbam = CBAM(768)

        self.fusion = nn.Sequential(
            nn.Conv2d(768 * 2, 768, 1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.4), nn.Linear(768, num_classes)
        )

    def forward(self, x):
        feat_a = self.self_attn(self.branch_a(x)[-1])
        feat_b = self.cbam(self.branch_b(x)[-1])

        fused = torch.cat([feat_a, feat_b], dim=1)
        fused = self.fusion(fused)

        return self.head(fused)

class DualBranch_NoAttention(nn.Module):
    """Two ConvNeXt branches without any attention"""
    def __init__(self, num_classes=3):
        super().__init__()

        self.branch_a = timm.create_model('convnext_small', pretrained=True, features_only=True)
        self.branch_b = timm.create_model('convnext_small', pretrained=True, features_only=True)

        self.fusion = nn.Sequential(
            nn.Conv2d(768 * 2, 768, 1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.4), nn.Linear(768, num_classes)
        )

    def forward(self, x):
        feat_a = self.branch_a(x)[-1]
        feat_b = self.branch_b(x)[-1]

        fused = torch.cat([feat_a, feat_b], dim=1)
        fused = self.fusion(fused)

        return self.head(fused)

MODEL_REGISTRY = {
    'Study1_ConvNeXtOnly': ConvNeXtOnly,
    'Study2_ConvNeXt_CBAM': ConvNeXt_CBAM,
    'Study3_ConvNeXt_SE': ConvNeXt_SE,
    'Study4_DualAttn_CBAM_SE': DualAttention_CBAM_SE,
    'Study5_DualAttn_CBAM_ECA': DualAttention_CBAM_ECA,
    'Study6_DualAttn_Channel_Spatial': DualAttention_Channel_Spatial,
    'Study7_DualAttn_Self_CBAM': DualAttention_Self_CBAM,
    'Study8_DualBranch_NoAttn': DualBranch_NoAttention,
}

print(f"\n📊 Total studies: {len(MODEL_REGISTRY)}")
print(f"\n⭐ PROPOSED MODELS:")
print(f"   Study4: CBAM + SE (Complementary channel attentions)")
print(f"   Study6: Channel + Spatial (Explicit What/Where separation) ⭐⭐ BEST")
print(f"   Study7: Self-Attention + CBAM (Global + Local)")
