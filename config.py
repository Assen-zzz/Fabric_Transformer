"""
集中管理所有超参数配置
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ModelConfig:
    """模型结构参数"""
    # 光谱输入
    spectral_height: int = 16
    spectral_width: int = 16
    spectral_channels: int = 1        # 灰度图
    spectral_patch_size: int = 4      # 16/4=4 → 4×4=16个patch
    
    # 视觉图像输入
    image_height: int = 224
    image_width: int = 224
    image_channels: int = 3           # RGB
    image_patch_size: int = 16        # 224/16=14 → 14×14=196个patch
    
    # 公共隐空间
    d_model: int = 256                # 注意力隐层维度
    n_heads: int = 4                  # 注意力头数
    attn_dropout: float = 0.1
    proj_dropout: float = 0.1
    
    # MLP分类头
    num_classes: int = 10             # 布料材质类别数
    mlp_hidden: int = 128             # 全连接隐藏层维度


@dataclass
class TrainConfig:
    """训练参数"""
    batch_size: int = 32
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0              # Windows下设为0避免dataloader报错
    
    # 模拟数据参数
    num_samples: int = 1000


# 全局单例配置
model_cfg = ModelConfig()
train_cfg = TrainConfig()