import logging
from typing import Optional, Any
from torch import nn
from model.ResNet import ResNet18, ResNet34, ResNet50,CustomResNet50

# Model registry dictionary
MODEL_REGISTRY = {
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50,
    'resnet50_ct': CustomResNet50,   #CustomResNet50 for dataset Kvasir
}

# 调试导入
for name, model_item in MODEL_REGISTRY.items():
    if not (callable(model_item) or isinstance(model_item, type)):
        logging.error(f"Model {name} is neither a callable function nor a class, got {type(model_item)}")
        raise TypeError(f"Model {name} must be a callable function or class, not {type(model_item)}")


class ModelManager:
    """Class to manage model instantiation."""

    @staticmethod
    def build_model(
            args: Any,
            model_name: Optional[str] = None,
            num_classes: Optional[int] = None,
            input_channels: Optional[int] = None
    ) -> nn.Module:
        """Builds and returns a model based on the provided name and parameters."""
        if model_name is None:
            model_name = getattr(args, 'model', None)
            if model_name is None:
                raise ValueError("Model name must be provided in args.model or as model_name parameter")
        if num_classes is None:
            num_classes = getattr(args.datasets, 'num_classes', 10)  # 默认值 10
        if input_channels is None:
            input_channels = getattr(args.datasets, 'input_channels', 3)  # 默认值 3

        if model_name not in MODEL_REGISTRY:
            raise NotImplementedError(
                f"Model {model_name} is not implemented. Available models: {list(MODEL_REGISTRY.keys())}")

        model_item = MODEL_REGISTRY[model_name]


        if model_name == 'HASSLEResnet':
            return model_item(
                args=args,
                num_classes=num_classes,
                input_channels=input_channels,
                resnet_size=8,
                scaling=4,
                save_activations=False,
                group_norm_num_groups=None,
                freeze_bn=False,
                freeze_bn_affine=False
            )
        else:
            # 其他模型使用标准参数
            return model_item(args, num_classes, input_channels)


# Function to get model for external use
def get_model(
        args: Any,
        model_name: Optional[str] = None,
        num_classes: Optional[int] = None,
        input_channels: Optional[int] = None
) -> nn.Module:
    """External interface to build a model."""
    return ModelManager.build_model(args, model_name, num_classes, input_channels)