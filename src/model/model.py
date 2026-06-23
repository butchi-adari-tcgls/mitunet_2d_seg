import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_mitunet(
    encoder_name: str = "mit_b4",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
    decoder_attention_type: str = "scse",
    checkpoint_path: str | None = None,
    device: str | torch.device = DEVICE,
) -> torch.nn.Module:
    """Build the hybrid MitUNet architecture.

    This model uses a Segformer encoder from segmentation_models_pytorch
    and a Unet decoder with scSE attention.
    """
    segformer_full = smp.Segformer(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
    )

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        classes=classes,
        decoder_attention_type=decoder_attention_type,
    )

    model.encoder = segformer_full.encoder
    model = model.to(device)

    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)

    return model


# def replace_output_head(model: torch.nn.Module, classes: int = 1, activation: str | None = None) -> torch.nn.Module:
#     """Replace the final segmentation head of the model.

#     This is useful when you want to update the number of output channels/classes.
#     """
#     if not hasattr(model, "segmentation_head"):
#         raise AttributeError("Model does not have a segmentation_head attribute")

#     in_channels = model.segmentation_head[0].in_channels if isinstance(model.segmentation_head, nn.Sequential) else model.segmentation_head.in_channels

#     activation_layer = nn.Identity()
#     if activation == "sigmoid":
#         activation_layer = nn.Sigmoid()
#     elif activation == "softmax":
#         activation_layer = nn.Softmax(dim=1)

#     model.segmentation_head = nn.Sequential(
#         nn.Conv2d(in_channels, classes, kernel_size=1),
#         activation_layer,
#     )

#     return model


def replace_output_head(
    model: torch.nn.Module,
    classes: int = 1,
    activation: str | None = None,
    dropout: float = 0.0,
) -> torch.nn.Module:

    if not hasattr(model, "segmentation_head"):
        raise AttributeError("Model does not have a segmentation_head attribute")

    if isinstance(model.segmentation_head, nn.Sequential):
        in_channels = model.segmentation_head[0].in_channels
    else:
        in_channels = model.segmentation_head.in_channels

    layers = []

    if dropout > 0:
        layers.append(nn.Dropout2d(p=dropout))

    layers.append(nn.Conv2d(in_channels, classes, kernel_size=1))

    if activation == "sigmoid":
        layers.append(nn.Sigmoid())
    elif activation == "softmax":
        layers.append(nn.Softmax(dim=1))

    model.segmentation_head = nn.Sequential(*layers)

    return model

if __name__ == "__main__":
    model = build_mitunet(classes=1, checkpoint_path=None)
    print(model)
