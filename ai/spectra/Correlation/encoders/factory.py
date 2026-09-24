from ai.spectra.Correlation.config import (
    CorrelationConfig,
)
from ai.spectra.Correlation.encoders.base import (
    IdentityEncoder,
)
from ai.spectra.Correlation.encoders.resnet_encoder import (
    ResNetIdentityEncoder,
)


def create_identity_encoder(
    config: CorrelationConfig,
) -> IdentityEncoder:

    encoder_type = (
        config.identity_encoder_type
        .strip()
        .lower()
    )

    if encoder_type == "resnet":
        return ResNetIdentityEncoder(
            backbone_name=(
                config.identity_backbone
            ),

            pretrained=(
                config.identity_pretrained
            ),

            device=(
                config.device
            ),
        )

    if encoder_type in {
        "reid",
        "person_reid",
    }:
        raise NotImplementedError(
            "Person Re-ID ainda não foi "
            "implementado. "
            "Use identity_encoder_type='resnet'."
        )

    raise ValueError(
        "IdentityEncoder desconhecido: "
        f"{encoder_type}"
    )