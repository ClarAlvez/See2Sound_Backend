from torchvision import transforms


IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


def get_train_transforms(
    image_size=224,
):
    """
    Transformações genéricas usadas por modelos visuais
    como SceneNet, ObjectNet e outros.

    Possuem augmentation mais forte para melhorar
    a generalização espacial e visual.
    """

    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=10
        ),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),

        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


def get_atmosphere_train_transforms(
    image_size=224,
):
    """
    Transformações específicas para o AtmosphereNet.

    Evita mudanças artificiais de:
    - brilho
    - contraste
    - saturação
    - tonalidade

    porque essas propriedades são informações
    importantes para reconhecer atmosfera,
    iluminação e condições climáticas.
    """

    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=3
        ),

        transforms.RandomAffine(
            degrees=0,
            translate=(0.02, 0.02),
            scale=(0.98, 1.02),
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


def get_validation_transforms(
    image_size=224,
):
    """
    Transformações determinísticas usadas
    em validação.
    """

    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


def get_test_transforms(
    image_size=224,
):
    """
    Transformações determinísticas usadas
    durante inferência e teste.
    """

    return get_validation_transforms(
        image_size=image_size
    )