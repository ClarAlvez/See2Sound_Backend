from torchvision import transforms


def get_train_transforms(image_size=224):
    """
    Transformações usadas durante o treinamento.

    Aqui usamos data augmentation para aumentar a variedade dos dados:
    - espelhamento horizontal
    - pequenas rotações
    - mudanças leves de brilho, contraste e saturação
    - pequenas translações e zooms

    Isso ajuda a rede a generalizar melhor.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomRotation(degrees=10),

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
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_validation_transforms(image_size=224):
    """
    Transformações usadas em validação e teste.

    Aqui não usamos transformações aleatórias, porque validação e teste
    precisam ser estáveis e comparáveis.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_test_transforms(image_size=224):
    """
    Alias para manter separado conceitualmente o teste da validação.
    Por enquanto, usa as mesmas transformações da validação.
    """
    return get_validation_transforms(image_size=image_size)