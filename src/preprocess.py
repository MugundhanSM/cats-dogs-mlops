from torchvision import transforms

def get_train_transforms(image_size):

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
        [0.5,0.5,0.5],
        [0.5,0.5,0.5]
        )
    ])



def get_val_transforms(image_size):

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])
