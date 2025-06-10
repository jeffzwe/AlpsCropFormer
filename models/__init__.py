from models.TSViT.TSViTdense import TSViT
from models.TSViT.TSViTcls import TSViTcls

def get_model(config, device):
    model_config = config['MODEL']

    if model_config['architecture'] == "TSViTcls":
        model_config['device'] = device
        return TSViTcls(model_config).to(device)

    if model_config['architecture'] == "TSViT":
        return TSViT(model_config).to(device)

    else:
        raise NameError("Model architecture %s not found, choose from: 'UNET3D', 'UNET3Df', 'UNET2D-CLSTM', 'TSViT', 'TSViTcls'")
