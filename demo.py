import argparse
import torch
import torch.nn.parallel

from models import modules, net, resnet, densenet, senet
import numpy as np
import loaddata_demo as loaddata
import pdb

import matplotlib.image
import matplotlib.pyplot as plt
plt.set_cmap("jet")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        original_model = resnet.resnet50(pretrained = True)
        Encoder = modules.E_resnet(original_model) 
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model
   
def _strip_module_prefix(state_dict):
    if any(k.startswith('module.') for k in state_dict.keys()):
        return {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    return state_dict


def main():
    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Wczytaj stan z mapowaniem na aktualne urządzenie (CPU lub GPU)
    path = './pretrained_model/model_senet'
    loaded = torch.load(path, map_location=device)

    # obsłuż różne formaty pliku (bezpośredni state_dict lub dict zawierający 'state_dict')
    state_dict = loaded.get('state_dict', loaded) if isinstance(loaded, dict) else loaded
    state_dict = _strip_module_prefix(state_dict)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    nyu2_loader = loaddata.readNyu2('data/demo/img_nyu2.png')

    test(nyu2_loader, model, device)


def test(nyu2_loader, model, device):
    for i, image in enumerate(nyu2_loader):     
        image = image.to(device)

        with torch.no_grad():
            out = model(image)
        # obsłuż różne wymiary wyjścia: usuń wymiary batch i channel jeśli są
        out_np = out.squeeze().cpu().numpy()

        # zapisz obraz wyniku
        matplotlib.image.imsave('data/demo/out.png', out_np)


if __name__ == '__main__':
    main()
