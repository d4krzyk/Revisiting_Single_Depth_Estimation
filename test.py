import argparse
import torch
import torch.nn as nn
import torch.nn.parallel

from models import modules, net, resnet, densenet, senet
import loaddata
import util
import numpy as np
import sobel


def load_checkpoint_safe(model, path, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and ('state_dict' in ckpt or 'model' in ckpt):
        state_dict = ckpt.get('state_dict', ckpt.get('model'))
    else:
        state_dict = ckpt

    # usuń 'module.' jeśli występuje
    def strip_module(k):
        return k[len("module."):] if k.startswith("module.") else k
    state_dict = {strip_module(k): v for k, v in state_dict.items()}

    model_state = model.state_dict()
    model_keys = set(model_state.keys())

    # lista funkcji transformujących klucze checkpointu
    transforms = [
        lambda k: k,
        lambda k: 'E.' + k,
        lambda k: k.replace('features.', 'E.base.'),
        lambda k: k.replace('encoder.', 'E.'),
        lambda k: k.replace('conv1.', 'E.base.conv1.') if k.startswith('conv1.') else k,
    ]

    best_filtered = {}
    best_count = -1
    best_name = None

    for idx, fn in enumerate(transforms):
        transformed = {}
        count = 0
        for k, v in state_dict.items():
            newk = fn(k)
            if newk in model_state and model_state[newk].size() == v.size():
                transformed[newk] = v
                count += 1
        if count > best_count:
            best_count = count
            best_filtered = transformed
            best_name = idx

    # fallback: intersection bez transformacji, z porównaniem rozmiaru
    if best_count <= 0:
        filtered = {}
        for k, v in state_dict.items():
            if k in model_state and model_state[k].size() == v.size():
                filtered[k] = v
        best_filtered = filtered
        best_count = len(filtered)
        best_name = 'intersection'

    print(f"Selected mapping #{best_name}, loading {best_count} tensors (matching by name+size).")

    res = model.load_state_dict(best_filtered, strict=False)
    missing = getattr(res, "missing_keys", None) or (res[0] if isinstance(res, (list, tuple)) and len(res) > 0 else [])
    unexpected = getattr(res, "unexpected_keys", None) or (res[1] if isinstance(res, (list, tuple)) and len(res) > 1 else [])
    if len(missing) > 0:
        print("Missing keys (model expects):", len(missing))
        if len(missing) <= 20:
            print(missing)
        else:
            print("... (too many to show)")

    if len(unexpected) > 0:
        print("Unexpected keys (in checkpoint but not used):", len(unexpected))
        if len(unexpected) <= 20:
            print(unexpected)
        else:
            print("... (too many to show)")

    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
    # Użyj DataParallel tylko gdy są dostępne GPU
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)
    # map_location zapewnia poprawne wczytanie na CPU gdy brak CUDA
    load_checkpoint_safe(model, './pretrained_model/model_senet', device)
    test_loader = loaddata.getTestingData(1)
    test(test_loader, model, 0.25, device)


def test(test_loader, model, thre, device):

    model.eval()

    totalNumber = 0

    Ae = 0
    Pe = 0
    Re = 0
    Fe = 0

    errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}

    non_blocking = True if device.type == 'cuda' else False

    for i, sample_batched in enumerate(test_loader):
        image, depth = sample_batched['image'], sample_batched['depth']

        depth = depth.to(device, non_blocking=non_blocking)
        image = image.to(device, non_blocking=non_blocking)

        with torch.no_grad():
            output = model(image)
            output = torch.nn.functional.interpolate(output, size=(depth.size(2), depth.size(3)), mode='bilinear',
                                                     align_corners=False)

            depth_edge = edge_detection(depth)
            output_edge = edge_detection(output)

            batchSize = depth.size(0)
            totalNumber = totalNumber + batchSize
            errors = util.evaluateError(output, depth)
            errorSum = util.addErrors(errorSum, errors, batchSize)
            averageError = util.averageErrors(errorSum, totalNumber)

            edge1_valid = (depth_edge > thre)
            edge2_valid = (output_edge > thre)

            nvalid = np.sum((edge1_valid == edge2_valid).cpu().numpy())
            A = nvalid / (depth.size(2) * depth.size(3))

            nvalid2 = np.sum((edge1_valid & edge2_valid).cpu().numpy())

            denomP = np.sum(edge2_valid.cpu().numpy())
            denomR = np.sum(edge1_valid.cpu().numpy())

            P = (nvalid2 / denomP) if denomP > 0 else 0.0
            R = (nvalid2 / denomR) if denomR > 0 else 0.0
            denomF = P + R
            F = (2.0 * P * R / denomF) if denomF > 0 else 0.0

            Ae += A
            Pe += P
            Re += R
            Fe += F

        Av = Ae / totalNumber
        Pv = Pe / totalNumber
        Rv = Re / totalNumber
        Fv = Fe / totalNumber
        print('PV', Pv)
        print('RV', Rv)
        print('FV', Fv)

        averageError['RMSE'] = np.sqrt(averageError['MSE'])
        print(averageError)

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
   

def edge_detection(depth):
    get_edge = sobel.Sobel().to(depth.device)

    edge_xy = get_edge(depth)
    edge_sobel = torch.pow(edge_xy[:, 0, :, :], 2) + \
        torch.pow(edge_xy[:, 1, :, :], 2)
    edge_sobel = torch.sqrt(edge_sobel)

    return edge_sobel


if __name__ == '__main__':
    main()
