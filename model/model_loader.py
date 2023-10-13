import torch
from model.GuideDepth import GuideDepth


def load_model(model_name, weights_pth=None):
    if model_name == 'GuideDepth':
        model = GuideDepth(True)

    else:
        print("Invalid model")
        exit(0)

    if weights_pth is not None:
        state_dict = torch.load(weights_pth, map_location='cpu')
        model.load_state_dict(state_dict)

    return model



