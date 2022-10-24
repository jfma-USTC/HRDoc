import torch


def load_checkpoint(checkpoint, model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    checkpoint = torch.load(checkpoint, map_location='cpu')
    state_dict = model.state_dict()
    for key in state_dict.keys():
        if key in checkpoint['model_param']:
            if state_dict[key].shape == checkpoint['model_param'][key].shape:
                state_dict[key] = checkpoint['model_param'][key]
    model.load_state_dict(state_dict)
