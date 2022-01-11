import torch



def get_default_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(d, device) for d in data]
    return data.to_device(device, non_blocking=True)

class DeviceDataLoader():
    """
    Wrapper for a data loader with device
    """
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for d in self.dl:
            yield to_device(self.dl, self.device)
