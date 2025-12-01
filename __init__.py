from .iresnet import iresnet50

def get_model(name, **kwargs):

    if name == "r50":
        return iresnet50(False, **kwargs)
    else:
        raise ValueError()