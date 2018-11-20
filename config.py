from easydict import EasyDict as edict

# default settings
default = edict()

# network settings
default.network = 'MobileNet_4'
default.lr = 0.001
default.image_size = 28
default.epoch=  2
default.batch_size = 128


def generate_config(config_dict=''):
    config = default.copy()
    config.update(config_dict)
    return edict(config)
