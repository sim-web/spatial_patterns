from snep.configuration import config

if config['network_type'] == 'brian':
    from network_brian import *
elif config['network_type'] == 'empty':
    from network_empty import *
else:
    raise Exception('Unknown network type. Check your configuration.')
