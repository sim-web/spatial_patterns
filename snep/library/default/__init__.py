from snep.configuration import config

if config['network_type'] == 'brian':
    from default_brian import *
elif config['network_type'] == 'empty':
    from default_empty import *
else:
    raise Exception('Unknown network type. Check your configuration.')
