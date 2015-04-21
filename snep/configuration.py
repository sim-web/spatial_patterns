import multiprocessing as mp
import os, sys
import argparse

class Config(object):
    """
    Stores all the run time configuration options. There should only
    be one instance of this object, 'config' which should be imported
    and used like a dictionary:

    from Config import config
    config['simulationgroup_name'] = 'my_sim_group_name'
    if config['multiproc']:
        etc...
    """
    def __init__(self, default_group=None):
        self._config={}
        self.default_config()
        self.initialize_argparse()
    
    def initialize_argparse(self):
        parser = argparse.ArgumentParser(description=
                            'SNEP - Simultaneous Numerical Exploration Package')
        parser.add_argument('--procs', action='store', type=int, default=0,
                           help='number of processors to use (default: prompt when run)')
        
        self._parser = parser

    def default_config(self):
        # control multiprocessing behavior
        hastrace = sys.gettrace()
        multiproc = hastrace is None
        print('Multiprocessing ' + ('enabled' if multiproc else 'disabled'))
        self._config['multiproc'] = multiproc
        self._config['dtm'] = False
        #  no of worker processes to spawn
        # Now prompts each time the value is read. 
        #self._config['processes'] = None

        self._config['units_off'] = True

        self._config['simulationdata_exists_action'] = 'run_new_only' # 'run_all'#

        self._config['image_suffix'] = '.svg'  
        
        self._config['network_type'] = 'brian'

    def __getitem__(self, key):
        try:
            ret = self._config[key]
        except KeyError:
            if key == 'processes':
                return self.num_procs()
        return ret

    def __setitem__(self, key, value):
        if key not in self._config:
            raise Exception('Configuration item does not exist')
        else:
            self._config[key] = value
    
    def num_procs(self):
        numprocs = self._parser.parse_args().procs
        if numprocs < 1:
            numprocs = self.prompt_procs()
        
        return numprocs

    def prompt_procs(self):
        default = mp.cpu_count()
        print('Enter number of worker processes ({0}):'.format(default))
        inp = 0
        while inp < 1:
            try:
                raw = raw_input()
                inp = default if raw == '' else int(raw)
            except ValueError:
                print('Invalid selection')
        return inp
            
config = Config()
