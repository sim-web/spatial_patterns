import tables
from .network import NetworkTables
from .results import ResultsTables
from .paramspace import ParameterSpaceTables
from lock import LockAllFunctionsMeta
import os, logging, sys

class ExperimentTables(NetworkTables, ResultsTables, ParameterSpaceTables):
    __metaclass__ = LockAllFunctionsMeta
    '''
    This is the master hdf5 file handler class. It is responsible for creating an experiment
    file, in which network configuration and data should be stored. It inherits from NetworkTables
    and ResultsTables which do no file handling, but can read and write subtrees from the
    master experiment file.
    '''
    def __init__(self, filename, coord_map={}, flat_results_groups=True):
        '''
        filename - must be a complete path, including the hdf5 file name
        '''
        self.filename = filename

        consoleformatter = logging.Formatter('%(process)d %(levelname)s %(message)s')
        stdout = logging.StreamHandler(sys.stdout)#stdout)
        stdout.setFormatter(consoleformatter)
        stdout.setLevel(logging.INFO)
        logger = logging.getLogger('snep.experiment')
        logger.setLevel(logging.INFO)
        rl = logging.getLogger('')
        rl.addHandler(stdout)
        
        ParameterSpaceTables.__init__(self, coord_map, flat_results_groups)
        ResultsTables.__init__(self)
        NetworkTables.__init__(self)

    def __del__(self):
        self.close_file()

    def log_info(self, msg):
        logging.getLogger('snep.experiment').info(msg)
        #print(msg)

    def open_file(self, readonly=False):
        if readonly:
            self.h5f = tables.openFile(self.filename, mode = "r")
            ResultsTables.set_root(self, self.h5f.root.results)
            NetworkTables.set_root(self, self.h5f.root.network)
            ParameterSpaceTables.set_root(self,self.h5f.root.paramspace)
        else:
            self.h5f = tables.openFile(self.filename, mode = "a", 
                                     title = "Master Experiment File")
    def close_file(self):
        if self.h5f and self.h5f.isopen:
            self.h5f.close()

    def initialize(self):
        '''
        Once a new file is opened, it should be populated with the appropriate default
        groups and tables. This class only needs to create the experiment group at the root
        of the file. It then passes that group as the parent group for the NetworkTables
        and ResultsTables, where they will construct their own groups and tables.
        '''
        ParameterSpaceTables.initialize(self,self.h5f, self.h5f.root)
        NetworkTables.initialize(self, self.h5f, self.h5f.root)
        ResultsTables.initialize(self, self.h5f, self.h5f.root)

    def results_file(self, resultpath):
        '''
        This should probably be specified somehow in the configuration.py but for now
        it's here.
        '''
        return os.path.join(resultpath, 'results.h5')
    
