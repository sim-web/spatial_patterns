import tables
from .rows import ParamSpaceCoordinate
from ..utils import Parameter, ParameterArray, ParameterSpace, \
                    write_named_param, read_sparse
from lock import LockAllFunctionsMeta

class ParameterSpaceReader(object):
    __metaclass__ = LockAllFunctionsMeta
    '''
    A class that knows how to read a subtree of the hdf5 file defined by ExperimentTables. That
    subtree specifies everything about the network and simulation that is part of the experiment.
    '''
    _results_map = property(fget=lambda self: self.paramspace.results_map)
    _coordinate_map = property(fget=lambda self: self.paramspace.coordinate_map)
    _coordinates = property(fget=lambda self: self.paramspace.coordinates)
    def __init__(self, coord_map, flat):
        ''' 
        coord_map - Defines the order in which to place "coordinates" from the parameter space
                    points into a directory name or the group names. It should be a dictionary that 
                    maps from a coordinate to an integer defining the rank of that coordinate.
                    i.e. {
                    ('exc', 'El'): 0,
                    ('inh', 'g_l'): 1,
                    ('exc_exc', 'tau_e'): 2,
                    } 
                    this would result in names like:
                    exc_El_-50mV_inh_g_l_100nS_exc_exc_tau_e_5ms
        flat - if true the results are stored in directories and groups that are only
                one deep, otherwise they are stored in a nested hierarchy.
        '''
        self.coord_map = coord_map
        self.flat = flat

    def set_root(self, paramspace):
        '''
        paramspace - a group in the ExperimentTables hdf5 file.
        '''
        self.paramspace = paramspace

    def make_column_from_coord(self, coord):
        ''' A coord is one (pop,param) or (conn,param) tuple from the paramspace_pt '''
        return '_'.join(coord)

    def num_paramspace_pts(self):
        return self._results_map.nrows

    def _get_tuple_for_coord(self, coordname):
        coord_array = self._coordinates._f_getChild(coordname)
        return tuple(x.tostring().replace('\x00','') for x in coord_array.read())

    def paramspace_pts(self, onlyfinished=False, onlyunfinished=False):
        '''
        This used to be an iterator, but because we now lock the tables when
        reading or writing, we can't yield results here because the lock
        is released when this function returns the iterable.
        
        Should try modifying queries to look more like:
        np.fromiter((r for r in table if 'CLZ' in r['symbol']), dtype=table.dtype)
        '''
        assert(not (onlyfinished and onlyunfinished))
        coord_unmap = {c['coord']:(c['units'],c['alias'])
                            for c in self._coordinate_map}

        inset = lambda st: not ((onlyfinished and st != 'finished') or \
                                (onlyunfinished and st == 'finished'))

        try:
            pts = [{self._get_tuple_for_coord(coord):self._get_param_for_coord(r,coord, units, alias)
                        for coord, (units,alias) in coord_unmap.iteritems()}
                            for r in self._results_map if inset(r['status'])]
        except tables.exceptions.NoSuchNodeError:
            pts = []
        return pts

    def _get_param_for_coord(self, row, coord, units, alias):
        coordvalue = row[coord]
        if alias:
            aliased_group = self.paramspace._f_getChild('aliased/'+coord)
            child = aliased_group._f_getChild(coordvalue)
            if isinstance(child, tables.Group):
                array = read_sparse(aliased_group, coordvalue)
            else:
                array = child.read()
            param = ParameterArray(array,units,coordvalue)
        else:
            param = Parameter(coordvalue,units)
        return param

    def _get_results_row(self, paramspace_pt):
        '''
        Given a paramspace_pt we perform a look-up in the results_map table and return the
        row in the results table for the corresponding simulation subprocess.
        '''
        for res in self._results_map:
            mismatch = False
            for coord, coord_val in paramspace_pt.iteritems():
                col_name = self.make_column_from_coord(coord)
                mismatch = mismatch or res[col_name] != coord_val.coordvalue()
            if not mismatch:
                return res
        assert(False)
        
    def _get_results_row_col(self, paramspace_pt, col):
        '''
        Given a paramspace_pt we perform a look-up in the results_map table and return the
        column in the results table for the corresponding simulation subprocess.
        '''
#        for res in self._results_map:
#            mismatch = False
#            for coord, coord_val in paramspace_pt.iteritems():
#                col_name = self.make_column_from_coord(coord)
#                mismatch = mismatch or res[col_name] != coord_val.value
#            if not mismatch:
#                return res[col]
#        assert(False)
        res = self._get_results_row(paramspace_pt)
        return res[col]

    def get_results_group_path(self, paramspace_pt):
        '''
        Given a paramspace_pt we perform a look-up in the results_map table and return the
        group in which the corresponding simulation subprocess stored its results.
        '''
        return self._get_results_row_col(paramspace_pt, 'group')

    def get_results_directory(self, paramspace_pt):
        '''
        Given a paramspace_pt we perform a look-up in the results_map table and return the
        directory in which the corresponding simulation subprocess stored its results.
        '''
        return self._get_results_row_col(paramspace_pt, 'directory')
    
    def get_results_status(self, paramspace_pt):
        '''
        Each simulation should have a status:
        none     - Not yet run
        finished - Completed successfully
        timedout - Simulation was stopped before it returned due to timeout
        error    - Unknown error, or exception occurred
        '''
        return self._get_results_row_col(paramspace_pt, 'status')
        
#    def get_seed_for_random(self, paramspace_pt):
#        '''
#        Each simulation needs a unique seed. We save it in the tables so that in principle all
#        randomly generated values can be reproduced.
#        '''
#        return self._get_results_row_col(paramspace_pt, 'seed')

class ParameterSpaceTables(ParameterSpaceReader):
    __metaclass__ = LockAllFunctionsMeta
    '''
    A class that knows how to create a subtree of the hdf5 file defined by ExperimentTables. That
    subtree specifies everything about the network and simulation that is part of the experiment.
    '''
    def initialize(self, h5f, parentgroup):
        '''
        Given a parent group this function creates all the necessary default groups and
        tables that this class is responsible for.
        '''
        try:
            paramspace = parentgroup.paramspace
        except tables.NoSuchNodeError:
            paramspace =  h5f.createGroup(parentgroup,'paramspace')
            h5f.createGroup(paramspace,'coordinates')
            h5f.createTable(paramspace, 'coordinate_map', ParamSpaceCoordinate,
                                         'Parameter space coordinate map')
        ParameterSpaceReader.set_root(self,paramspace)

        self.h5f = h5f

    def _define_coordinate_map_table(self, paramspace_pts):
        '''
        Each point in the parameter space is made of coordinates, i.e. given the paramspace_pt
        (exc ge X nS, inh Vr Y mV, exc C Z nF) one coordinate is (exc ge X nS). We build a table 
        that maps from coordinates to column numbers in the results table.
        We also build a ResultRow datatype that defines the rows for the results table.    
        '''
        import numpy as np
        results_row = []
        assert(paramspace_pts)

        # if the user specified a coordinate map, we should use that rather than computing our own.
        # see the constructor of ResultsTablesReader for a description of the coord_map
        make_coord_map = len(self.coord_map) == 0
        
        coords = paramspace_pts[0]
        if make_coord_map:
            for i,k in enumerate([k for k in coords.keys() 
                                    if k!=('iter',) and k!=('seed',)]):
                self.coord_map[k] = i
            if ('iter',) in coords:
                self.coord_map[('iter',)] = len(self.coord_map)
        if ('seed',) in coords:
            self.coord_map[('seed',)] = -1

        row = self._coordinate_map.row
        for coord,val in coords.iteritems():
            coord_name = self.make_column_from_coord(coord)
            alias = val.name is not None

            row['coord' ] = coord_name
            row['units' ] = val.units
            row['column'] = self.coord_map[coord]
            row['alias' ] = alias
            row.append()
            
            coordvalue = val.coordvalue()
            col_type = 'S32' if isinstance(coordvalue,str) else np.array([coordvalue]).dtype
            results_col = (coord_name , col_type)
            results_row.append(results_col)
        self._coordinate_map.flush()

        results_row.append(("directory" , "S256" ))
        results_row.append(("group"     , "S256" ))
        results_row.append(("status"    , "S32"  ))
        ResultRow = np.dtype(results_row)
        return ResultRow

    def define_coordinates(self, coords):
        from snep.tables.rows import name_size
        for coord in coords:
            coord_name = self.make_column_from_coord(coord)
            try:
                self._coordinates._f_getChild(coord_name)
            except tables.NoSuchNodeError:
                coord_array = self.h5f.createVLArray(self._coordinates,coord_name,
                                                 tables.StringAtom(name_size))
                for c in coord:
                    coord_array.append(c)
        self.h5f.flush()
        
    def build_parameter_space(self):
        '''
        Once the experiment has been fully specified and the user calls process() we have to
        compute the complete parameter space. This involves collecting all the parameter ranges
        defined in the NetworkTables, plus any linked variables and passing all those to the
        ParameterSpace class which will generate a list of all the points in the space.
        That list of points is used to build a table which maps from a point to a directory
        on the disk and a group in the hdf5 file where simulation results are stored.
        
        IMPORTANT: The models and synaptic parameter range dictionaries are specified in terms of
        the populations and connections that inherit them. This is done so that any parameter ranges
        defined explicitly on the populations and connections automatically supercede (overwrite) 
        the model and synapse ranges when adding them to the param_ranges dictionary.
        '''
        param_ranges = self._read_param_ranges()

        param_ranges = self._define_seeds(param_ranges)
        self.define_coordinates(param_ranges.keys())

        param_links = self._read_param_links()

        paramspace_pts = [ps for ps in ParameterSpace(param_ranges, param_links)]

        # Once we have the complete list of points, we build a table to map from points to
        # simulation results directories and groups.
        self._define_results_table(paramspace_pts)

    def _define_results_table(self, paramspace_pts):
        '''
        Given a list of all the points in the parameter space that will be simulated, we need
        to build a map from those points, to a directory where the subprocesses can store their
        results, as well as a group into which we should later copy those results.
        '''
        ResultRow = self._define_coordinate_map_table(paramspace_pts)

        results_map = self.h5f.createTable(self.paramspace, 'results_map', ResultRow)
        aliased_group = self.h5f.createGroup(self.paramspace, 'aliased')
        num_coords = len([None for c in self.coord_map.itervalues() if c >= 0])
        row = results_map.row
        for paramspace_pt in paramspace_pts:
            result_dir = [None] * num_coords
            for coord, coord_val in paramspace_pt.iteritems():
                col_idx = self.coord_map[coord]
                col_name = self.make_column_from_coord(coord)
                coordvalue = coord_val.coordvalue()
                alias = coord_val.name is not None
                row[col_name] = coordvalue
                if col_idx >= 0:
                    units = '' if coord_val.units == '1' else coord_val.units
                    subpath = '{0}_{1}{2}'.format(col_name,coordvalue,units)
                    result_dir[col_idx] = subpath
                if alias:
                    try:
                        coord_group = aliased_group._f_getChild(col_name)
                    except:
                        coord_group = self.h5f.createGroup(aliased_group, col_name)
                    try:
                        coord_group._f_getChild(coord_val.name)
                    except:
                        write_named_param(self.h5f,coord_group,coord_val.name,coord_val.value)
            
            row['directory'] = '.'.join(result_dir) if self.flat else '/'.join(result_dir)

            groupname = '_'.join(result_dir) if self.flat else '/'.join(result_dir)
            groupname = groupname.replace('.','p')
            row['group'] = groupname

            #row['seed'] = random.randint(0,1e9)
            row['status'] = "none"
            row.append()
        results_map.flush()

    def _define_seeds(self, param_ranges):
        import numpy as np
        from snep.utils import ParameterArray
        #import random # no need to seed with time, since it's done automatically on import
        iters = param_ranges[('iter',)].value.size if ('iter',) in param_ranges else 1
        ii32 = np.iinfo(np.int32)
        if iters > 1:
            seeds = ParameterArray(np.random.randint(ii32.max,size=iters))
            self.add_parameter_ranges({'seed':seeds})
            self.link_parameter_ranges([('iter',),('seed',)])
            param_ranges.update({('seed',):seeds})
        else:
            seed = Parameter(np.random.randint(ii32.max,size=1)[0])
            self.add_parameters({'seed':seed})

        return param_ranges
        
    
    def set_results_status(self, paramspace_pt, status):
        '''
        Each simulation should have a status:
        none     - Not yet run
        finished - Completed successfully
        timedout - Simulation was stopped before it returned due to timeout
        error    - Unknown error, or exception occurred
        '''
        #it appears that it does not work to change a returned row, then flush the table
#        res = self._get_results_row(paramspace_pt)
#        res['status'] = status
#        res.update()
        for res in self._results_map:
            mismatch = False
            for coord, coord_val in paramspace_pt.iteritems():
                col_name = self.make_column_from_coord(coord)
                mismatch = mismatch or res[col_name] != coord_val.coordvalue()
            if not mismatch:
                res['status'] = status
                res.update()
        self._results_map.flush()
        #self.h5f.flush()
    
    def delete_results_row(self, paramspace_pt):
        pt_name = self.get_results_directory(paramspace_pt)
        self.log_info('Deleting: {0}'.format(pt_name))
        res = self._get_results_row(paramspace_pt)
        self._results_map.removeRows(res.nrow)
        self.h5f.flush()
        return True

    def copy_parameterspace_tables(self, destination_tables):
        self.h5f.copyNode(self._results_map, destination_tables.paramspace)
        self.h5f.copyNode(self._coordinate_map, destination_tables.paramspace,
                          overwrite=True)
        self.h5f.copyNode(self._coordinates, destination_tables.paramspace,
                          overwrite=True)

