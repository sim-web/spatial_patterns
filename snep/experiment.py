from configuration import config
import os, gc, time, threading, sys
#import logging, sys
from snep.library import default
from snep.tables.experiment import ExperimentTables
import multiprocessing as mp

class Experiment(object):
    def __init__(self,
                 root_dir,
                 makenet=None,
                 preproc=None,
                 runnet=None,
                 postproc=None,
                 phases=None,
                 #tableclass=ExperimentTables,
                 deletesubprocfiles=True,
                 repeat_experiment_dir=None,
                 repeat_experiment_only_unfinished=False,
                 repeat_experiment_new_paramspace=False,
                 results_coord_map={},
                 suffix=None,
                 flat_results_groups=True
                 ):
        ''' 
        deletesubprocfiles - After the simulations are finished, this controls whether the results
                    files for each of them are kept.
        repeat_experiment_dir - To repeat an experiment, pass the directory it is contained in
                    for this parameter, and the network tables will be copied into a new
                    tables file and the parameter space will be re-simulated.
        repeat_experiment_only_unfinished - Resumes the experiment at repeat_experiment_dir
                    writing the results from any previously unfinished simulations
                    into the existing file.
        repeat_experiment_new_paramspace - Copies experiment from repeat_experiment_dir
                    but removes all the parameter ranges, effectively deleting
                    the original parameter space.
        results_coord_map - Defines the order in which to place "coordinates" from the parameter space
                    points into a directory name or the group names. It should be a dictionary that 
                    maps from a coordinate to an integer defining the rank of that coordinate.
                    i.e. {
                    ('exc', 'El'): 0,
                    ('inh', 'g_l'): 1,
                    ('exc_exc', 'tau_e'): 2,
                    } 
                    this would result in names like:
                    exc_El_-50mV_inh_g_l_100nS_exc_exc_tau_e_5ms
        flat_results_groups - if true the results are stored in directories and groups that are only
                one deep, otherwise they are stored in a nested hierarchy.
        '''
        self.root_dir   = root_dir

        if repeat_experiment_only_unfinished:
            exp_dir = repeat_experiment_dir
        else:
            timestr = time.strftime('%Y-%m-%d-%Hh%Mm%Ss')
            exp_dir = '-'.join((timestr,suffix)) if suffix else timestr
            
        self.experiment_dir = os.path.join(self.root_dir,exp_dir)
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        
        self.tables = ExperimentTables(os.path.join(self.experiment_dir,'experiment.h5'),
                                 results_coord_map,
                                 flat_results_groups)
        self.tables.open_file()
        self.tables.initialize()

        if not repeat_experiment_only_unfinished and repeat_experiment_dir:
            repeat_tables_path = os.path.join(self.root_dir,repeat_experiment_dir,'experiment.h5')
            repeat_tables = ExperimentTables(repeat_tables_path)
            repeat_tables.open_file(readonly=True)
            repeat_tables.initialize()
            repeat_tables.copy_network(self.tables)
            if repeat_experiment_new_paramspace:
                self.tables.delete_all_parameter_ranges()
            else:
                repeat_tables.copy_parameterspace_tables(self.tables)
            repeat_tables.close_file()

        
        self.runnet     = runnet if runnet else default.run
        self.makenet    = makenet if makenet else default.make
        self.preproc    = preproc if preproc else default.preproc
        self.postproc   = postproc if postproc else default.postproc
        self.monitors   = {'spikes':set(),'statevar':{},'poprate':{},'weights':{}}
        self.phases     = phases if phases else []
        
        self.deletesubprocfiles = deletesubprocfiles

        self.manager    = mp.Manager()
        self.result_q   = self.manager.Queue()

    def add_monitors_state_variables(self, monitors):
        self.monitors['statevar'].update(monitors)
    
    def add_monitors_population_rate(self, monitors):
        self.monitors['poprate'].update(monitors)

    def add_monitors_spike_times(self, monitors):
        self.monitors['spikes'].update(monitors)

    def add_monitors_weights(self, monitors):
        self.monitors['weights'].update(monitors)

    def add_monitors_custom(self, monitors):
        self.monitors.update(monitors)
    
    def log_info(self,msg):
        print(msg)
    
    def log_err(self,msg):
        print(msg)
        
#    def make_logger(self):
#        consoleformatter = logging.Formatter('%(levelname)s %(message)s')
#        console = logging.StreamHandler(sys.stderr)
#        console.setFormatter(consoleformatter)
##        if subproc:
##            console.setLevel(logging.WARN)
##        else:
#        console.setLevel(logging.INFO)
#        
#        sg_name = 'my_experiment'
#        self.logger = 'snep.{0}'.format(sg_name)
#        
#        logger = logging.getLogger(self.logger)
#        logger.setLevel(logging.INFO)
#
#        datetime = time.strftime('%Y-%m-%d-%Hh%Mm%Ss')
#        logfile = logging.FileHandler('{0}/{1}.log'.format(logpath,datetime))
#        fileformatter = logging.Formatter('%(asctime)-6s: %(name)s: %(levelname)s %(message)s')
#        logfile.setFormatter(fileformatter)
#        logfile.setLevel(logging.INFO)
#
#        # Direct all loggers to the console and file
#        rl = logging.getLogger('')
#        rl.addHandler(logfile)
#        rl.addHandler(console)


    def subprocess(self, paramspace_pt):
        '''
        Given a point in the parameter space, this function prepares all the data
        necessary for a subprocess to build and simulate a network. Essentially this is
        a few things: 
        1) a dictionary of Python and Brian.Quantity objects that specify the entire network.
        2) the directory in which the subprocess results should be stored must be created and
             passed to the subprocess
        3) all the functions for making, pre-processing, running, and post-processing the network.
        '''
        msg = 'Getting info for pid: {0}, thr: {1}, {2}'
        print(msg.format(os.getpid(),
                         threading.current_thread().name,
                         self.tables.get_results_group_path(paramspace_pt)))

        params = self.tables.as_dictionary(paramspace_pt, True)

        res_directory = self.tables.get_results_directory(paramspace_pt)
        subprocdir = os.path.join(self.experiment_dir, res_directory)
        params['results_group'] = self.tables.get_results_group_path(paramspace_pt)
        params['subprocdir'] = subprocdir

        
        tempdir_new = os.path.join(self.tempdir_user,res_directory)
        params['tempdir_original'] = self.tempdir_orig 
        params['tempdir_subproc'] = tempdir_new 

        if 'PYTHONCOMPILED' in os.environ:
            PYTHONCOMPILED_original = os.environ['PYTHONCOMPILED']
        else:
            PYTHONCOMPILED_original = None
        PYTHONCOMPILED_subproc = (tempdir_new,PYTHONCOMPILED_original) if PYTHONCOMPILED_original else (tempdir_new,)
        params['PYTHONCOMPILED_original'] = PYTHONCOMPILED_original 
        params['PYTHONCOMPILED_subproc'] = (';' if (sys.platform=='win32') else ':').join(PYTHONCOMPILED_subproc)
        
        phases = [(t.quantity,phase) for t,phase in self.phases]

        return (paramspace_pt, self.result_q, 
                self.makenet, self.preproc, self.runnet, self.postproc, 
                phases, self.monitors, params)

    def process(self,timeout=None):
        '''
        This is responsible for actually distributing the simulations to subprocesses for execution. 
        '''
        import numexpr as ne, numpy as np
        import tempfile

        paramspace_pts = self.tables.paramspace_pts(onlyunfinished=True)
        if not paramspace_pts:
            self.tables.build_parameter_space()
            paramspace_pts = self.tables.paramspace_pts()

        self.tempdir_orig = tempfile.gettempdir()
        login = os.getlogin()
        self.tempdir_user = os.path.join(self.tempdir_orig,login)
        if not os.path.exists(self.tempdir_user):
            os.mkdir(self.tempdir_user)

        num_sims = len(paramspace_pts)

        if config['multiproc']:
            num_procs = min(num_sims,config['processes']) if num_sims > 1 else 1
        else:
            num_procs = 1
        
        delay = np.zeros(num_sims)
        # delay can be used to stagger startup of process. no longer needed 
        # thanks to change of tempfile.tempdir
        #delay[1:num_procs] = 30. + np.arange(0., 20.*(num_procs-1), 20.)
        #for i in range((num_sims-num_procs) / num_procs):
        #delay[num_procs:] = 10.*np.random.rand(num_sims - num_procs)
        all_simulations = ((run_simulation_single,self.subprocess(paramspace_pt),d,timeout) 
                                for paramspace_pt,d in zip(paramspace_pts,delay))

        print('Starting in master pid: {0}, thr: {1}'.format(os.getpid(),
                                                               threading.current_thread().name))

        # measure time of all simulations
        global_start=time.time()
        
        pool = None
        if config['multiproc']:
#            try:
#                from deap import dtm
#                if not config['dtm']:
#                    raise Exception
#                # Stupid dtm doesn't support generators
#                all_simulations = list(all_simulations)
#                results = dtm.imap_unordered(run_with_timeout, all_simulations, chunksize=16)
#                start_log = "Starting DTM pool with {0} processes for {1} simulations.\n".format(num_procs,num_sims)
#            except:
            pool = mp.Pool(num_procs,maxtasksperchild=1)
            results = pool.imap_unordered(run_with_timeout, all_simulations)
            start_log = "Starting multiprocessing pool with {0} processes for {1} simulations.\n".format(num_procs,num_sims)
        else:
            start_log = "Starting {0} simulations sequentially without multiprocessor\n".format(num_sims)
            results = (run_with_timeout(ns) for ns in all_simulations)

        #ne_threads = max(1,(mp.cpu_count()-num_procs)/num_procs)
        #ne_threads = min(ne_threads,num_procs) # only one thread per process for compute2
        ne_threads = 1
        ne.set_num_threads(ne_threads)
        self.log_info('numexpr.set_num_threads({0})'.format(ne_threads))

        self.log_info(start_log)
        timedout_pts = []
        error_pts = []
        single_times = []
        log = {'ns':num_sims,
               'div':'================================================================'}
        for log['fc'], (paramspace_pt, single_time) in enumerate(results,1):
            elapsed_time = (time.time() - global_start) / 60.
            result = self.result_q.get(block=True)
            self.collect_result(result)
            if result.status == 'timedout':
                timedout_pts.append(paramspace_pt)
                msg = '{div}\n{elh}h{elm}m minutes, {fc}/{ns}. !!! SIMULATION TIMED OUT !!! '
            elif result.status == 'error':
                error_pts.append(paramspace_pt)
                msg = '{div}\n{elh}h{elm}m minutes, {fc}/{ns}. !!! SIMULATION THREW EXCEPTION !!! '
            else:
                msg = '{div}\n{elh}h{elm}m minutes elapsed, {fc}/{ns} simulations complete. '
            est = '~{esh}h{esm}m remain\n{div}'

            single_times.append(single_time / 60.)
            estimated_time = estimate_time_remaining(single_times, elapsed_time, 
                                                 num_sims, num_procs)
            log['esh'],log['esm'] = int(estimated_time/60),int(estimated_time%60)
            log['elh'],log['elm'] = int(elapsed_time  /60),int(elapsed_time  %60)
            self.log_info(msg.format(**log) + est.format(**log))
            gc.collect()

        if pool:
            pool.close()
            del pool
               
        for paramspace_pt in timedout_pts:
            pt_name = self.tables.get_results_directory(paramspace_pt)
            self.log_err('TIMED OUT: {0}'.format(pt_name))

        for paramspace_pt in error_pts:
            pt_name = self.tables.get_results_directory(paramspace_pt)
            self.log_err('EXCEPTION: {0}'.format(pt_name))

        global_end=time.time()
        self.log_info("...{0} simulations done in {1:.1f} minutes.".format(num_sims,
                                                               (global_end-global_start)/60.))
        self.log_info("Results stored in {0}".format(self.experiment_dir))

        self.tables.close_file()
        return

    def collect_result(self, result):
        import traceback
        status = 'error'
        try:
            finaldata = result.finaldata
            if 'raw_data' in finaldata:
                self.tables.add_raw_data(result.paramspace_pt, 
                                         finaldata['raw_data'])
            if 'computed' in finaldata:
                self.tables.add_computed(result.paramspace_pt, 
                                         finaldata['computed'])
            if 'callstack' in finaldata:
                self.tables.add_log_file(result.paramspace_pt, 'callstack',
                                        finaldata['callstack'])
            if 'exc_info' in finaldata:
                self.tables.add_log_file(result.paramspace_pt, 'exc_info',
                                        finaldata['exc_info'])

            ###################################################################
            # DEPRECATED
            ###################################################################
            if 'poprate' in finaldata:
                for pop_name, (times, rates) in finaldata['poprate'].iteritems():
                    self.tables.add_population_rates(result.paramspace_pt, 
                                                     pop_name, times, rates)
            if 'spikes' in finaldata:
                for pop_name, spikes in finaldata['spikes'].iteritems():
                    self.tables.add_spiketimes(result.paramspace_pt, 
                                               pop_name, spikes)
            if 'statevar' in finaldata:
                for pop_name, all_vars in finaldata['statevar'].iteritems():
                    for varname, (times, values) in all_vars.iteritems():
                        self.tables.add_state_variables(result.paramspace_pt, 
                                                     pop_name, varname, times, values)
#            if 'computed' in finaldata:
#                for comp_name, compresults in finaldata['computed'].iteritems():
#                    for result_name, names_values in compresults.iteritems():
#                        self.tables.add_computed_results(result.paramspace_pt,
#                                                         comp_name, result_name,
#                                                         names_values)
            if 'weights' in finaldata:
                for con_name, weights in finaldata['weights'].iteritems():
                    w_bins, w_times, w_values = weights
                    self.tables.add_connection_weights(result.paramspace_pt,
                                                       con_name, w_bins, 
                                                       w_times, w_values)

        except:
            traceback.print_exc()
            err = 'Exception occurred while collecting results for {0}'
            self.log_err(err.format(result.paramspace_pt))
        else:
            status = result.status
        finally:
            self.tables.set_results_status(result.paramspace_pt, status)

def estimate_time_remaining(single_times, elapsed_time, num_sims, num_procs):
    import numpy as np
    avg_wall_time = np.mean(single_times)
    finished_counter = len(single_times)
    
#    parallel_time = avg_wall_time / num_procs
#    fully_parallel = (num_sims / num_procs) * num_procs
#    partial_parallel = num_sims % num_procs
#    estimated_time = parallel_time*fully_parallel \
#                    + (avg_wall_time if partial_parallel else 0.)
#    remaining_est_A = estimated_time - elapsed_time
    
    num_remaining = num_sims - finished_counter
    fully_parallel = num_remaining / num_procs #integer division
    partial_parallel = np.minimum(num_remaining % num_procs,1)
    remaining_est = avg_wall_time*(fully_parallel+partial_parallel)

    return remaining_est

def run_with_timeout((target, args, presleep, timeout)):
    import traceback
    paramspace_pt = args[0]
    result_q = args[1]
    # staggered startups to fix problem with weave generating empty/truncated files
    time.sleep(presleep) 
    time_start = time.time()
    try:
        if timeout:
            t = threading.Thread(target=target, args=(args,))
            t.start()
            tid = t.ident
            t.join(timeout)
            timedout = t.is_alive()
            if timedout:
                all_frames = sys._current_frames()
                if tid in all_frames:
                    stack = all_frames[tid]
                    # filename, linenum, funcname, code
                    msg = 'File: "{0}", line {1}, in {2}, code: {3}'
                    callstack = [msg.format(fn,ln,f,c.strip() if c else 'None')
                                    for fn,ln,f,c in traceback.extract_stack(stack)]
                    finaldata = {'callstack':callstack}
                else:
                    finaldata = {'callstack':'Thread info not available.'}
                result = SimulationResult(paramspace_pt,finaldata,'timedout')
                result_q.put(result, block=True)
        else:
            target(args)
    except:
        traceback.print_exc()
        finaldata = {'exc_info':traceback.format_exc()}
        result = SimulationResult(paramspace_pt,finaldata,'error')
        result_q.put(result, block=True)

    run_time = time.time()-time_start
    return paramspace_pt, run_time

class SimulationResult(object):
    def __init__(self, paramspace_pt, finaldata, status):
        self.paramspace_pt = paramspace_pt
        self.finaldata = finaldata
        self.status = status

def run_simulation_single((paramspace_pt, result_q,
                           makenet, preproc, runnet, postproc, 
                           phases, monitors, params)):
    '''
    The function used to run a simulation in a subprocess.
    '''
    import numpy, random, shutil
    import traceback, tempfile
    numpy.random.seed(int(params['seed']))
    random.seed(int(params['seed']))

    tempdir_original = params['tempdir_original']
    tempdir_subproc = params['tempdir_subproc']
    PYTHONCOMPILED_original = params['PYTHONCOMPILED_original']
    PYTHONCOMPILED_subproc  = params['PYTHONCOMPILED_subproc']

    all_exc_info = ''
    finaldata = {}
    try:
        # Change the temp directories used to build the weave stuff.
        # Without this the build will fail, due to weave_imp.o being accessed
        # by multiple processes.
        if os.path.exists(tempdir_subproc):
            shutil.rmtree(tempdir_subproc)
        os.mkdir(tempdir_subproc)
        tempfile.tempdir = tempdir_subproc
        os.environ['PYTHONCOMPILED'] = PYTHONCOMPILED_subproc
    except:
        traceback.print_exc()
        all_exc_info += '\nEXCEPTION SETTING TEMPDIRS: {0}'.format(traceback.format_exc())

    try:
        neuron_groups = makenet(params)
        neuron_groups, monitor_objs = preproc(params, neuron_groups, monitors)
        rawdata = runnet(params, neuron_groups, monitor_objs, *([phases] if phases else []))
        procdata = postproc(params, rawdata)
        finaldata.update(procdata)
    except:
        traceback.print_exc()
        all_exc_info += '\nEXCEPTION IN RUN: {0}'.format(traceback.format_exc())
        status = 'error'
    else:
        status = 'finished'
    
    try:
        if PYTHONCOMPILED_original:
            os.environ['PYTHONCOMPILED'] = PYTHONCOMPILED_original
        else:
            del os.environ['PYTHONCOMPILED']
    except:
        traceback.print_exc()
        all_exc_info += '\nEXCEPTION RESETTING PYTHONCOMPILED: {0}'.format(traceback.format_exc())

    try:
        tempfile.tempdir = tempdir_original
        shutil.rmtree(tempdir_subproc)
    except:
        traceback.print_exc()
        all_exc_info += '\nEXCEPTION RESETTING TEMPDIR: {0}'.format(traceback.format_exc())

    try:
        if all_exc_info != '':
            finaldata['exc_info'] = all_exc_info
        result = SimulationResult(paramspace_pt,finaldata,status)
        result_q.put(result, block=True)
    except IOError:
        traceback.print_exc()
        result.status = 'error_ipc'
        if 'exc_info' in result.finaldata:
            result.finaldata['exc_info'] += traceback.format_exc()
        else:
            result.finaldata = traceback.format_exc()
        result_q.put(result, block=True)

