"""
The "main script" of this repository: Read in a configuration file and
generate synthetic GW data according to the provided specifications.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from __future__ import print_function

import argparse
import copy
import numpy as np
import os
import re
import sys
import time

from itertools import count
from multiprocessing import Process, Queue
from tqdm import tqdm
import h5py

from utils.configfiles import read_ini_config, read_json_config
from utils.hdffiles import NoiseTimeline
from utils.hdffiles_DeepClean import NoiseTimeline_DC
from utils.hdffiles_original import NoiseTimeline_original
from utils.samplefiles import SampleFile
from utils.samplegeneration import generate_sample
from utils.waveforms import WaveformParameterGenerator
import concurrent.futures

from astropy.utils import iers
iers.conf.auto_download = False


def _parse_mode_keys(raw_modes, prefix):
    """Parse mode labels from argparse (e.g. "22" or "2,2")."""

    mode_keys = []
    for raw_mode in raw_modes:
        normalized = raw_mode.replace('_', ',').replace('-', ',')
        if ',' in normalized:
            parts = normalized.split(',')
            if len(parts) != 2:
                raise ValueError('Invalid {} mode value: {}.'.format(prefix,
                                                                     raw_mode))
            mode_l, mode_m = parts
        else:
            if len(normalized) < 2 or not normalized.isdigit():
                raise ValueError('Invalid {} mode value: {}.'.format(prefix,
                                                                     raw_mode))
            mode_l = normalized[:-1]
            mode_m = normalized[-1]

        mode_keys.append('{},{}'.format(int(mode_l), int(mode_m)))

    return mode_keys


def _parse_deviation_range(static_arguments, prefix):
    """Read a deviation range from static args.

    Supported INI keys in [static_args]:
      * {prefix}_range = <min>,<max>
      * {prefix}_range_min = <min>
        {prefix}_range_max = <max>
    """

    min_key = '{}_range_min'.format(prefix)
    max_key = '{}_range_max'.format(prefix)
    combined_key = '{}_range'.format(prefix)

    if min_key in static_arguments or max_key in static_arguments:
        if min_key not in static_arguments or max_key not in static_arguments:
            raise ValueError('Both {} and {} must be set in [static_args].'.format(
                min_key,
                max_key
            ))
        min_value = float(static_arguments[min_key])
        max_value = float(static_arguments[max_key])
    elif combined_key in static_arguments:
        raw_value = static_arguments[combined_key]
        pieces = [piece.strip() for piece in re.split(r'[,:\s]+', raw_value)
                  if piece.strip()]
        if len(pieces) != 2:
            raise ValueError('Expected {} to contain exactly two values.'.format(
                combined_key
            ))
        min_value, max_value = map(float, pieces)
    else:
        return None

    if min_value > max_value:
        raise ValueError('{} must satisfy min <= max.'.format(combined_key))

    return min_value, max_value


def _sample_range_value(value_range):
    """Draw a single value uniformly from the given (min, max) range."""

    if value_range is None:
        return None
    min_value, max_value = value_range
    return np.random.uniform(min_value, max_value)


def _parse_mode_keys(raw_modes, prefix):
    """Parse mode labels from argparse (e.g. "22" or "2,2")."""

    mode_keys = []
    for raw_mode in raw_modes:
        normalized = raw_mode.replace('_', ',').replace('-', ',')
        if ',' in normalized:
            parts = normalized.split(',')
            if len(parts) != 2:
                raise ValueError('Invalid {} mode value: {}.'.format(prefix,
                                                                     raw_mode))
            mode_l, mode_m = parts
        else:
            if len(normalized) < 2 or not normalized.isdigit():
                raise ValueError('Invalid {} mode value: {}.'.format(prefix,
                                                                     raw_mode))
            mode_l = normalized[:-1]
            mode_m = normalized[-1]

        mode_keys.append('{},{}'.format(int(mode_l), int(mode_m)))

    return mode_keys


def _parse_deviation_range(static_arguments, prefix):
    """Read a deviation range from static args.

    Supported INI keys in [static_args]:
      * {prefix}_range = <min>,<max>
      * {prefix}_range_min = <min>
        {prefix}_range_max = <max>
    """

    min_key = '{}_range_min'.format(prefix)
    max_key = '{}_range_max'.format(prefix)
    combined_key = '{}_range'.format(prefix)

    if min_key in static_arguments or max_key in static_arguments:
        if min_key not in static_arguments or max_key not in static_arguments:
            raise ValueError('Both {} and {} must be set in [static_args].'.format(
                min_key,
                max_key
            ))
        min_value = float(static_arguments[min_key])
        max_value = float(static_arguments[max_key])
    elif combined_key in static_arguments:
        raw_value = static_arguments[combined_key]
        pieces = [piece.strip() for piece in re.split(r'[,:\s]+', raw_value)
                  if piece.strip()]
        if len(pieces) != 2:
            raise ValueError('Expected {} to contain exactly two values.'.format(
                combined_key
            ))
        min_value, max_value = map(float, pieces)
    else:
        return None

    if min_value > max_value:
        raise ValueError('{} must satisfy min <= max.'.format(combined_key))

    return min_value, max_value


def _sample_range_value(value_range):
    """Draw a single value uniformly from the given (min, max) range."""

    if value_range is None:
        return None
    min_value, max_value = value_range
    return np.random.uniform(min_value, max_value)


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def queue_worker(arguments, results_queue):
    """
    Helper function to generate a single sample in a dedicated process.

    Args:
        arguments (dict): Dictionary containing the arguments that are
            passed to generate_sample().
        results_queue (Queue): The queue to which the results of this
            worker / process are passed.
    """
    
    # Try to generate a sample using the given arguments and store the result
    # in the given result_queue (which is shared across all worker processes).
    
    index, arguments = arguments
    try:
        result = generate_sample(**arguments)
        results_queue.put((index, result))

    # For some arguments, LALSuite crashes during the sample generation.
    # In this case, terminate with a non-zero exit code to make sure a new
    # set of argument is added to the main arguments_queue
    except RuntimeError as e:
        print(f"Runtime Error in process: {e}")
    finally:
#        results_queue.put((index, None))  # Indicate failure or no result
        sys.exit(0)
        
        
def _worker(arg):
    """Unpack ((index, arguments)) and call generate_sample."""
    idx, arguments = arg
    result = generate_sample(**arguments)
    return idx, result
    
    
#    try:
#        result = generate_sample(**arguments)
#        results_queue.put(result)
#        sys.exit(0)
    
#    except RuntimeError:
#        sys.exit('Runtime Error')


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    # Disable output buffering ('flush' option is not available for Python 2)
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w')

    # Start the stopwatch
    script_start = time.time()

    print('')
    print('GENERATE A GW DATA SAMPLE FILE')
    print('')
    
    # -------------------------------------------------------------------------
    # Parse the command line arguments
    # -------------------------------------------------------------------------

    # Set up the parser and add arguments
    parser = argparse.ArgumentParser(description='Generate a GW data sample.')
    parser.add_argument('--config-file',
                        help='Name of the JSON configuration file which '
                             'controls the sample generation process.',
                        default='default.json')
    
    parser.add_argument('--negative-latency', 
                        help='integer seconds of template to truncate.', type=int,
                        default=0)

    parser.add_argument('--n-noise-realizations', 
                        help='Number of noise realizations for each template.', type=int,
                        default=1)
    
    parser.add_argument('--detectors', 
                    help='List of detectors to use', 
                    type=str,
                    nargs='+',  # allows multiple arguments as a list
                    default=['H1', 'L1'])


    parser.add_argument('--add_glitches_noise', type=str,
                        help='What type of glitch to add in pure noise',
                        default=None)
    
    parser.add_argument('--add_glitches_injection', type=str,
                        help='What type of glitch to add in injection',
                        default=None)

    parser.add_argument('--domega-22', type=float, default=0.0,
                        help='Deviation in mode frequency for mode (2,2).')
    parser.add_argument('--domega-33', type=float, default=0.0,
                        help='Deviation in mode frequency for mode (3,3).')
    parser.add_argument('--domega-21', type=float, default=0.0,
                        help='Deviation in mode frequency for mode (2,1).')
    parser.add_argument('--domega-32', type=float, default=0.0,
                        help='Deviation in mode frequency for mode (3,2).')
    parser.add_argument('--domega-44', type=float, default=0.0,
                        help='Deviation in mode frequency for mode (4,4).')
    parser.add_argument('--domega-43', type=float, default=0.0,
                        help='Deviation in mode frequency for mode (4,3).')
    parser.add_argument('--domega-55', type=float, default=0.0,
                        help='Deviation in mode frequency for mode (5,5).')
    parser.add_argument('--dtau-22', type=float, default=0.0,
                        help='Deviation in damping time for mode (2,2).')
    parser.add_argument('--dtau-33', type=float, default=0.0,
                        help='Deviation in damping time for mode (3,3).')
    parser.add_argument('--dtau-21', type=float, default=0.0,
                        help='Deviation in damping time for mode (2,1).')
    parser.add_argument('--dtau-32', type=float, default=0.0,
                        help='Deviation in damping time for mode (3,2).')
    parser.add_argument('--dtau-44', type=float, default=0.0,
                        help='Deviation in damping time for mode (4,4).')
    parser.add_argument('--dtau-43', type=float, default=0.0,
                        help='Deviation in damping time for mode (4,3).')
    parser.add_argument('--dtau-55', type=float, default=0.0,
                        help='Deviation in damping time for mode (5,5).')
    parser.add_argument('--domega-range-modes', type=str, nargs='+', default=[],
                        help='Modes to which a sampled domega range value '
                             'from [static_args] should be applied '
                             '(e.g. 22 33 or 2,2 3,3).')
    parser.add_argument('--dtau-range-modes', type=str, nargs='+', default=[],
                        help='Modes to which a sampled dtau range value '
                             'from [static_args] should be applied '
                             '(e.g. 22 33 or 2,2 3,3).')

    # Parse the arguments that were passed when calling this script
    print('Parsing command line arguments...', end=' ')
    parsed_arguments = parser.parse_args()
    command_line_arguments = vars(parsed_arguments)

    try:
        domega_range_modes = _parse_mode_keys(
            command_line_arguments['domega_range_modes'],
            prefix='domega'
        )
        dtau_range_modes = _parse_mode_keys(
            command_line_arguments['dtau_range_modes'],
            prefix='dtau'
        )
    except ValueError as error:
        parser.error(str(error))


    

#    if unknown_arguments:
#        parser.error('unrecognized arguments: {}'.format(
#            ' '.join(unknown_arguments)
#        ))
    
    # Access the detectors argument
    detectors = command_line_arguments['detectors']

    # Perform operations based on detectors
    detectors_set = set(detectors)  # Convert to set for easy membership testing

    # Asking for additional input if --add-glitches is True
    if command_line_arguments['add_glitches_noise'] is not None:
        
        print(f"Glitch to add in noise: {command_line_arguments['add_glitches_noise']}")
        
    if command_line_arguments['add_glitches_injection'] is not None:
        
        print(f"Glitch to add in injection: {command_line_arguments['add_glitches_injection']}")

    #    glitch_name = glitch_name.replace(" ", "_").lower()

    else:
        print("No glitches will be added.")


    print('Done!')

    # -------------------------------------------------------------------------
    # Read in JSON config file specifying the sample generation process
    # -------------------------------------------------------------------------

    # Build the full path to the config file
    json_config_name = command_line_arguments['config_file']
    json_config_path = os.path.join('.', 'config_files', json_config_name)
    
    # Read the JSON configuration into a dict
    print('Reading and validating in JSON configuration file...', end=' ')
    config = read_json_config(json_config_path)
    print('Done!')

    # -------------------------------------------------------------------------
    # Read in INI config file specifying the static_args and variable_args
    # -------------------------------------------------------------------------

    # Build the full path to the waveform params file
    ini_config_name = config['waveform_params_file_name']
    ini_config_path = os.path.join('.', 'config_files', ini_config_name)

    # Read in the variable_arguments and static_arguments
    print('Reading and validating in INI configuration file...', end=' ')
    variable_arguments, static_arguments = read_ini_config(ini_config_path)
    print('Done!\n')

#    try:
#        domega_range = _parse_deviation_range(static_arguments, 'domega')
#        dtau_range = _parse_deviation_range(static_arguments, 'dtau')
#    except ValueError as error:
#        parser.error(str(error))

#    static_arguments['domega_dict'] = domega_dict
#    static_arguments['dtau_dict'] = dtau_dict



    try:
        domega_range = _parse_deviation_range(static_arguments, 'domega')
        dtau_range = _parse_deviation_range(static_arguments, 'dtau')
    except ValueError as error:
        parser.error(str(error))

    # Build per-mode deviation dictionaries from CLI arguments (defaults are 0.0)
    domega_dict = {
        '2,2': float(command_line_arguments.get('domega_22', 0.0)),
        '3,3': float(command_line_arguments.get('domega_33', 0.0)),
        '2,1': float(command_line_arguments.get('domega_21', 0.0)),
        '3,2': float(command_line_arguments.get('domega_32', 0.0)),
        '4,4': float(command_line_arguments.get('domega_44', 0.0)),
        '4,3': float(command_line_arguments.get('domega_43', 0.0)),
        '5,5': float(command_line_arguments.get('domega_55', 0.0)),
    }

    dtau_dict = {
        '2,2': float(command_line_arguments.get('dtau_22', 0.0)),
        '3,3': float(command_line_arguments.get('dtau_33', 0.0)),
        '2,1': float(command_line_arguments.get('dtau_21', 0.0)),
        '3,2': float(command_line_arguments.get('dtau_32', 0.0)),
        '4,4': float(command_line_arguments.get('dtau_44', 0.0)),
        '4,3': float(command_line_arguments.get('dtau_43', 0.0)),
        '5,5': float(command_line_arguments.get('dtau_55', 0.0)),
    }

    # Optionally override selected modes with a sampled value from [static_args]
    if domega_range is not None:
        sampled_domega = _sample_range_value(domega_range)
        for mode_key in domega_range_modes:
            domega_dict[mode_key] = float(sampled_domega)

    if dtau_range is not None:
        sampled_dtau = _sample_range_value(dtau_range)
        for mode_key in dtau_range_modes:
            dtau_dict[mode_key] = float(sampled_dtau)

    # Store for waveform generation
    static_arguments['domega_dict'] = domega_dict
    static_arguments['dtau_dict'] = dtau_dict

 

    # -------------------------------------------------------------------------
    # Shortcuts and random seed
    # -------------------------------------------------------------------------

    # Set the random seed for this script
    np.random.seed(config['random_seed'])

    # Define some useful shortcuts
    random_seed = config['random_seed']
    max_runtime = config['max_runtime']
    bkg_data_dir = config['background_data_directory']

    # -------------------------------------------------------------------------
    # Construct a generator for sampling waveform parameters
    # -------------------------------------------------------------------------

    # Initialize a waveform parameter generator that can sample injection
    # parameters from the distributions specified in the config file
    waveform_parameter_generator = \
        WaveformParameterGenerator(config_file=ini_config_path,
                                   random_seed=random_seed)

    # Wrap it in a generator expression so that we can we can easily sample
    # from it by calling next(waveform_parameters)
    waveform_parameters = \
        (waveform_parameter_generator.draw() for _ in iter(int, 1))

    # -------------------------------------------------------------------------
    # Construct a generator for sampling valid noise times
    # -------------------------------------------------------------------------

    # If the 'background_data_directory' is None, we will use synthetic noise
    if config['background_data_directory'] is None:

        print('Using synthetic noise! (background_data_directory = None)\n')

        # Create a iterator that returns a fake "event time", which we will
        # use as a seed for the RNG to ensure the reproducibility of the
        # generated synthetic noise.
        # For the HDF file path that contains that time, we always yield
        # None, so that we know that we need to generate synthetic noise.
        noise_times = ((1000000000 + _, None) for _ in count())

    elif config['background_data_directory'] == 'Deep_Clean_data' or config['background_data_directory'] == 'Deep_Clean_data_test':

        print('Using data from Deep Clean '
            '(background_data_directory = {})'.format(bkg_data_dir))
        print('Reading in Deep Clean data. This may take several minutes...', end=' ')

        # Create a timeline object by running over all HDF files once
        noise_timeline = NoiseTimeline_DC(background_data_directory=bkg_data_dir,
                                        random_seed=random_seed)

        
        delta_t = int(static_arguments['noise_interval_width'] / 2)
        
        noise_times = (noise_timeline.sample(delta_t=delta_t,
                                            return_paths=True)
                            for _ in iter(int, 1))
            

        print('Done!\n')

    elif config['background_data_directory'] == 'Original_data' or config['background_data_directory'] == 'Original_data_test':

        print('Using original data '
            '(background_data_directory = {})'.format(bkg_data_dir))
        print('Reading in original data. This may take several minutes...', end=' ')

        # Create a timeline object by running over all HDF files once
        noise_timeline = NoiseTimeline_original(background_data_directory=bkg_data_dir,
                                        random_seed=random_seed)

        
        delta_t = int(static_arguments['noise_interval_width'] / 2)
        
        noise_times = (noise_timeline.sample(delta_t=delta_t,
                                            return_paths=True)
                            for _ in iter(int, 1))
            

        print('Done!\n')


    # Otherwise, we set up a timeline object for the background noise, that
    # is, we read in all HDF files in the raw_data_directory and figure out
    # which parts of it are useable (i.e., have the right data quality and
    # injection bits set as specified in the config file).
    else:

    #    def generate_noise_times(num_iterations):
    #        delta_t = int(static_arguments['noise_interval_width'] / 2)
    #        for _ in range(num_iterations):
    #            yield (noise_timeline.sample(delta_t=delta_t,
    #                                 dq_bits=config['dq_bits'],
    #                                 inj_bits=config['inj_bits'],
    #                                 return_paths=True)
    #                    for _ in iter(int, 1))

        print('Using real noise from LIGO recordings! '
            '(background_data_directory = {})'.format(bkg_data_dir))
        print('Reading in raw data. This may take several minutes...', end=' ')

        # Create a timeline object by running over all HDF files once
        noise_timeline = NoiseTimeline(background_data_directory=bkg_data_dir,
                                        random_seed=random_seed)

#        noise_timeline_1 = NoiseTimeline(background_data_directory=bkg_data_dir,
#                                        random_seed=(random_seed+100))

        # Create a noise time generator so that can sample valid noise times
        # simply by calling next(noise_time_generator)
        delta_t = int(static_arguments['noise_interval_width'] / 2)

        noise_times = (noise_timeline.sample(delta_t=delta_t,
                                            dq_bits=config['dq_bits'],
                                            inj_bits=config['inj_bits'],
                                            return_paths=True)
                            for _ in iter(int, 1))
        
    #    noise_times_1 = (noise_timeline_1.sample(delta_t=delta_t,
    #                                        dq_bits=config['dq_bits'],
    #                                        inj_bits=config['inj_bits'],
    #                                        return_paths=True)
    #                    for _ in iter(int, 1))
        
        print('Done!\n')

    # -------------------------------------------------------------------------
    # Define a convenience function to generate arguments for the simulation
    # -------------------------------------------------------------------------

    def generate_arguments(injection=True):

        # Only sample waveform parameters if we are making an injection
        waveform_params = next(waveform_parameters) if injection else None

        sample_static_arguments = copy.deepcopy(static_arguments)

        if injection and domega_range is not None and domega_range_modes:
            sampled_domega = _sample_range_value(domega_range)
            sample_static_arguments['domega_dict'] = dict(
                sample_static_arguments['domega_dict']
            )
            for mode_key in domega_range_modes:
                sample_static_arguments['domega_dict'][mode_key] = sampled_domega

        if injection and dtau_range is not None and dtau_range_modes:
            sampled_dtau = _sample_range_value(dtau_range)
            sample_static_arguments['dtau_dict'] = dict(
                sample_static_arguments['dtau_dict']
            )
            for mode_key in dtau_range_modes:
                sample_static_arguments['dtau_dict'][mode_key] = sampled_dtau

        # Return all necessary arguments as a dictionary
        return dict(static_arguments=sample_static_arguments,
                        event_tuple=next(noise_times),
                        add_glitches_noise=command_line_arguments['add_glitches_noise'],
                        add_glitches_injection=command_line_arguments['add_glitches_injection'],
#                       glitch_name=glitch_name,
                        detector=command_line_arguments['detectors'],
                        waveform_params=waveform_params)
        
    # -------------------------------------------------------------------------
    # Finally: Create our samples!
    # -------------------------------------------------------------------------

    # Keep track of all the samples (and parameters) we have generated
    samples = dict(injection_samples=[], noise_samples=[])
    injection_parameters = dict(injection_samples=[], noise_samples=[])

    # The procedure for generating samples with and without injections is
    # mostly the same; the only real difference is which arguments_generator
    # we have have to use:
    for sample_type in ('injection_samples', 'noise_samples'):
    
        # ---------------------------------------------------------------------
        # Define some sample_type-specific shortcuts
        # ---------------------------------------------------------------------
        
        if sample_type == 'injection_samples':
            print('Generating samples containing an injection...')
            n_samples = config['n_injection_samples']*command_line_arguments['n_noise_realizations']
            arguments_generator = \
                (generate_arguments(injection=True) for _ in iter(int, 1))
            
        else:
            print('Generating samples *not* containing an injection...')
            n_samples = config['n_noise_samples']
            arguments_generator = \
                (generate_arguments(injection=False) for _ in iter(int, 1))
                
                
#        # build a flat list of (index, arguments) tuples
#        args_list = []
#        prev_args = None
#        for i in range(n_samples):
#            if i == 0:
#                args = next(arguments_generator)
#                prev_args = args
#            elif (i % command_line_arguments['n_noise_realizations'] == 0):
#                args = next(arguments_generator)
#                prev_args = args
#            else:
#                # same waveform but new noise time
#                args = dict(prev_args,
#                            event_tuple=next(noise_times))
#                prev_args = args
#            args_list.append((i, args))


        # ---------------------------------------------------------------------
        # Use a generator to avoid upfront computation of all noise times
        # ---------------------------------------------------------------------

        def iter_args():
            prev_args = None
            for i in range(n_samples):
                if i == 0:
                    args = next(arguments_generator)
                    prev_args = args
                elif (i % command_line_arguments['n_noise_realizations'] == 0):
                    args = next(arguments_generator)
                    prev_args = args
                else:
                    # Reuse waveform params but draw a new noise time
                    args = dict(prev_args,
                                event_tuple=next(noise_times))
                    prev_args = args
                yield (i, args)
        
        
        results_list = [None] * n_samples
        
        
        
        # run them in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=config['n_processes']) as executor:
            # submit all tasks
#            futures = {executor.submit(_worker, arg): arg[0] for arg in args_list}
            futures = {executor.submit(_worker, arg): arg[0] for arg in iter_args()}
            # iterate as they complete, updating progress
            for future in tqdm(concurrent.futures.as_completed(futures),
                            total=n_samples, ncols=80, unit='sample'):
                idx, res = future.result()
                results_list[idx] = res

        # ---------------------------------------------------------------------
        # Unpack results exactly as before
        # ---------------------------------------------------------------------
        samples[sample_type], injection_parameters[sample_type] = zip(*results_list)
        print('Sample generation completed!\n')
        
        
        
#        # ---------------------------------------------------------------------
#        # If we do not need to generate any samples, skip ahead:
#        # ---------------------------------------------------------------------

#        if n_samples == 0:
#            print('Done! (n_samples=0)\n')
#            continue

#        # ---------------------------------------------------------------------
#        # Initialize queues for the simulation arguments and the results
#        # ---------------------------------------------------------------------

#        # Initialize a Queue and fill it with as many arguments as we
#        # want to generate samples
#        arguments_queue = Queue()
#        start = next(arguments_generator)  # Fetch the first set of arguments

#        for i in range(n_samples):
#            if i == 0:
#            # Always put the first arguments into the queue
#                arguments_queue.put((i, start))
#                prev_arguments_generator = start  # Save the first arguments to use later

#            elif (i % command_line_arguments['n_noise_realizations'] == 0):
#            # Fetch new arguments from the generator every 'n_noise_realizations' steps
#               new_args = next(arguments_generator)
#                arguments_queue.put((i, new_args))
#                prev_arguments_generator = new_args  # Update the previous arguments
#            else:
#            # Generate new arguments with the same waveform_params but a new noise realization
#                new_args = {
#                    'static_arguments': prev_arguments_generator['static_arguments'],
#                    'event_tuple': next(noise_times),  # Fetch a new noise realization
#                    'add_glitches_noise': command_line_arguments['add_glitches_noise'],  # Add glitches parameter
#                    'add_glitches_injection': command_line_arguments['add_glitches_injection'],  # Add glitches parameter
#                    'detector': command_line_arguments['detectors'],  # Add detectors parameter
#                    'waveform_params': prev_arguments_generator['waveform_params']  # Keep the same waveform_params
#                }
        
#                arguments_queue.put((i, new_args))
#                prev_arguments_generator = new_args  # Update the previous arguments to include the new noise realization

#        # Initialize a Queue and a list to store the generated samples
#        results_queue = Queue()
#        results_list = []

#        # ---------------------------------------------------------------------
#        # Use process-based multiprocessing to generate samples in parallel
#        # ---------------------------------------------------------------------

#        # Use a tqdm context manager for the progress bar
#        tqdm_args = dict(total=n_samples, ncols=80, unit='sample')
#        with tqdm(**tqdm_args) as progressbar:

#            # Keep track of all running processes
#            list_of_processes = []

#            # While we haven't produced as many results as desired, keep going
#            while len(results_list) < n_samples:
    
#                # -------------------------------------------------------------
#                # Loop over processes to see if anything finished or got stuck
#                # -------------------------------------------------------------
                
#                for process_dict in list_of_processes:
        
#                    # Get the process object and its current runtime
#                    process = process_dict['process']
#                    runtime = time.time() - process_dict['start_time']
        
#                    # Check if the process is still running when it should
#                    # have terminated already (according to max_runtime)
#                    if process.is_alive() and (runtime > max_runtime):
            
#                        # Kill process that's been running too long
#                        process.terminate()
#                        process.join()
#                        list_of_processes.remove(process_dict)
            
#                        # Add new arguments to queue to replace the failed ones
#                        new_arguments = next(arguments_generator)
#                        arguments_queue.put(new_arguments)
        
#                    # If process has terminated already
#                    elif not process.is_alive():
            
#                        # If the process failed, add new arguments to queue
#                        if process.exitcode != 0:
#                            new_arguments = next(arguments_generator)
#                            arguments_queue.put(new_arguments)
            
#                        # Remove process from the list of running processes
#                        list_of_processes.remove(process_dict)

#                # -------------------------------------------------------------
#                # Start new processes if necessary
#                # -------------------------------------------------------------
    
#                # Start new processes until the arguments_queue is empty, or
#                # we have reached the maximum number of processes
#                while (arguments_queue.qsize() > 0 and
#                       len(list_of_processes) < config['n_processes']):
                    
#                    # Get arguments from queue and start new process
#                    arguments = arguments_queue.get()
#                    p = Process(target=queue_worker,
#                                kwargs=dict(arguments=arguments,
#                                            results_queue=results_queue))
        
#                    # Remember this process and its starting time
#                    process_dict = dict(process=p, start_time=time.time())
#                    list_of_processes.append(process_dict)
                    
#                    # Finally, start the process
#                    p.start()

#                # -------------------------------------------------------------
#                # Move results from results_queue to results_list
#                # -------------------------------------------------------------

#                # Without this part, the results_queue blocks the worker
#                # processes so that they won't terminate
#                while results_queue.qsize() > 0:
#                    results_list.append(results_queue.get())

#                # Update the progress bar based on the number of results
#                progressbar.update(len(results_list) - progressbar.n)

#                # Sleep for some time before we check the processes again
#                time.sleep(0.5)
            
#        # ---------------------------------------------------------------------
#        # Process results in the results_list
#        # ---------------------------------------------------------------------

#        # Sort results by index and extract just the results
#        results_list.sort(key=lambda x: x[0])  # Sort by the first element of the tuple, the index
        
##        print(results_list)
        
#        results_list = [result[1] for result in results_list]  # Extract the second element, the actual result

#        # Separate the samples and the injection parameters
#        samples[sample_type], injection_parameters[sample_type] = \
#            zip(*results_list)


        print('Sample generation completed!\n')

    # -------------------------------------------------------------------------
    # Compute the normalization parameters for this file
    # -------------------------------------------------------------------------

    print('Computing normalization parameters for sample...', end=' ')

    # Gather all samples (with and without injection) in one list
    all_samples = list(samples['injection_samples']) + list(samples['noise_samples'])

    # Group all samples by detector
    if detectors_set == {'H1'}:
        h1_samples = [_['h1_strain'] for _ in all_samples]
        h1_samples = np.vstack(h1_samples)
    elif detectors_set == {'L1'}:
        l1_samples = [_['l1_strain'] for _ in all_samples]
        l1_samples = np.vstack(l1_samples)
    elif detectors_set == {'H1', 'L1'}:
        h1_samples = [_['h1_strain'] for _ in all_samples]
        l1_samples = [_['l1_strain'] for _ in all_samples]
        
        h1_samples = np.vstack(h1_samples)
        l1_samples = np.vstack(l1_samples)
        
#   v1_samples = [_['v1_strain'] for _ in all_samples]

    # Stack recordings along first axis

#    a = static_arguments['seconds_before_event']
#    b = static_arguments['seconds_after_event']
#    end_idx = (a+b)*static_arguments['target_sampling_rate']

#    h1_samples = [row[0:2048] for row in h1_samples]
#    l1_samples = [row[0:2048] for row in l1_samples]
#    v1_samples = [row[0:2048] for row in v1_samples]

    
    
#   v1_samples = np.vstack(v1_samples)

    # Compute the mean and standard deviation for both detectors as the median
    # of the means / standard deviations for each sample. This is more robust
    # towards outliers than computing "global" parameters by concatenating all
    # samples and treating them as a single, long time series.
    
    if detectors_set == {'H1'}:
        normalization_parameters = \
        dict(h1_mean=np.median(np.mean(h1_samples, axis=1), axis=0),
            h1_std=np.median(np.std(h1_samples, axis=1), axis=0))

    elif detectors_set == {'L1'}:
        normalization_parameters = \
        dict(l1_mean=np.median(np.mean(l1_samples, axis=1), axis=0),
            l1_std=np.median(np.std(l1_samples, axis=1), axis=0))
    
    elif detectors_set == {'H1', 'L1'}:
        normalization_parameters = \
            dict(h1_mean=np.median(np.mean(h1_samples, axis=1), axis=0),
                l1_mean=np.median(np.mean(l1_samples, axis=1), axis=0),
                h1_std=np.median(np.std(h1_samples, axis=1), axis=0),
                l1_std=np.median(np.std(l1_samples, axis=1), axis=0))

    print('Done!\n')

    # -------------------------------------------------------------------------
    # Create a SampleFile dict from list of samples and save it as an HDF file
    # -------------------------------------------------------------------------

    print('Saving the results to HDF file ...', end=' ')

    # Initialize the dictionary that we use to create a SampleFile object
    sample_file_dict = dict(command_line_arguments=command_line_arguments,
                            injection_parameters=dict(),
                            injection_samples=dict(),
                            noise_samples=dict(),
                            noise_parameters=dict(),
                            normalization_parameters=normalization_parameters,
                            static_arguments=static_arguments)

    # Collect and add samples (with and without injection)
    for sample_type in ('injection_samples', 'noise_samples'):
        
        if detectors_set == {'H1'}:
            
            for key in ('event_time', 'h1_strain'):
                if samples[sample_type]:
                    value = np.array([_[key] for _ in list(samples[sample_type])])
                        
                else:
                    value = None
                sample_file_dict[sample_type][key] = value
            
        elif detectors_set == {'L1'}:
            for key in ('event_time', 'l1_strain'):
                if samples[sample_type]:
                    value = np.array([_[key] for _ in list(samples[sample_type])])
                        
                else:
                    value = None
                sample_file_dict[sample_type][key] = value
        
        elif detectors_set == {'H1', 'L1'}:
            for key in ('event_time', 'h1_strain', 'l1_strain'):
                if samples[sample_type]:
                    value = np.array([_[key] for _ in list(samples[sample_type])])
                        
                else:
                    value = None
                sample_file_dict[sample_type][key] = value

    # Collect and add injection_parameters (ignore noise samples here, because
    # for those, the injection_parameters are always None)
    
    if detectors_set == {'H1'}:
#        other_keys = ['h1_signal', 'h1_signal_whitened', 'h1_snr', 'scale_factor', 'psd_noise_h1']
#        other_keys = ['h1_signal', 'h1_snr', 'scale_factor', 'psd_noise_h1']
        other_keys = ['h1_signal_whitened', 'h1_snr', 'scale_factor']
    
    elif detectors_set == {'L1'}:
#        other_keys = ['l1_signal', 'l1_signal_whitened', 'l1_snr', 'scale_factor', 'psd_noise_l1']
#        other_keys = ['l1_signal', 'l1_snr', 'scale_factor', 'psd_noise_l1']
        other_keys = ['l1_signal_whitened', 'l1_snr', 'scale_factor']
    
    elif detectors_set == {'H1', 'L1'}:
#        other_keys = ['h1_signal', 'h1_signal_whitened', 'h1_snr', 'l1_signal', 'l1_signal_whitened', 'l1_snr', 'scale_factor', 'psd_noise_h1', 'psd_noise_l1']
#        other_keys = ['h1_signal', 'h1_snr', 'l1_signal', 'l1_snr', 'scale_factor', 'psd_noise_h1', 'psd_noise_l1']
        other_keys = ['h1_signal_whitened', 'h1_snr', 'l1_signal_whitened', 'l1_snr', 'scale_factor']
#        other_keys = ['h1_snr', 'l1_snr', 'scale_factor']
    
#    other_keys = ['h1_signal', 'h1_snr', 'l1_signal', 'l1_snr', 'scale_factor']
    
        
    for key in list(variable_arguments + other_keys):
        if injection_parameters['injection_samples']:
            value = np.array([_[key] for _ in
                              injection_parameters['injection_samples']])

        
        
#            for i in range(n_samples):
#                if neglat_seconds > 0:
#                    truncate_length = static_arguments["target_sampling_rate"]*neglat_seconds
#                    temp_h1 = np.copy(sample_file_dict['injection_samples']['h1_signal'][i])
#                    temp_l1 = np.copy(sample_file_dict['injection_samples']['l1_signal'][i])
#                    temp_v1 = np.copy(sample_file_dict['injection_samples']['v1_signal'][i])
#                    temp_h1[-truncate_length:] = np.zeros(truncate_length)
#                    temp_l1[-truncate_length:] = np.zeros(truncate_length)
#                    temp_v1[-truncate_length:] = np.zeros(truncate_length)
#                    sample_file_dict['injection_samples']['h1_signal'][i] = temp_h1
#                    sample_file_dict['injection_samples']['l1_signal'][i] = temp_l1
#                    sample_file_dict['injection_samples']['v1_signal'][i] = temp_v1
                    
#            value = np.array([_[key] for _ in injection_parameters['injection_samples']])
                        
        else:
            value = None
        sample_file_dict['injection_parameters'][key] = value
        
    if detectors_set == {'H1'}:
#        noise_keys = ['h1_signal', 'h1_signal_whitened']
#        noise_keys = ['h1_signal']
        noise_keys = ['h1_signal_whitened']
        
    elif detectors_set == {'L1'}:
#        noise_keys = ['l1_signal', 'l1_signal_whitened']
        noise_keys = ['l1_signal_whitened']
#        noise_keys = ['l1_signal_whitened']
    
    elif detectors_set == {'H1', 'L1'}:
#        noise_keys = ['h1_signal', 'h1_signal_whitened', 'l1_signal', 'l1_signal_whitened'] 
#        noise_keys = ['h1_signal', 'l1_signal']   
        noise_keys = ['h1_signal_whitened', 'l1_signal_whitened']   
    
#    noise_keys = ['h1_signal', 'l1_signal']    
    for key in list(noise_keys):
        if injection_parameters['noise_samples']:
            value = np.array([_[key] for _ in injection_parameters['noise_samples']])
        else:
            value = None
        sample_file_dict['noise_parameters'][key] = value
                

    # Construct the path for the output HDF file
    output_dir = os.path.join('.', 'output')
#    output_dir = '/group/pmc005/cchatterjee/SNR_time_series_sample_files/SNR_Variable/' # change for SNR variable
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    sample_file_path = os.path.join(output_dir, config['output_file_name'])

    # Create the SampleFile object and save it to the specified output file
#    print(sample_file_dict)
    sample_file = SampleFile(data=sample_file_dict)
    sample_file.to_hdf(file_path=sample_file_path)

    print('Done!')

    # Get file size in MB and print the result
    sample_file_size = os.path.getsize(sample_file_path) / 1024**2
    print('Size of resulting HDF file: {:.2f}MB'.format(sample_file_size))
    print('')

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    # PyCBC always create a copy of the waveform parameters file, which we
    # can delete at the end of the sample generation process
    duplicate_path = os.path.join('.', config['waveform_params_file_name'])
    if os.path.exists(duplicate_path):
        os.remove(duplicate_path)

    # Print the total run time
    print('Total runtime: {:.1f} seconds!'.format(time.time() - script_start))
    print('')
