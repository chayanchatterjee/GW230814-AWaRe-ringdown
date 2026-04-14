"""
Provide classes and functions for reading and writing HDF files.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from __future__ import print_function

import numpy as np
import h5py
import os
import sys

import gwpy
from gwpy.timeseries import TimeSeries
from matplotlib import pyplot as plt

from pycbc.catalog import Catalog
#from pycbc.types.timeseries import TimeSeries
from lal import LIGOTimeGPS


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_file_paths(directory, extensions=None):
    """
    Take a directory and return the paths to all files in this
    directory and its subdirectories. Optionally filter out only
    files specific extensions.

    Args:
        directory (str): Path to a directory.
        extensions (list): List of allowed file extensions,
            for example: `['hdf', 'h5']`.

    Returns:
        List of paths of all files matching the above descriptions.
    """

    file_paths = []

    # Walk over the directory and find all files
    for path, dirs, files in os.walk(directory):
        for f in files:
            file_paths.append(os.path.join(path, f))

    # If a list of extensions is provided, only keep the corresponding files
    if extensions is not None:
        file_paths = [_ for _ in file_paths if any([_.endswith(ext) for
                                                    ext in extensions])]

    return file_paths


# -----------------------------------------------------------------------------


def get_strain_from_gwf_file(gwf_file_paths,
                             gps_time,
                             interval_width,
                             original_sampling_rate=4096,
                             target_sampling_rate=4096,
                             as_pycbc_timeseries=False):
    """
    For a given `gps_time`, select the interval of length
    `interval_width` (centered around `gps_time`) from the HDF files
    specified in `hdf_file_paths`, and resample them to the given
    `target_sampling_rate`.

    Args:
        hdf_file_paths (dict): A dictionary with keys `{'H1', 'L1'}`,
            which holds the paths to the HDF files containing the
            interval around `gps_time`.
        gps_time (int): A (valid) background noise time (GPS timestamp).
        interval_width (int): The length of the strain sample (in
            seconds) to be selected from the HDF files.
        original_sampling_rate (int): The original sampling rate (in
            Hertz) of the HDF files sample. Default is 4096.
        target_sampling_rate (int): The sampling rate (in Hertz) to
            which the strain should be down-sampled (if desired). Must
            be a divisor of the `original_sampling_rate`.
        as_pycbc_timeseries (bool): Whether to return the strain as a
            dict of numpy arrays or as a dict of objects of type
            `pycbc.types.timeseries.TimeSeries`.

    Returns:
        A dictionary with keys `{'H1', 'L1'}`. For each key, the
        dictionary contains a strain sample (as a numpy array) of the
        given length, centered around `gps_time`, (down)-sampled to
        the desired `target_sampling_rate`.
    """

    # -------------------------------------------------------------------------
    # Perform some basic sanity checks on the arguments
    # -------------------------------------------------------------------------

    assert isinstance(gps_time, int), \
        'time is not an integer!'
    assert isinstance(interval_width, int), \
        'interval_width is not an integer'
    assert isinstance(original_sampling_rate, int), \
        'original_sampling_rate is not an integer'
    assert isinstance(target_sampling_rate, int), \
        'target_sampling_rate is not an integer'
    assert original_sampling_rate % target_sampling_rate == 0, \
        'Invalid target_sampling_rate: Not a divisor of ' \
        'original_sampling_rate!'

    # -------------------------------------------------------------------------
    # Read out the strain from the HDF files
    # -------------------------------------------------------------------------

    # Compute the offset = half the interval width (intervals are centered
    # around the given gps_time)
    offset = int(interval_width / 2)

    # Compute the resampling factor
    sampling_factor = int(original_sampling_rate / target_sampling_rate)

    # Store the sample we have selected from the HDF files
    sample = dict()

    # Loop over both detectors
    for detector in ('H1', 'L1'):

        # Extract the path to the HDF file
        file_path = gwf_file_paths

        # Read in the HDF file and select the noise sample
    #    with h5py.File(file_path, 'r') as hdf_file:

        strain = TimeSeries.read(file_path, 'H1:GDS-CALIB_STRAIN_DC')
        # Get the start_time and compute array indices
        ### FIXME: data.times.value[0] or strain.times.value[0]? ##########
#        start_time = data.times.value[0]
        start_time = strain.times.value[0]
        start_idx = \
                (gps_time - start_time - offset) * original_sampling_rate
        end_idx = \
                (gps_time - start_time + offset) * original_sampling_rate

        # Select the sample from the strain
        sample[detector] = strain[start_idx:end_idx]

        # Down-sample the selected sample to the target_sampling_rate
        sample[detector] = sample[detector][::sampling_factor]

    # -------------------------------------------------------------------------
    # Convert to PyCBC time series, if necessary
    # -------------------------------------------------------------------------

    # If we just want a plain numpy array, we can return it right away
    if not as_pycbc_timeseries:
        return sample

    # Otherwise we need to convert the numpy array to a time series first
    else:

        # Initialize an empty dict for the time series results
        timeseries = dict()

        # Convert strain of both detectors to a TimeSeries object
        for detector in ('H1', 'L1'):

            timeseries[detector] = \
                pycbc.types.timeseries.TimeSeries(initial_array=sample[detector],
                           delta_t=1.0/target_sampling_rate,
                           epoch=LIGOTimeGPS(gps_time - offset))

        return timeseries


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class NoiseTimeline_DC:
    """
    A ``NoiseTimeline`` object stores information about the data
    quality and hardware injection flags of the files in the given
    `background_data_directory`. This is information is read in only
    once at the beginning of the sample generation and can then be
    utilized to quickly sample "valid" noise times, that is, GPS times
    where the files in `background_data_directory` provide data which
    pass certain desired quality criteria.
    
    Args:
        background_data_directory (str): Path to the directory which
            contains the raw data (HDF files). These files may also be
            distributed over several subdirectories.
        random_seed (int): Seed for the random number generator which
            is used for sampling valid noise times.
        verbose (bool): Whether or not this instance should print
            logging information to the command line.
    """

    def __init__(self,
                 background_data_directory,
                 random_seed=42,
                 verbose=False):

        # Store the directory and sampling rate of the raw HDF files
        self.background_data_directory = background_data_directory

        # Print debug messages or not?
        self.verbose = verbose

        # Create a new random number generator with the given seed to
        # decouple this from the global numpy RNG (for reproducibility)
        self.rng = np.random.RandomState(seed=random_seed)

        # Get the list of all HDF files in the specified directory
        self.vprint('Getting GWF file paths...', end=' ')
        self.gwf_file_paths = get_file_paths(self.background_data_directory,
                                             extensions=['gwf'])
        self.vprint('Done!')

        # Read in the meta information and masks from GWF files
        self.vprint('Reading information from GWF files', end=' ')
        self.gwf_files = self._get_gwf_files()
        self.vprint('Done!')

        # Build the timeline for these GWF files
        self.vprint('Building timeline object...', end=' ')
        self.timeline = self._build_timeline()
        self.vprint('Done!')

    # -------------------------------------------------------------------------

    def vprint(self, string, *args, **kwargs):
        """
        Verbose printing: Wrapper around `print()` to only call it if
        `self.verbose` is set to true.

        Args:
            string (str): String to be printed if `self.verbose`
                is `True`.
            *args: Arguments passed to `print()`.
            **kwargs: Keyword arguments passed to `print()`.
        """

        if self.verbose:
            print(string, *args, **kwargs)
            sys.stdout.flush()

    # -------------------------------------------------------------------------

    def _get_gwf_files(self):

        # Keep track of all the files whose information we need to store
        gwf_files = []

        # Open every HDF file once to read in the meta information as well
        # as the injection and data quality (DQ) masks
        n_files = len(self.gwf_file_paths)

        for i, gwf_file_path in enumerate(self.gwf_file_paths):

            self.vprint('({:>4}/{:>4})...'.format(i, n_files), end=' ')

            data = TimeSeries.read(gwf_file_path, 'H1:GDS-CALIB_STRAIN_DC')

            # Select necessary information from the HDF file
            start_time = data.times.value[0]
            detector = 'H1'
            duration = 4096
            dq_mask = np.ones(4096)*127
#            inj_mask = np.array(f['quality']['injections']['Injmask'],
#                                    dtype=np.int32)
#            dq_mask = np.array(f['quality']['simple']['DQmask'],
#                                   dtype=np.int32)

            # Perform some basic sanity checks
            assert detector in ['H1', 'L1'], \
                    'Invalid detector {}!'.format(detector)
            assert duration == len(dq_mask), \
                    'Length of InjMask or DQMask does not match the duration!'


            # Collect this information in a dict
            gwf_files.append(dict(file_path=gwf_file_path,
                                  detector=detector,
                                  duration=duration,
                                  dq_mask=dq_mask,
                                  start_time=start_time))

            self.vprint('\033[15D\033[K', end='')

        # Sort the read in HDF files by start time and return them
        self.vprint('({:>4}/{:>4})...'.format(n_files, n_files), end=' ')
        return sorted(gwf_files, key=lambda _: _['start_time'])

    # -------------------------------------------------------------------------

    def _build_timeline(self):

        # Get the size of the arrays that we need to initialize
        n_entries = self.gps_end_time - self.gps_start_time

        # Initialize the empty timeline
        timeline = dict(h1_dq_mask=np.zeros(n_entries, dtype=np.int32),
                        l1_dq_mask=np.zeros(n_entries, dtype=np.int32))
#			v1_dq_mask=np.zeros(n_entries, dtype=np.int32))
        # Add information from HDF files to timeline
        for gwf_file in self.gwf_files:

            # Define some shortcuts
            detector = gwf_file['detector']
            dq_mask = gwf_file['dq_mask']
#            inj_mask = gwf_file['inj_mask']

            # Map start/end from GPS time to array indices
            idx_start = int(gwf_file['start_time']) - int(self.gps_start_time)
            idx_end = int(idx_start) + int(gwf_file['duration'])

            # Add the mask information to the correct detector
            timeline['h1_dq_mask'][idx_start:idx_end] = dq_mask
            timeline['l1_dq_mask'][idx_start:idx_end] = dq_mask
     #       else:
     #           timeline['v1_inj_mask'][idx_start:idx_end] = inj_mask
     #           timeline['v1_dq_mask'][idx_start:idx_end] = dq_mask


        # Return the completed timeline
        return timeline

    # -------------------------------------------------------------------------

    def is_valid(self,
                 gps_time,
                 delta_t=16):
        """
        For a given `gps_time`, check if is a valid time to sample
        noise from by checking if all data points in the interval
        `[gps_time - delta_t, gps_time + delta_t]` have the specified
        `dq_bits` and `inj_bits` set.
        
        .. seealso:: For more information about the `dq_bits` and
            `inj_bits`, check out the website of the GW Open Science
            Center, which explains these for the case of O1:
            
                https://www.gw-openscience.org/archive/dataset/O1

        Args:
            gps_time (int): The GPS time whose validity we are checking.
            delta_t (int): The number of seconds around `gps_time`
                which we also want to be valid (because the sample will
                be an interval).
            dq_bits (tuple): The Data Quality Bits which one would like
                to require (see note above).
                *For example:* `dq_bits=(0, 1, 2, 3)` means that the
                data quality needs  to pass all tests up to `CAT3`.
            inj_bits (tuple): The Injection Bits which one would like
                to require (see note above).
                *For example:* `inj_bits=(0, 1, 2, 4)` means that only
                continuous wave (CW) injections are permitted; all
                recordings containing any of other type of injection
                will be invalid for sampling.

        Returns:
            `True` if `gps_time` is valid, otherwise `False`.
        """ 
        
        return True

    # -------------------------------------------------------------------------

    def sample(self,
               delta_t=16,
               return_paths=False):

        """
        Randomly sample a time from `[gps_start_time, gps_end_time]`
        which passes the :func:`NoiseTimeline.is_valid()` test.

        Args:
            delta_t (int): For an explanation, see
                :func:`NoiseTimeline.is_valid()`.
            dq_bits (tuple): For an explanation, see
                :func:`NoiseTimeline.is_valid()`.
            inj_bits (tuple): For an explanation, see
                :func:`NoiseTimeline.is_valid()`.
            return_paths (bool): Whether or not to return the paths to
                the HDF files containing the `gps_time`.

        Returns:
            A valid GPS time and optionally a `dict` with the file
            paths to the HDF files containing that GPS time (keys will
            correspond to the different detectors).
        """

        # Keep sampling random times until we find a valid one...
        while True:

            # Randomly choose a GPS time between the start and end
            gps_time = self.rng.randint(int(self.gps_start_time) + delta_t,
                                        int(self.gps_end_time) - delta_t)

            if return_paths:
                return gps_time, self.get_file_paths_for_time(gps_time)
            else:
                return gps_time

    # -------------------------------------------------------------------------

    def get_file_paths_for_time(self, gps_time):
        """
        For a given (valid) GPS time, find the two HDF files (for the
        two detectors H1 and L1) which contain the corresponding strain.

        Args:
            gps_time (int): A valid GPS time stamp.

        Returns:
            A dictionary with keys `{'H1', 'L1'}` containing the paths
            to the HDF files, or None if no such files could be found.
        """

        # Keep track of the results, i.e., the paths to the HDF files
        result = dict()

        # Loop over all HDF files to find the ones containing the given time
        for gwf_file in self.gwf_files:

            # Get the start and end time for the current HDF file
            start_time = gwf_file['start_time']
            end_time = start_time + gwf_file['duration']

            # Check if the given GPS time falls into the interval of the
            # current HDF file, and if so, store the file path for it
            if start_time < gps_time < end_time:
                result[gwf_file['detector']] = gwf_file['file_path']

            # If both files were found, we are done!
            if 'H1' in result.keys() and 'L1' in result.keys():
                return result

        # If we didn't both files, return None
        return None

    # -------------------------------------------------------------------------

    def idx2gps(self, idx):
        """
        Map an index to a GPS time by correcting for the start time of
        the observation run, as determined from the HDF files.

        Args:
            idx (int): An index of a time series array (covering an
                observation run).

        Returns:
            The corresponding GPS time.
        """

        return idx + self.gps_start_time

    # -------------------------------------------------------------------------

    def gps2idx(self, gps):
        """
        Map an GPS time to an index by correcting for the start time of
        the observation run, as determined from the HDF files.

        Args:
            gps (int): A GPS time belonging to a point in time between
                the start and end of an observation run.

        Returns:
            The corresponding time series index.
        """

        return gps - self.gps_start_time

    # -------------------------------------------------------------------------

    @property
    def gps_start_time(self):
        """
        The GPS start time of the observation run.
        """

        return int(self.gwf_files[0]['start_time'])

    # -------------------------------------------------------------------------

    @property
    def gps_end_time(self):
        """
        The GPS end time of the observation run.
        """

        return int(self.gwf_files[-1]['start_time'] + \
            self.gwf_files[-1]['duration'])

