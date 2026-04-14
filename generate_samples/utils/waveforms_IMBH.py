"""
Provide methods for generating and processing simulated
gravitational-wave waveforms.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from __future__ import print_function

import numpy as np
from scipy.signal.windows import tukey

from pycbc.distributions import JointDistribution, read_params_from_config, \
    read_constraints_from_config, read_distributions_from_config
from pycbc.transforms import apply_transforms, read_transforms_from_config
from pycbc.workflow import WorkflowConfigParser
from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.detector import Detector
from pycbc.types.timeseries import TimeSeries
import math


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class WaveformParameterGenerator(object):
    """
    :class:`WaveformParameterGenerator` objects are essentially just a
    simple convenience wrapper to construct the joint probability
    distribution (and provide a method to draw samples from it) of the
    parameters specified by the `[variable_args]` section of an
    `*.ini` configuration file and their distributions as defined in
    the corresponding `[prior-*]` sections.
    
    Args:
        config_file (str): Path to the `*.ini` configuration file,
            which contains the information about the parameters to be
            generated and their distributions.
        random_seed (int): Seed for the random number generator.
            Caveat: We can only set the seed of the global numpy RNG.
    """

    def __init__(self,
                 config_file,
                 random_seed):

        # Fix the seed for the random number generator
        np.random.seed(random_seed)

        # Read in the configuration file using a WorkflowConfigParser.
        # Note that the argument `configFiles` has to be a list here,
        # so we need to wrap the `config_file` argument accordingly...
        config_file = WorkflowConfigParser(configFiles=[config_file])

        # Extract variable arguments and constraints
        # We don't need the static_args here, hence they do not get amended.
        self.var_args, _ = read_params_from_config(config_file)
        self.constraints = read_constraints_from_config(config_file)

        # Extract distributions
        dist = read_distributions_from_config(config_file)

        # Extract transformations
        self.trans = read_transforms_from_config(config_file, section="waveform_transforms")

        # Set up a joint distribution to sample from
        self.pval = JointDistribution(self.var_args,
                                      *dist,
                                      **{'constraints': self.constraints})

    # -------------------------------------------------------------------------

    def draw(self):
        """
        Draw a sample from the joint distribution and construct a
        dictionary that maps the parameter names to the values
        generated for them.

        Returns:
            A `dict` containing a the names and values of a set of
            randomly sampled waveform parameters (e.g., masses, spins,
            position in the sky, ...).
        """
        values = apply_transforms(self.pval.rvs(), self.trans)[0]
        result = dict(zip(self.var_args, values))

        return result


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def fade_on(timeseries,
            alpha=0.25):
    """
    Take a PyCBC time series and use a one-sided Tukey window to "fade
    on" the waveform (to reduce discontinuities in the amplitude).

    Args:
        timeseries (pycbc.types.timeseries.TimeSeries): The PyCBC
            TimeSeries object to be faded on.
        alpha (float): The alpha parameter for the Tukey window.

    Returns:
        The `timeseries` which has been faded on.
    """

    # Save the parameters from the time series we are about to fade on
    delta_t = timeseries.delta_t
    epoch = timeseries.start_time
    duration = timeseries.duration
    sample_rate = timeseries.sample_rate

    # Create a one-sided Tukey window for the turn on
    window = tukey(M=int(duration * sample_rate), alpha=alpha)
    window[int(0.5*len(window)):] = 1

    # Apply the one-sided Tukey window for the fade-on
    ts = window * np.array(timeseries)

    # Create and return a TimeSeries object again from the resulting array
    # using the original parameters (delta_t and epoch) of the time series
    return TimeSeries(initial_array=ts,
                      delta_t=delta_t,
                      epoch=epoch)


def get_waveform(static_arguments,
                 waveform_params):
    """
    Simulate a waveform (using methods provided by PyCBC / LALSuite)
    based on the `static_arguments` (which define, e.g., the waveform
    model to be used) and the `waveform_params`, which specify the
    physical parameters of the waveform (e.g., the masses and spins).
    
    .. note::
       The actual simulation of the waveform is, depending on your
       choice of the `domain`, performed by the PyCBC methods
       :func:`get_td_waveform()` and :func:`get_fd_waveform()`,
       respectively.
       These take as arguments a combination of the `static_arguments`
       and the `waveform_params.` A (more) comprehensive explanation of
       the parameters that are supported by these methods can be found
       in the `PyCBC documentation <https://pycbc.org/pycbc/latest/html/
       pycbc.waveform.html#pycbc.waveform.waveform.get_td_waveform>`_.
       Currently, however, only the following keys are actually passed
       to the simulation routines:
       
       .. code-block:: python
          
          {'approximant', 'coa_phase', 'delta_f', 'delta_t',
           'distance', 'f_lower', 'inclination', 'mass1', 'mass2',
           'spin1z', 'spin2z'}
           
    .. warning::
       If you want to use a different waveform model or a different
       parameter space, you may need to edit this function according
       to your exact needs!
    
    
    Args:
        static_arguments (dict): The static arguments (e.g., the
            waveform approximant and the sampling rate) defined in the
            `*.ini` configuration file, which specify technical aspects
            of the simulation process.
        waveform_params (dict): The physical parameters of the
            waveform to be simulated, such as the masses or the
            position in the sky. Usually, these values are sampled
            using a :class:`WaveformParameterGenerator` instance,
            which is based in the variable arguments section in the
            `*.ini` configuration file.
    
    Returns:
        A tuple `(h_plus, h_cross)` with the two polarization modes of
        the simulated waveform, resized to the desired length.
    """

    # Check if we are using a time domain (TD) or frequency domain (FD)
    # approximant and retrieve the required parameters for the simulation
    if static_arguments['domain'] == 'time':
        simulate_waveform = get_td_waveform
        length = int(static_arguments['td_length'])
    elif static_arguments['domain'] == 'frequency':
        simulate_waveform = get_fd_waveform
        length = int(static_arguments['fd_length'])
    else:
        raise ValueError('Invalid domain! Must be "time" or "frequency"!')

    # Collect all the required parameters for the simulation from the given
    # static and variable parameters

    if 'chirp_distance' in waveform_params:    

        config_file_path = '/fred/oz016/Chayan/samplegen_old/config_files/waveform_params_NRSur.ini'
        config_file = WorkflowConfigParser(configFiles=[config_file_path])
        trans = read_transforms_from_config(config_file,section="waveform_transforms")

        chirp_distance_array = waveform_params['chirp_distance']
        mass1_array = waveform_params['mass1'][()]
        mass2_array = waveform_params['mass2'][()]

        samples = {
            'chirp_distance': chirp_distance_array,
            'mass1': mass1_array,
            'mass2': mass2_array
        }
        values = apply_transforms(samples, trans)
        distance = values['distance'][()]


    if 'spin1_a' in waveform_params:
        
        spin1_a = waveform_params['spin1_a']
        spin2_a = waveform_params['spin2_a']
        spin1_polar = waveform_params['spin1_polar']
        spin2_polar = waveform_params['spin2_polar']
        spin1_azimuthal = waveform_params['spin1_azimuthal']
        spin2_azimuthal = waveform_params['spin2_azimuthal']

        spin1x = spin1_a * math.sin(spin1_polar) * math.cos(spin1_azimuthal)
        spin2x = spin2_a * math.sin(spin2_polar) * math.cos(spin2_azimuthal)
        spin1y = spin1_a * math.sin(spin1_polar) * math.sin(spin1_azimuthal)
        spin2y = spin2_a * math.sin(spin2_polar) * math.sin(spin2_azimuthal)
        spin1z = spin1_a * math.cos(spin1_polar)
        spin2z = spin2_a * math.cos(spin2_polar)
        
        #### For sampling over M_total and q #####
        
        config_file_path = '/home/chattec/samplegen/config_files/waveform_params_IMBH.ini'
        config_file = WorkflowConfigParser(configFiles=[config_file_path])
        trans = read_transforms_from_config(config_file,section="waveform_transforms")

        mtotal_array = waveform_params['mtotal'][()]
        q_array = waveform_params['q'][()]

        samples = {
            'mtotal': mtotal_array,
            'q': q_array
        }
        
        values = apply_transforms(samples, trans)
        mass1 = values['mass1'][()]
        mass2 = values['mass2'][()]
        
        #########################################


        simulation_parameters = dict(approximant=static_arguments['approximant'],
                                   coa_phase=waveform_params['coa_phase'],
                                   delta_f=static_arguments['delta_f'],
                                   delta_t=static_arguments['delta_t'],
                                   distance=static_arguments['distance'], # Change for SNR Variable
#                                   chirp_distance=waveform_params['chirp_distance'],
#                                   distance=distance,
                                   f_lower=static_arguments['f_lower'],
#                                   f_ref=static_arguments['f_ref'],
                                   inclination=waveform_params['inclination'],
                                   
                                   #### For sampling over M_total and q #####
                                   mass1=mass1,
                                   mass2=mass2,
                                   #########################################
                                   
#                                   mass1=waveform_params['mass1'],
#                                   mass2=waveform_params['mass2'],
                                   
                                   spin1x=spin1x,
                                   spin2x=spin2x,
                                   spin1y=spin1y,
                                   spin2y=spin2y,
                                   spin1z=spin1z,
                                   spin2z=spin2z)
#                                   mode_array=[(2,2)])

    elif 'eccentricity' in waveform_params:

        simulation_parameters = dict(approximant=static_arguments['approximant'],
                                   coa_phase=waveform_params['coa_phase'],
                                   delta_f=static_arguments['delta_f'],
                                   delta_t=static_arguments['delta_t'],
                                   distance=static_arguments['distance'], # Change for SNR Variable
#                                   chirp_distance=waveform_params['chirp_distance'],
#                                   distance=distance,
                                   eccentricity=waveform_params['eccentricity'],
                                   f_lower=static_arguments['f_lower'],
                                   f_ref=static_arguments['f_ref'],
                                   inclination=waveform_params['inclination'],
                                   mass1=waveform_params['mass1'],
                                   mass2=waveform_params['mass2'])
    
    
    elif 'spin1_a' not in waveform_params and 'eccentricity' not in waveform_params:
        simulation_parameters = dict(approximant=static_arguments['approximant'],
                                 coa_phase=waveform_params['coa_phase'],
                                 delta_f=static_arguments['delta_f'],
                                 delta_t=static_arguments['delta_t'],
                                 distance=static_arguments['distance'],  # Change for SNR Variable
#                                 chirp_distance=waveform_params['chirp_distance'],
#                                 distance=distance,
                                 f_lower=static_arguments['f_lower'],
#                                 f_ref=static_arguments['f_ref'],
                                 inclination=waveform_params['inclination'],
                                 mass1=waveform_params['msass1'],
                                 mass2=waveform_params['mass2'],
                                 spin1z=waveform_params['spin1z'],
                                 spin2z=waveform_params['spin2z'])
#                                 mode_array=[(2,2)])

    # Perform the actual simulation with the given parameters
    h_plus, h_cross = simulate_waveform(**simulation_parameters)

    # Apply the fade-on filter to them
    h_plus = fade_on(h_plus, alpha=static_arguments['tukey_alpha'])
    h_cross = fade_on(h_cross, alpha=static_arguments['tukey_alpha'])

    # Resize the simulated waveform to the specified length
    h_plus.resize(length)
    h_cross.resize(length)

    return h_plus, h_cross


# -----------------------------------------------------------------------------


def get_detector_signals(static_arguments,
                         waveform_params,
                         event_time,
                         detector,
                         waveform):
    """
    Project the raw `waveform` (i.e., the tuple `(h_plus, h_cross)`
    returned by :func:`get_waveform()`) onto the antenna patterns of
    the detectors in Hanford and Livingston. This requires the position
    of the source in the sky, which is contained in `waveform_params`.

    Args:
        static_arguments (dict): The static arguments (e.g., the
            waveform approximant and the sampling rate) defined in the
            `*.ini` configuration file.
        waveform_params (dict): The parameters that were used as inputs
            for the waveform simulation, although this method will only
            require the following parameters to be present:
        
                - ``ra`` = Right ascension of the source
                - ``dec`` = Declination of the source
                - ``polarization`` = Polarization angle of the source
                
        event_time (int): The GPS time for the event, which, by
            convention, is the time at which the simulated signal
            reaches its maximum amplitude in the `H1` channel.
        waveform (tuple): The pure simulated wavefrom, represented by
            a tuple `(h_plus, h_cross)`, which is usually generated
            by :func:`get_waveform()`.

    Returns:
        A dictionary with keys `{'H1', 'L1'}` that contains the pure
        signal as it would be observed at Hanford and Livingston.
    """

    # Retrieve the two polarization modes from the waveform tuple
    h_plus, h_cross = waveform

    # Extract the parameters we will need later for the projection
    right_ascension = 2*np.pi*waveform_params['ra']
    declination = np.arcsin(1-2*waveform_params['dec'])

#    right_ascension = waveform_params['ra']
#    declination = waveform_params['dec']
    polarization = waveform_params['polarization']
    
    if detector == ['H1']:
        detector_signals = {}
        # Calculate the antenna pattern for this detector
        f_plus, f_cross = \
            Detector('H1').antenna_pattern(right_ascension=right_ascension,
                                        declination=declination,
                                        polarization=polarization,
                                        t_gps=1376089759.8)

        # Calculate the time offset from H1 for this detector
        delta_t_h1 = \
            Detector('H1').time_delay_from_detector(other_detector=Detector('H1'),
                                                right_ascension=right_ascension,
                                                declination=declination,
                                                t_gps=1376089759.8)
        # Project the waveform onto the antenna pattern
        detector_signal = f_plus * h_plus + f_cross * h_cross
        
        # Map the signal from geocentric coordinates to the specific
        # reference frame of the detector. This depends on whether we have
        # simulated the waveform in the time or frequency domain:
        if static_arguments['domain'] == 'time':
            offset = 100 + delta_t_h1 + detector_signal.start_time
            detector_signal = detector_signal.cyclic_time_shift(offset)
            detector_signal.start_time = event_time - 100
        elif static_arguments['domain'] == 'frequency':
            offset = 100 + delta_t_h1
            detector_signal = detector_signal.cyclic_time_shift(offset)
            detector_signal.start_time = event_time - 100
            detector_signal = detector_signal.to_timeseries()
        else:
            raise ValueError('Invalid domain! Must be "time" or "frequency"!')
    
    elif detector == ['L1']:
        # Calculate the antenna pattern for this detector
        f_plus, f_cross = \
            Detector('L1').antenna_pattern(right_ascension=right_ascension,
                                        declination=declination,
                                        polarization=polarization,
                                        t_gps=1376089759.8)

        # Calculate the time offset from H1 for this detector
        delta_t_h1 = \
            Detector('L1').time_delay_from_detector(other_detector=Detector('H1'),
                                                right_ascension=right_ascension,
                                                declination=declination,
                                                t_gps=1376089759.8)
            
        # Project the waveform onto the antenna pattern
        detector_signal = f_plus * h_plus + f_cross * h_cross
        
        # Map the signal from geocentric coordinates to the specific
        # reference frame of the detector. This depends on whether we have
        # simulated the waveform in the time or frequency domain:
        if static_arguments['domain'] == 'time':
            offset = 100 + delta_t_h1 + detector_signal.start_time
            detector_signal = detector_signal.cyclic_time_shift(offset)
            detector_signal.start_time = event_time - 100
        elif static_arguments['domain'] == 'frequency':
            offset = 100 + delta_t_h1
            detector_signal = detector_signal.cyclic_time_shift(offset)
            detector_signal.start_time = event_time - 100
            detector_signal = detector_signal.to_timeseries()
        else:
            raise ValueError('Invalid domain! Must be "time" or "frequency"!')

    elif set(detector) == {'H1', 'L1'}:
        # Store the detector signals we will get through projection

        # Set up detectors
        detectors = {'H1': Detector('H1'), 'L1': Detector('L1')}

        # Loop over both detectors and calculate the signal we would see there
        for detector_name in ('H1', 'L1'):

            # Set up the detector based on its name
            det = detectors[detector_name]

            # Calculate the antenna pattern for this detector
            f_plus, f_cross = \
                det.antenna_pattern(right_ascension=right_ascension,
                                        declination=declination,
                                        polarization=polarization,
                                        t_gps=1376089759.8)

            # Calculate the time offset from H1 for this detector
            delta_t_h1 = \
                det.time_delay_from_detector(other_detector=detectors['H1'],
                                                right_ascension=right_ascension,
                                                declination=declination,
                                                t_gps=1376089759.8)
            # Project the waveform onto the antenna pattern
            detector_signal = f_plus * h_plus + f_cross * h_cross

        # Map the signal from geocentric coordinates to the specific
        # reference frame of the detector. This depends on whether we have
        # simulated the waveform in the time or frequency domain:
        if static_arguments['domain'] == 'time':
            offset = 100 + delta_t_h1 + detector_signal.start_time
            detector_signal = detector_signal.cyclic_time_shift(offset)
            detector_signal.start_time = event_time - 100
        elif static_arguments['domain'] == 'frequency':
            offset = 100 + delta_t_h1
            detector_signal = detector_signal.cyclic_time_shift(offset)
            detector_signal.start_time = event_time - 100
            detector_signal = detector_signal.to_timeseries()
        else:
            raise ValueError('Invalid domain! Must be "time" or "frequency"!')

    detector_signals = {}
    if detector == ['H1']:
        detector_signals['H1'] = detector_signal
        return detector_signals
        
    elif detector == ['L1']:
        detector_signals['L1'] = detector_signal
#        print(detector_signals['L1'].numpy())
        return detector_signals
        
    elif set(detector) == {'H1', 'L1'}:
        for detector_name in ('H1', 'L1'):
            # Store the result
            detector_signals[detector_name] = detector_signal
            
        return detector_signals
    
