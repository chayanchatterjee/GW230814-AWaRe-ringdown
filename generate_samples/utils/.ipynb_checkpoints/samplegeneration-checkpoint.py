"""
Provide the :func:`generate_sample()` method, which is at the heart of
the sample generation process.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from __future__ import print_function

import numpy as np
import pycbc

from lal import LIGOTimeGPS
from pycbc.psd import interpolate
from pycbc.psd.analytical import aLIGOZeroDetHighPower
from pycbc.noise import noise_from_psd
from pycbc.filter import sigma
from pycbc.types import TimeSeries

from .hdffiles import get_strain_from_hdf_file
from .hdffiles_DeepClean import get_strain_from_gwf_file
from .hdffiles_original import get_strain_from_original_file
from .waveforms import get_detector_signals, get_waveform
from pycbc.psd import inverse_spectrum_truncation, interpolate
from scipy.fft import fft, ifft
import h5py
import random


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

# Define the list of glitches
glitch_types = ['blip', 'low_freq_burst', 'whistle', 'koi_fish', 'scattered_light', 'repeating_blips']

# Function to pick a random glitch
def get_random_glitch_type():
    return random.choice(glitch_types)

def get_random_ifo():
    return random.choice(['H1', 'L1'])


# Initialize variables to manage glitch reuse
current_random_id = None
current_glitch_strain = None
injection_counter = 0

choice = get_random_ifo()

def get_glitch_for_injection(glitch_strain_data, n_injections_per_glitch, strain, det, whitened_waveforms):
    """
    Dynamically selects and returns the appropriate glitch for the current injection.
    Reuses the same glitch for `n_injections_per_glitch` injections.
    """
    global current_random_id, current_glitch_strain, injection_counter

    # Select a new glitch every `n_injections_per_glitch` injections
    if injection_counter % n_injections_per_glitch == 0:
        current_random_id = random.randint(0, len(glitch_strain_data) - 1)
        current_glitch_strain = TimeSeries(glitch_strain_data[current_random_id], delta_t=strain[det].delta_t)

        # Ensure lengths match
        if len(strain[det]) != len(current_glitch_strain):
            current_glitch_strain = current_glitch_strain[:len(strain[det])]

        # Set the epoch of the glitch strain to match strain[det]
        current_glitch_strain._epoch = whitened_waveforms[det]._epoch

    # Increment the counter for each injection
    injection_counter += 1

    # Return the current glitch strain
    return current_glitch_strain, current_random_id


def generate_sample(static_arguments,
                    event_tuple,
                    add_glitches_noise,
                    add_glitches_injection,
                    detector,
                    waveform_params=None):
    """
    Generate a single sample (or example) by taking a piece of LIGO
    background noise (real or synthetic, depending on `event_tuple`),
    optionally injecting a simulated waveform (depending on
    `waveform_params`) and post-processing the result (whitening,
    band-passing).
    
    Args:
        static_arguments (dict): A dictionary containing global
            technical parameters for the sample generation, for example
            the target_sampling_rate of the output.
        event_tuple (tuple): A tuple `(event_time, file_path)`, which
            specifies the GPS time at which to make an injection and
            the path of the HDF file which contains said GPS time.
            If `file_path` is `None`, synthetic noise will be used
            instead and the `event_time` only serves as a seed for
            the corresponding (random) noise generator.
        waveform_params (dict): A dictionary containing the randomly
            sampled parameters that are passed as inputs to the
            waveform model (e.g., the masses, spins, position, ...).

    Returns:
        A tuple `(sample, injection_parameters)`, which contains the
        generated `sample` itself (a dict with keys `{'event_time',
        'h1_strain', 'l1_strain'}`), and the `injection_parameters`,
        which are either `None` (in case no injection was made), or a
        dict containing the `waveform_params` and some additional
        parameters (e.g., single detector SNRs).
    """

    # -------------------------------------------------------------------------
    # Define shortcuts for some elements of self.static_arguments
    # -------------------------------------------------------------------------

    # Read out frequency-related arguments
    original_sampling_rate = static_arguments['original_sampling_rate']
    target_sampling_rate = static_arguments['target_sampling_rate']
    f_lower = static_arguments['f_lower']
    delta_f = static_arguments['delta_f']
    fd_length = static_arguments['fd_length']

    # Get the width of the noise sample that we either select from the raw
    # HDF files, or generate synthetically
    noise_interval_width = static_arguments['noise_interval_width']

    # Get how many seconds before and after the event time to use
    seconds_before_event = static_arguments['seconds_before_event']
    seconds_after_event = static_arguments['seconds_after_event']

    # Get the event time and the dict containing the HDF file path
    event_time, hdf_file_paths = event_tuple

    # -------------------------------------------------------------------------
    # Get the background noise (either from data or synthetically)
    # -------------------------------------------------------------------------

    # If the event_time is None, we generate synthetic noise
    if hdf_file_paths is None:

        # Create an artificial PSD for the noise
        # TODO: Is this the best choice for this task?
        psd = aLIGOZeroDetHighPower(length=fd_length,
                                    delta_f=delta_f,
                                    low_freq_cutoff=f_lower)

#        filename_LIGO = '/home/cchatterjee/samplegen/utils/LIGO_O4_PSD.txt'
#        psd_LIGO = pycbc.psd.from_txt(filename_LIGO, fd_length, delta_f,
#                         f_lower, is_asd_file=False)

#        filename_Virgo = '/home/cchatterjee/samplegen/utils/Virgo_O4_PSD.txt'
#        psd_Virgo = pycbc.psd.from_txt(filename_Virgo, fd_length, delta_f,
#                         f_lower, is_asd_file=False)

        # Actually generate the noise using the PSD and LALSimulation
        noise = dict()
#        for i, det in enumerate(('H1', 'L1')):
#        for i, det in enumerate(detector):
        for det in detector:
            # Compute the length of the noise sample in time steps
            noise_length = noise_interval_width * target_sampling_rate
#            print(det)
            i = 0
            if((det == 'H1') or (det == 'L1')):
            # Generate the noise for this detector
                noise[det] = noise_from_psd(length=noise_length,
                                            delta_t=(1.0 / target_sampling_rate),
                                            psd=psd,
                                            seed=(2 * event_time + i))
         #   elif(det == 'V1'):
         #   # Generate the noise for this detector
         #       noise[det] = noise_from_psd(length=noise_length,
         #                                   delta_t=(1.0 / target_sampling_rate),
         #                                   psd=psd,
         #                                   seed=(2 * event_time + i))

            # Manually fix the noise start time to match the fake event time.
            # However, for some reason, the correct setter method seems broken?
            start_time = event_time - noise_interval_width / 2
            # noinspection PyProtectedMember
            noise[det]._epoch = LIGOTimeGPS(start_time)
        
#        print(noise['L1'].numpy())

    elif hdf_file_paths == 'Deep_Clean_data' or hdf_file_paths == 'Deep_Clean_data_test':

        kwargs = dict(gwf_file_paths=hdf_file_paths,
                      gps_time=event_time,
                      interval_width=noise_interval_width,
                      original_sampling_rate=original_sampling_rate,
                      target_sampling_rate=target_sampling_rate,
                      as_pycbc_timeseries=True)

        
        noise = get_strain_from_gwf_file(**kwargs)

    elif hdf_file_paths == 'Original_data' or hdf_file_paths == 'Original_data_test':

        kwargs = dict(hdf_file_paths=hdf_file_paths,
                      gps_time=event_time,
                      interval_width=noise_interval_width,
                      original_sampling_rate=original_sampling_rate,
                      target_sampling_rate=target_sampling_rate,
                      as_pycbc_timeseries=True)

        
        noise = get_strain_from_original_file(**kwargs)


    
    # Otherwise we select the noise from the corresponding HDF file
    else:

        kwargs = dict(hdf_file_paths=hdf_file_paths,
                      gps_time=event_time,
                      interval_width=noise_interval_width,
                      original_sampling_rate=original_sampling_rate,
                      target_sampling_rate=target_sampling_rate,
                      as_pycbc_timeseries=True)

        
        noise = get_strain_from_hdf_file(**kwargs)

    # -------------------------------------------------------------------------
    # If applicable, make an injection
    # -------------------------------------------------------------------------

    # If no waveform parameters are given, we are not making an injection.
    # In this case, there are no detector signals and no injection
    # parameters, and the strain is simply equal to the noise
    if waveform_params is None:
        detector_signals = {}
        zero_signals = {}
        injection_parameters = {}
        strain = noise
        psds_noise = {}
        length = (seconds_before_event + seconds_after_event)*target_sampling_rate
        
#        for det in ('H1', 'L1'):
        for det in detector:
            # Estimate the Power Spectral Density from the dummy strain
 #           psds_noise[det] = noise[det].psd(4)
 #           psds_noise[det] = interpolate(psds_noise[det], delta_f=delta_f)
                        
            # Save whitened noise and a vector of zeros as the pure signals.
 #           f_lower = 30 
 #           idx = int(psds_noise[det].duration * f_lower)
 #           psds_noise[det][:idx] = psds_noise[det][idx]
 #           psds_noise[det][-1:] = psds_noise[det][-2]
            detector_signals[det] = np.zeros(int(length))
            detector_signals[det] = TimeSeries(detector_signals[det], delta_t = 1.0/target_sampling_rate)
            
        
        if set(detector) == {'H1', 'L1'}:
            injection_parameters = {'h1_signal': detector_signals['H1'],
                                    'l1_signal': detector_signals['L1']}
        #                           'v1_signal': detector_signals['V1'],
        #                            'h1_signal_whitened': detector_signals['H1'],
        #                            'l1_signal_whitened': detector_signals['L1']}
        #                           'v1_signal_whitened': detector_signals['V1']}
        #                           'psd_noise_h1':psds_noise['H1'],
        #                           'psd_noise_l1':psds_noise['L1'],
        #                           'psd_noise_v1':psds_noise['V1'],}
        
        elif detector == ['H1']:
            injection_parameters = {'h1_signal': detector_signals['H1']}
#                                    'h1_signal_whitened': detector_signals['H1']}
            
        elif detector == ['L1']:
            injection_parameters = {'l1_signal': detector_signals['L1']}
#                                    'l1_signal_whitened': detector_signals['L1']}

    # Otherwise, we need to simulate a waveform for the given waveform_params
    # and add it into the noise to create the strain
    else:

        # ---------------------------------------------------------------------
        # Simulate the waveform with the given injection parameters
        # ---------------------------------------------------------------------

        # Actually simulate the waveform with these parameters
        waveform = get_waveform(static_arguments=static_arguments,
                                waveform_params=waveform_params)

        # Get the detector signals by projecting on the antenna patterns
        detector_signals = \
            get_detector_signals(static_arguments=static_arguments,
                                 waveform_params=waveform_params,
                                 event_time=event_time,
                                 detector=detector,
                                 waveform=waveform)
            
#        print(detector_signals['L1'].numpy())

        # ---------------------------------------------------------------------
        # Add the waveform into the noise as is to calculate the NOMF-SNR
        # ---------------------------------------------------------------------

        # Store the dummy strain, the PSDs and the SNRs for the two detectors
        strain_ = {}
        psds_noise = {}
        psds = {}
        snrs = {}
        whitened_waveforms = {}

        # Calculate these quantities for both detectors
#        for det in ('H1', 'L1'):
        for det in detector:
            # Estimate the Power Spectral Density from the dummy strain
            psds_noise[det] = noise[det].psd(4)
            psds_noise[det] = interpolate(psds_noise[det], delta_f=delta_f)
                        
            # Add the simulated waveform into the noise to get the dummy strain
            strain_[det] = noise[det].add_into(detector_signals[det])
            
            # Save whitened waveforms
            f_lower = 20
            idx = int(psds_noise[det].duration * f_lower)
            psds_noise[det][:idx] = psds_noise[det][idx]
            psds_noise[det][-1:] = psds_noise[det][-2]
            
            
#            print(detector_signals[det])
            # Use the PSD estimate to calculate the optimal matched
            # filtering SNR for this injection and this detector
            snrs[det] = sigma(htilde=detector_signals[det],
                              psd=psds_noise[det],
                              low_frequency_cutoff=f_lower)
            
#            h_fft = fft(whitened_waveforms[det].numpy())
#            snr_freq = (h_fft * h_fft.conjugate())
#            snr_time = np.abs(ifft(snr_freq))
#            snrs[det] = np.sqrt(np.max(snr_time)) 

#            h_fft = fft(whitened_waveforms[det].numpy())
#            snr_freq = np.inner(h_fft,h_fft.conjugate())
#            snrs[det] = np.sqrt(np.abs(snr_freq)) 


        # Calculate the network optimal matched filtering SNR for this
        # injection (which we need for scaling to the chosen injection SNR)
#        nomf_snr = np.sqrt(snrs['H1']**2 + snrs['L1']**2) 

            # Calculate nomf_snr based on the detectors present
        if detector == ['H1']:
            nomf_snr = np.sqrt(np.abs(snrs['H1'])**2)
        elif detector == ['L1']:
            nomf_snr = np.sqrt(np.abs(snrs['L1'])**2)
        elif set(detector) == {'H1', 'L1'}:
            nomf_snr = np.sqrt(snrs['H1']**2 + snrs['L1']**2)
        else:
            raise ValueError("Invalid detector combination")

#        print(nomf_snr)
        
        # ---------------------------------------------------------------------
        # Add the waveform into the noise with the chosen injection SNR
        # ---------------------------------------------------------------------

        # Compute the rescaling factor
        injection_snr = waveform_params['injection_snr']
        scale_factor = 1.0 * injection_snr / nomf_snr        
        
        strain = {}
#        for det in ('H1', 'L1'):
        for det in detector:
                        
#            f_lower = 20
#            psds_noise[det] = interpolate(psds_noise[det], noise[det].delta_f)
#            idx = int(psds_noise[det].duration * f_lower)
#            psds_noise[det][:idx] = psds_noise[det][idx]
#            psds_noise[det][-1:] = psds_noise[det][-2]
            # Whiten the noise
#            noise[det] = (noise[det].to_frequencyseries() / psds_noise[det]**0.5).to_timeseries()           
            
            # Add the simulated waveform into the noise, using a scaling
            # factor to ensure that the resulting NOMF-SNR equals the chosen
            # injection SNR

#            strain[det] = noise[det].add_into(scale_factor *
#                                              detector_signals[det])          # Change for SNR Variable
            
            strain[det] = noise[det].add_into(detector_signals[det])          # Change for SNR Variable

            
            whitened_waveforms[det] = ((scale_factor * detector_signals[det]).to_frequencyseries() / psds_noise[det]**0.5).to_timeseries()  

#            whitened_waveforms[det] = ((detector_signals[det]).to_frequencyseries() / psds_noise[det]**0.5).to_timeseries()
            
#        print(strain['L1'].numpy())
        # ---------------------------------------------------------------------
        # Store some information about the injection we just made
        # ---------------------------------------------------------------------

        # Store the information we have computed ourselves
        if detector == ['H1']:
            injection_parameters = {'scale_factor': scale_factor,
                                'h1_snr': snrs['H1']}
    
        elif detector == ['L1']:
            injection_parameters = {'scale_factor': scale_factor,
                                'l1_snr': snrs['L1']}
    
        elif set(detector) == {'H1', 'L1'}:
            injection_parameters = {'scale_factor': scale_factor,
                                'h1_snr': snrs['H1'],
                                'l1_snr': snrs['L1']}
    

        # Also add the waveform parameters we have sampled
        for key, value in waveform_params.items():
            injection_parameters[key] = value

    # -------------------------------------------------------------------------
    # Whiten and bandpass the strain (also for noise-only samples)
    # -------------------------------------------------------------------------
    
#    for det in ('H1', 'L1'):
    for det in detector:

        # Get the whitening parameters
        segment_duration = static_arguments['whitening_segment_duration']
        max_filter_duration = static_arguments['whitening_max_filter_duration']

        # Whiten the strain (using the built-in whitening of PyCBC)
        # We don't need to remove the corrupted samples here, because we
        # crop the strain later on
        strain[det] = \
            strain[det].whiten(segment_duration=segment_duration,
                               max_filter_duration=max_filter_duration,
                               remove_corrupted=False)
            
# Standardize the strain, if necessary. 

    #    std_dev = np.std(strain[det].numpy())
    #    strain[det] = strain[det]/std_dev   

        # Get the limits for the bandpass
        bandpass_lower = static_arguments['bandpass_lower']
        bandpass_upper = static_arguments['bandpass_upper']

        # Apply a high-pass to remove everything below `bandpass_lower`;
        # If bandpass_lower = 0, do not apply any high-pass filter.
        if bandpass_lower != 0:
            strain[det] = strain[det].highpass_fir(frequency=bandpass_lower,
                                                   remove_corrupted=False,
                                                   order=512)

        # Apply a low-pass filter to remove everything above `bandpass_upper`.
        # If bandpass_upper = sampling rate, do not apply any low-pass filter.
        if bandpass_upper != target_sampling_rate:
            strain[det] = strain[det].lowpass_fir(frequency=bandpass_upper,
                                                  remove_corrupted=False,
                                                  order=512)
        
    # -------------------------------------------------------------------------
    # Cut strain (and signal) time series to the pre-specified length
    # -------------------------------------------------------------------------

#    print(strain['L1'].numpy().max())
    
    strain_with_glitch = {}
#    for det in ('H1', 'L1'):
    for det in detector:
        
        if waveform_params is not None:
            t_shift = waveform_params['t_shift']

            # Define some shortcuts for slicing
            a = event_time - seconds_before_event + t_shift
            b = event_time + seconds_after_event + t_shift
            
        else:
            
            # Define some shortcuts for slicing
            a = event_time - seconds_before_event
            b = event_time + seconds_after_event
            
        # Cut the strain to the desired length
        strain[det] = strain[det].time_slice(a, b)

        total_length = seconds_before_event + seconds_after_event
        end_index = int(total_length*target_sampling_rate)
        strain[det] = strain[det][0:end_index]
        
#        print(strain[det].numpy())

        # If we've made an injection, also cut the simulated signal
        if waveform_params is not None:

            # Cut the detector signals to the specified length
#            detector_signals[det] = detector_signals[det].time_slice(a, b)
            random_glitch_id = {}
            whitened_waveforms[det] = whitened_waveforms[det].time_slice(a, b)
                        
            # Main conditional block
#            if add_glitches_injection is not None:

#                if add_glitches_injection == 'random':
#                    add_glitches_injection = get_random_glitch_type()
                    
##                glitch_file_path = '/workspace/LIGO/samplegen/Glitch_data/combined_strains_snr_'+ add_glitches_injection +'_whitened_1.hdf5'
#                glitch_file_path = '/workspace/LIGO/samplegen/Glitch_data/combined_strains_snr_' + add_glitches_injection + '_whitened_1_snr-8to20.hdf5'

#                try:
#                    with h5py.File(glitch_file_path, 'r') as f1:
#                        glitch_strain_data = f1['Strain'][()]

#                        # Dynamically call the helper function to get the glitch
#                        glitch_strain, random_id = get_glitch_for_injection(glitch_strain_data, n_injections_per_glitch, strain, det, whitened_waveforms)

#                        random_glitch_id[det] = random_id
#                        # Perform addition
#                        strain[det] = glitch_strain + whitened_waveforms[det]

#                except Exception as e:
#                    print(f"Error occurred: {e}")

            
            
            if add_glitches_injection is not None:
                
                if add_glitches_injection == 'random':
                    
                    add_glitches_injection = get_random_glitch_type()
                    glitch_file_path = '/home/chattec/AWaRe_train/notebooks/Glitch_data/combined_strains_snr_'+ add_glitches_injection +'_whitened_1.hdf5'
                
                elif add_glitches_injection == 'scattered_light':
                
                    glitch_file_path = '/home/chattec/AWaRe_train/notebooks/Glitch_data/gspy_'+ add_glitches_injection +'_processed.hdf5'
                    
                else:
                    glitch_file_path = '/home/chattec/AWaRe_train/notebooks/Glitch_data/combined_strains_snr_'+ add_glitches_injection +'_whitened_1.hdf5'
                
                try:
                    with h5py.File(glitch_file_path, 'r') as f1:
                        glitch_strain_data = f1['Strain'][()]
                        random_id = random.randint(0, len(glitch_strain_data)-1)
                        glitch_strain = TimeSeries(glitch_strain_data[random_id], delta_t=strain[det].delta_t)
                
                        # Ensure lengths match
                        if len(strain[det]) != len(glitch_strain):
                            glitch_strain = glitch_strain[:len(strain[det])]
                    
                        # Set the epoch of glitch_strain to match strain[det]
                        glitch_strain._epoch = whitened_waveforms[det]._epoch
                
#                        # Perform addition
#                        if set(detector) == {'H1', 'L1'}:
#                            if choice == det:
#                                strain[det] = glitch_strain + whitened_waveforms[det]
                            
#                        else:
                        strain[det] = glitch_strain + whitened_waveforms[det]

                except Exception as e:
                    print(f"Error occurred: {e}")

            if set(detector) == {'H1', 'L1'}:
                # Also add the detector signals to the injection parameters
        #        injection_parameters['h1_signal'] = \
        #            np.array(detector_signals['H1'])
        #        injection_parameters['l1_signal'] = \
        #            np.array(detector_signals['L1'])
        #       injection_parameters['v1_signal'] = \
        #           np.array(detector_signals['V1'])
                injection_parameters['h1_signal_whitened'] = \
                    np.array(whitened_waveforms['H1'])
                injection_parameters['l1_signal_whitened'] = \
                    np.array(whitened_waveforms['L1'])
        #        injection_parameters['v1_signal_whitened'] = \
        #            np.array(whitened_waveforms['V1'])
    
                injection_parameters['psd_noise_h1'] = \
                    np.array(psds_noise['H1'])
                injection_parameters['psd_noise_l1'] = \
                    np.array(psds_noise['L1'])
        #        injection_parameters['psd_noise_v1'] = \
        #            np.array(psds_noise['V1'])

        #        injection_parameters['random_glitch_id_h1'] = \
        #            np.array(random_glitch_id['H1'])
        #        injection_parameters['random_glitch_id_l1'] = \
        #            np.array(random_glitch_id['L1'])
                
        
            elif detector == ['H1']:
        #        injection_parameters['h1_signal'] = \
        #            np.array(detector_signals['H1'])
                injection_parameters['h1_signal_whitened'] = \
                    np.array(whitened_waveforms['H1'])    
                injection_parameters['psd_noise_h1'] = \
                    np.array(psds_noise['H1'])
        #        injection_parameters['random_glitch_id_h1'] = \
        #            np.array(random_glitch_id['H1'])
                    
            elif detector == ['L1']:
        #        injection_parameters['l1_signal'] = \
        #            np.array(detector_signals['L1'])
                injection_parameters['l1_signal_whitened'] = \
                    np.array(whitened_waveforms['L1'])    
        #        injection_parameters['psd_noise_l1'] = \
        #            np.array(psds_noise['L1'])
        #        injection_parameters['random_glitch_id_l1'] = \
        #            np.array(random_glitch_id['L1'])
                    
#                print(strain['L1'].numpy())

            
        elif waveform_params is None:
            
            if add_glitches_noise is not None:
                
                if add_glitches_noise == 'random':
                    
                    add_glitches_noise = get_random_glitch_type()
                    glitch_file_path = '/home/chattec/AWaRe_train/notebooks/Glitch_data/combined_strains_snr_'+ add_glitches_noise +'_whitened_1.hdf5'
#                glitch_file_path = '/workspace/ligo_data/Glitch_classification/Strain_data/gspy_blip_H1_processed.hdf5'
                
                elif add_glitches_noise == 'scattered_light':
                
                    glitch_file_path = '/home/chattec/AWaRe_train/notebooks/Glitch_data/gspy_'+ add_glitches_noise +'_processed.hdf5'


                else:
                    glitch_file_path = '/home/chattec/AWaRe_train/notebooks/Glitch_data/combined_strains_snr_'+ add_glitches_noise +'_whitened_1.hdf5'


                try:
                    with h5py.File(glitch_file_path, 'r') as f1:
                        glitch_strain_data = f1['Strain'][()]
                        random_id = random.randint(0, len(glitch_strain_data)-1)
                        glitch_strain = TimeSeries(glitch_strain_data[random_id], delta_t=strain[det].delta_t)
                
                        # Ensure lengths match
                        if len(strain[det]) != len(glitch_strain):
                            glitch_strain = glitch_strain[:len(strain[det])]
                    
                        # Set the epoch of glitch_strain to match strain[det]
                        glitch_strain._epoch = strain[det]._epoch
                
#                        # Perform addition
#                        if set(detector) == {'H1', 'L1'}:
#                            if choice == det:
#                                strain[det] = glitch_strain
                                
#                        else:
#                            strain[det] = glitch_strain 
                            
                        strain[det] = glitch_strain
                        
                except Exception as e:
                    print(f"Error occurred: {e}")
            

            if set(detector) == {'H1', 'L1'}:
            #    injection_parameters['h1_signal'] = \
            #        np.array(detector_signals['H1'])
            #    injection_parameters['l1_signal'] = \
            #        np.array(detector_signals['L1'])
            #    injection_parameters['v1_signal'] = \
            #        np.array(detector_signals['V1'])

                injection_parameters['h1_signal_whitened'] = \
                    np.array(detector_signals['H1'])
                injection_parameters['l1_signal_whitened'] = \
                    np.array(detector_signals['L1'])
                    
            elif detector == ['H1']:
            #    injection_parameters['h1_signal'] = \
            #        np.array(detector_signals['H1'])

                injection_parameters['h1_signal_whitened'] = \
                    np.array(detector_signals['H1'])
            
            elif detector == ['L1']:
            #    injection_parameters['l1_signal'] = \
            #        np.array(detector_signals['L1'])

                injection_parameters['l1_signal_whitened'] = \
                    np.array(detector_signals['L1'])
                    
        
        #    injection_parameters['v1_signal_whitened'] = \
        #        np.array(detector_signals['V1'])
        
        ##    injection_parameters['psd_noise_h1'] = \
        ##        np.array(psds_noise['H1'])
        ##    injection_parameters['psd_noise_l1'] = \
        ##        np.array(psds_noise['L1'])
        ##    injection_parameters['psd_noise_v1'] = \
        ##        np.array(psds_noise['V1'])
            
    # -------------------------------------------------------------------------
    # Collect all available information about this sample and return results
    # -------------------------------------------------------------------------

    # The whitened strain is numerically on the order of O(1), so we can save
    # it as a 32-bit float (unlike the original signal, which is down to
    # O(10^-{30}) and thus requires 64-bit floats).
    
    if set(detector) == {'H1', 'L1'}:
        sample = {'event_time': event_time,
                'h1_strain': np.array(strain['H1']).astype(np.float32),
                'l1_strain': np.array(strain['L1']).astype(np.float32)}
    
#              'h1_strain_with_glitch': np.array(strain_with_glitch['H1']).astype(np.float32),
#              'l1_strain_with_glitch': np.array(strain_with_glitch['L1']).astype(np.float32)}
#             'v1_strain': np.array(strain['V1']).astype(np.float32)}
        
        return sample, injection_parameters

    elif detector == ['L1']:
#        print(strain['L1'].numpy())
        sample = {'event_time': event_time,
                'l1_strain': np.array(strain['L1']).astype(np.float32)}

        return sample, injection_parameters
        
    elif detector == ['H1']:
        sample = {'event_time': event_time,
                'h1_strain': np.array(strain['H1']).astype(np.float32)}
        
        return sample, injection_parameters
