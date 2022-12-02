# function for data preprocessing

import numpy as np
from cmath import exp
from os import remove
import os
import mne
import scipy.io as sio
import copy

def mat_to_rawfif(load_path,
               ch_types=['eeg'],
               montage_path=None,
               has_trigger=False,
               sampling_rate=None,
               save_path=None):
    """
    prepare .mat file into raw instance to be saved 

    Parameters
    ----------
    load_path : string
        folder of .mat file

    ch_types : string or list of strings
        types of channels that can be 'eeg', 'all', 'csd', 'stim'
        you can have 'stim' type in the ch_types list.

    montage_path : string
        .set file from EEGLAB that contain channel location

    has_trigger : bool
        input "True" if trigger event was recorded in the last channel

    sampling_rate : int
        sampling_rate of the data

    save_path : string
        path of folder to save the data
        if save_path is "None", the function will return raw instance

    Returns
    -------
    Raw Instance  
        prepared raw data
    """

    # get eeg montage from eeglab_file 
    if montage_path!=None:
        eeglab_example = mne.io.read_raw_eeglab(montage_path)
        montage = eeglab_example.get_montage()
    else:
        raise Exception("montage has not been determined.")

    raw_array = sio.loadmat(load_path)['raw_data']
    ch_names = montage.ch_names

    if (has_trigger):
        ch_names = ch_names + ['trigger']

    # create infomation of the data (which contains information of ch_names, sampling_rate, and ch_types)
    info = mne.create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types=ch_types)
    raw = mne.io.RawArray(raw_array, info=info)
    raw = raw.load_data()

    # set montage from eeglab to this file
    raw = raw.set_montage(montage)

    # save file
    if save_path!=None:
        mne.io.Raw.save(raw, save_path, overwrite=True)
        return None
    else:
        return raw

"""---------------------------------------------------------------------------------------------------"""

def preprocess_raw(load_path,
                   ch_types='all',
                   notch_freq=50,
                   bandpass_freq=[1, 40],
                   average_ref=True,
                   resampling_freq=None,
                   ica_param=[61, 'fastica'],
                   remove_eog=False,
                   remove_bad_channels=False,
                   csd=False,
                   save_path=None):
    
    """
    preprocess data with the following procedure:
    1. notch filter 
    2. bandpass filter
    3. resample data
    4. ica
    5. remove EOG (if needed)
    6. average re-reference
    7. remove bad channels

    Parameters
    ----------
    load_path : string
        folder of raw data

    ch_types : string or list of strings
        types of channels that can be 'eeg', 'all', 'csd', 'stim'

    notch_freq : int
        frequency of notch filter

    bandpass_freq : tuple
        range of bandpass freqency (min, max)

    average_ref : bool
        apply average reference if True

    resampling_freq : int 
        target frequency to resample

    ica_param : list
        list of two ica parameters --> [n_channels, algorithm]

    remove_eog : bool
        apply removal of eog

    remove_bad_channels : bool
        remove bad channels using pyprep
        more detail on handle_bad_channels

    save_path : string
        folder path to save the preprocessed data if save_path is None, 
        the preprocessed data will be returned variable

    Returns
    -------
    None 
        return nothing if save_path is determined
    """

    raw = mne.io.read_raw_fif(load_path, preload=True)
    
    if (ch_types != 'all'):
        raw = raw.pick_channels(ch_types)

    # notch filter with 50 Hz
    filtered = raw.copy().notch_filter(notch_freq, 
                                       picks=['eeg'])

    # bandpass filter with 1-40 Hz
    filtered = filtered.filter(bandpass_freq[0], 
                               bandpass_freq[1], 
                               picks=['eeg'])

    # resampling data
    if resampling_freq != None:
        filtered = filtered.resample(sfreq=resampling_freq)

    # ica
    ica = mne.preprocessing.ICA(ica_param[0], 
                                method=ica_param[1]) 
    ica.fit(filtered, picks='eeg')

    # find and remove the bad components based on fp1 and fp2 channels
    if remove_eog:
        eog_idx_fp1, scores_fp1 = ica.find_bads_eog(filtered, ch_name='fp1')
        eog_idx_fp2, scores_fp2 = ica.find_bads_eog(filtered, ch_name='fp2')
        excluded_ch = [eog_idx_fp1[eog_idx_fp1 == eog_idx_fp2]]
        processed = ica.apply(filtered, exclude=excluded_ch)
    else:
        processed = filtered

    # rereference using common avearage
    if average_ref:
        processed = processed.copy().set_eeg_reference(ref_channels='average', ch_type='eeg')

    # remove bad channels
    if remove_bad_channels:
        bad_chs, processed = handle_bad_channels(processed)
        print ("Bad channels interpolated: " + str(bad_chs))

    if csd:
        processed = mne.preprocessing.compute_current_source_density(processed)

    # save file
    if save_path!=None:
        mne.io.Raw.save(processed, save_path, overwrite=True)
        return None
    else:
        return processed

"""---------------------------------------------------------------------------------------------------"""

def raw_to_epoch(raw_path,
                filetype='.fif',
                picks=['eeg'],
                event_ids=None,
                time_range=None,
                baseline=None):

    # get event_info based on filetype
    if filetype=='.fif':
        data = mne.io.read_raw_fif(raw_path, preload=True)
        event_info = mne.find_events(data, stim_channel='trigger')
    elif filetype=='.set':
        data = mne.io.read_raw_eeglab(raw_path, preload=True)
        event_info = mne.events_from_annotations(data)[0]

    # extract epochs
    data_event = []
    for id in event_ids:
        epoch = mne.Epochs(data, 
                        event_info, 
                        picks=picks, 
                        event_id=[id],
                        tmin=time_range[0], 
                        tmax=time_range[1], 
                        preload=True, 
                        baseline=baseline)
        data_event.append(epoch)

    return data_event


# def prepare_epochs(load_path, 
#                 filetype='.fif', 
#                 picks=['eeg'], 
#                 n_subjects=0, 
#                 n_sessions=1, 
#                 event_ids=None, 
#                 time_range=None, 
#                 baseline=None):
#     """
#     import data and convert into lists of multiple epohcs [subjects/sessions/conditions]
#     filename should follow this format --> "subjectXXX_sessionYYY.zzz"
#     XXX = three digits of subjects' number
#     YYY = three digits of sessions
#     zzz = filetype

#     Parameters
#     ----------
#     load_path : string
#         folder of preprocessed data

#     filetype : string
#         '.fif' or '.set'

#     picks : list of string
#         channel types of interest can be ['eeg', 'csd']

#     n_subjects : int
#         number of subjects

#     n_sessions : int 
#         nubmer of sessions

#     event_ids : list
#         list of event ids of interest

#     time_range : tuple of length 2
#         time period of interest in second --> ex. (-2, 5)

#     baseline : tuple of length 2
#         baseline period if "None" then no baseline will be applied
 
#     Returns
#     -------
#     list of Epochs
#         lists of Epoch data which have 3 dimensions, inclulding [n_subjects, n_sessions, n_events]
#         if n_sessions = 1: the returned list includes [n_subjects, n_events]
#     """

#     if n_subjects == 0:
#         raise Exception("n_subjects should not be zero or None.")

#     epochs = []

#     # generate a filename according to specific name format
#     # filenames were recommended to be "subjectXXX_sessionYYY.zzz"
#     # XXX = three digits of subjects' number
#     # YYY = three digits of sessions
#     # zzz = filetype
#     for n_subj in range(n_subjects):
#         data_session = []
#         for n_sess in range(n_sessions):
#             # check whether your files have multiple sessions
#             if n_sessions>1:          
#                 filename = f"subject{n_subj:03}_session{n_sess+1:03}{filetype}"             
#             else:     
#                 filename = f"subject{n_subj:03}{filetype}"  
            
#             # check if the filename exist in the folder
#             if filename in os.listdir(load_path):
#                 data = mne.io.read_raw_fif(load_path + filename, preload=True)
#             else:
#                 print (f"{filename} is not exist")

#             # get event_info based on filetype
#             if filetype=='.fif':
#                 event_info = mne.find_events(data, stim_channel='trigger')
#             elif filetype=='.set':
#                 event_info = mne.events_from_annotations(data)[0]

#             # extract epochs
#             data_event = []
#             for id in event_ids:
#                 epoch = mne.Epochs(data, 
#                                 event_info, 
#                                 picks=picks, 
#                                 event_id=[id],
#                                 tmin=time_range[0], 
#                                 tmax=time_range[1], 
#                                 preload=True, 
#                                 baseline=baseline)
#                 data_event.append(epoch)

#             data_session.append(data_event)

#     # ready-to-use epochs       
#     epochs.append(data_session)

#     return epochs

"""---------------------------------------------------------------------------------------------------"""

def epoch_to_tfr(epoch,
                 frequency_range=[5, 40]):

    """
    Convert epochs to tfrs (time-frequency representation) 

    Parameters
    ----------
    epochs : string
        folder of preprocessed data

    filetype : string
        '.fif' or '.set'

    picks : list of string
        channel types of interest can be ['eeg', 'csd']

    n_subjects : int
        number of subjects

    n_sessions : int 
        nubmer of sessions

    event_ids : list
        list of event ids of interest

    time_range : tuple of length 2
        time period of interest in second --> ex. (-2, 5)

    baseline : tuple of length 2
        baseline period if "None" then no baseline will be applied
 
    Returns
    -------
    list of Epochs
        lists of Epoch data which have 3 dimensions, inclulding [n_subjects, n_sessions, n_events]
        if n_sessions = 1: the returned list includes [n_subjects, n_events]
    """

    freqs = np.logspace(*np.log10([5, 40]), num=80)
    freqs = np.arange(5, 40, 1)
    n_cycles = freqs / 2.  # different number of cycle per frequency

    # tfr is returned as AverageTFR class
    tfr = mne.time_frequency.tfr_morlet(epoch, 
                                    freqs=freqs, 
                                    n_cycles=n_cycles, 
                                    use_fft=True,
                                    return_itc=False, 
                                    picks=['csd', 'eeg'], 
                                    decim=3, 
                                    n_jobs=1, 
                                    zero_mean=True)

    return tfr

"""---------------------------------------------------------------------------------------------------"""

def epoch_to_conn(epoch,
                fmin,
                fmax,
                tmin,
                tmax,
                method='pli',
                return_binary=True):

    data = epoch.copy().pick_types(csd=True)
    sfreq = data.info['sfreq']
    conn = spectral_connectivity_epochs(
                data.load_data(),
                method=method,
                mode='multitaper',
                sfreq=sfreq,
                fmin=fmin,
                fmax=fmax,
                faverage=True,
                tmin=tmin,
                mt_adaptive=False,
                n_jobs=1)

    if return_binary:
        # thesholding by 5% of maximum connectivities (especially for plotting network)
        # comment it when the absolute values were needed
        sorted_cont = np.sort(np.squeeze(con.get_data()))
        thresh_idx = sorted_cont[int(-1*0.05*62*61/2)]
        
        # get binary connectivity matrix
        return (con.get_data(output='dense')[:, :, 0] > thresh_idx)*1

    # return absolute connectivity matrix
    return conn.get_data(output='dense')[:, :, 0]

"""---------------------------------------------------------------------------------------------------"""

def handle_bad_channels(data):
    # temporarily excluding trigger channel for bad_channels_handling
    temp = data.copy().drop_channels('trigger')
    original_bads = copy.deepcopy(temp.info['bads'])

    # get bad channels using pyprep
    prep = pyprep.NoisyChannels(temp)
    prep.find_all_bads()
    bads = prep.get_bads()

    # interpolate bads channels after removing it
    data.info['bads'].extend(bads)
    new_data = data.copy().interpolate_bads(reset_bads=False)
    new_data.info['bads'] = original_bads
    return bads, new_data





