import numpy as np
from scipy.signal import iirnotch, filtfilt, butter, sosfilt, find_peaks
import matplotlib.pyplot as plt
import pywt
import pandas as pd
import padasip as pa

"""
class that is used to process ECG signals

functions:

    read_data: reads in the ECG signals from a CSV file in the format [Mixed ECG, Caregiver ECG]

    preprocess_signals: applies notch -> band-pass filters to the ECG recordings

    ** these are opposite of each other but are used in sequence to get our frequency band of interest
    _apply_notch_filter: blocks a range of frequencies, allows other from outside the frequency to pass
    _apply_bandpass_filter: allows frequencies from a specific range, blocks those outside of that frequency

    adaptive_filter_setup: init the adaptive filter for real-time filtering

    process_signals: used to find the estimated caregiver ECG and find the QRS complexes of the child ECG

    _segment_signal: segments the ECGs into different windows

    detect_qrs_complexs: finds the QRS complex of the mixed ECG (aka find the child ECG peaks)

    plot_results: plots the signals
"""
class ECGProcessor:
    """
    A class to process ECG signals, including reading data, preprocessing,
    adaptive filtering, and plotting results.
    """
    def __init__(self, fs=250):
        
        self.fs = fs
        self.filter_params = {
            'notch_freq': 50,      # frequency of notch filter
            'notch_q': 30,         # quality factor for notch filter
            'bandpass_low': 5,     # low cutoff frequency for band-pass filter
            'bandpass_high': 30    # high cutoff frequency for band-pass filter
        }
        
    def read_data(self, file_path): 
        try:
            signals = np.loadtxt(file_path + '.csv', delimiter=",")

            # pull mixed and caregiver ECG signals
            mixed_ecg = signals[:, 0].reshape(-1, 1)
            caregiver_ecg = signals[:, 1].reshape(-1, 1)

            return mixed_ecg, caregiver_ecg

        except Exception as e:
            print(f"Error reading file: {e}")
            return None, None
    
    def preprocess_signals(self, 
                           mixed_ecg, 
                           caregiver_ecg):
        
        # apply notch filter to remove powerline interference
        mixed_notch = self._apply_notch_filter(mixed_ecg.flatten())
        caregiver_notch = self._apply_notch_filter(caregiver_ecg.flatten())

        # apply band-pass filter to get relevant ECG frequencies
        mixed_filtered = self._apply_bandpass_filter(mixed_notch)
        caregiver_filtered = self._apply_bandpass_filter(caregiver_notch)

        return mixed_filtered, caregiver_filtered
    
    def _apply_notch_filter(self, signal_data):
        
        # init the notch filter coefficients
        b, a = iirnotch(2 * self.filter_params['notch_freq'] / self.fs, 
                        self.filter_params['notch_q'])
        
        # apply the notch filter
        return filtfilt(b, a, signal_data)
    
    def _apply_bandpass_filter(self, signal_data):
        
        # init band-pass filter using second-order sections
        sos = butter(4, 
                     [self.filter_params['bandpass_low'], 
                      self.filter_params['bandpass_high']], 
                     'bandpass', fs=self.fs, output='sos')
        
        # apply the filter
        return sosfilt(sos, signal_data)
    
    def adaptive_filter_setup(self, 
                              filter_size=5, 
                              mu=0.1):

        # uses the LMS algorithm
        return real_time_adaptfilter('LMS', filter_size, mu)
        
    def process_signals(self, 
                        mixed_ecg, 
                        caregiver_ecg, 
                        window_size=256):
        
        y_hat = []
        residual = []

        # determine filter size based on wavelet decomposition
        sample_mecg = caregiver_ecg[:window_size]
        sample_mecg = normalize_ecg(sample_mecg.flatten(), 0, self.fs * 5)
        mecg_stk = wavelet_transform(sample_mecg, 'db4', 4)
        filter_size = mecg_stk.shape[1]

        # init adaptive filter
        adapt_filter = self.adaptive_filter_setup(filter_size=filter_size, mu=0.001)

        # initial adaptation with a larger window
        init_time = window_size * 8
        mixed_init = mixed_ecg[:init_time]
        caregiver_init = caregiver_ecg[:init_time]
        y, e, w = wavelet_adapt_filtering(adapt_filter, mixed_init, caregiver_init, self.fs)
        y_hat.extend(y)
        residual.extend(e)

        # segment the rest of the signals
        mixed_wins, caregiver_wins = self._segment_signal(mixed_ecg[init_time:], caregiver_ecg[init_time:], window_size)

        # process each window
        for i in range(mixed_wins.shape[0]):
            y, e, w = wavelet_adapt_filtering(adapt_filter, mixed_wins[i], caregiver_wins[i], self.fs)
            y_hat.extend(y)
            residual.extend(e)

        # convert lists to arrays and align lengths
        y_hat = np.array(y_hat)
        residual = np.array(residual)
        min_length = min(len(mixed_ecg), len(y_hat))
        y_hat = y_hat[:min_length]
        residual = residual[:min_length]

        return y_hat, residual, None
    
    # segment the window for processing
    def _segment_signal(self, 
                        signal1, 
                        signal2, 
                        window_size):
        
        signal_len = len(signal1)
        num_windows = (signal_len) // window_size
        
        # segment the signals into two windows
        wins1 = np.array([signal1[i*window_size:(i+1)*window_size].flatten() for i in range(num_windows)])
        wins2 = np.array([signal2[i*window_size:(i+1)*window_size].flatten() for i in range(num_windows)])

        return wins1, wins2
    

    def detect_qrs_complexes(self, signal, threshold_ratio=0.6):
        
        # calc dynamic threshold based on signal amplitude
        threshold = threshold_ratio * np.max(np.abs(signal))

        # detect peaks with specified distance + height
        peaks, _ = find_peaks(signal, distance=self.fs//5, height=threshold)

        return peaks
    
    def plot_results(self, 
                     original_mixed, 
                     original_caregiver, 
                     estimated_maternal, 
                     residual, 
                     time_window=None, 
                     save_path=None):

        # flatten signals and apply time window if specified
        original_mixed = np.array(original_mixed).flatten()
        original_caregiver = np.array(original_caregiver).flatten()
        estimated_maternal = np.array(estimated_maternal).flatten()
        residual = np.array(residual).flatten()
        
        if time_window:
            start, end = time_window
            original_mixed = original_mixed[start:end]
            original_caregiver = original_caregiver[start:end]
            estimated_maternal = estimated_maternal[start:end]
            residual = residual[start:end]
        else:
            start = 0
            end = len(original_mixed)
        
        # align signals to the same length
        min_length = min(len(original_mixed), len(original_caregiver), len(estimated_maternal), len(residual))
        original_mixed = original_mixed[:min_length]
        original_caregiver = original_caregiver[:min_length]
        estimated_maternal = estimated_maternal[:min_length]
        residual = residual[:min_length]
        
        # create time axis
        time = np.arange(start, start + min_length) / self.fs
        
        # plot signals
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        axes[0].plot(time, original_mixed, 'b-', label='Mixed ECG')
        axes[0].set_title('Original Mixed ECG Signal')
        axes[0].legend()
        
        axes[1].plot(time, original_caregiver, 'r-', label='Caregiver ECG')
        axes[1].set_title('Original Caregiver ECG Signal')
        axes[1].legend()
        
        axes[2].plot(time, estimated_maternal, 'g-', label='Estimated Caregiver ECG')
        axes[2].set_title('Estimated Maternal ECG')
        axes[2].legend()
        
        axes[3].plot(time, residual, 'm-', label='Extracted Child ECG')
        qrs_peaks = self.detect_qrs_complexes(residual)
        axes[3].plot(time[qrs_peaks], residual[qrs_peaks], 'r+', label='QRS Complexes')
        axes[3].set_title('Child ECG Signal from Mixed ECG Marked with QRS Complexes')
        axes[3].legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

#############################################################################################
def wavelet_transform(sig, 
                      wavelet_type, 
                      level):
    
    # wavelet decomposition
    coeffs = pywt.wavedec(sig, wavelet_type, level=level)
    n = len(sig)
    wa_dic = {}
    for i in range(1, level + 1):
        # Reconstruct detail coefficients at each level
        coeffs_i = [np.zeros_like(c) for c in coeffs]
        coeffs_i[-i] = coeffs[-i]
        rec = pywt.waverec(coeffs_i, wavelet_type)[:n]
        wa_dic[f'd{level - i + 1}'] = rec
    
    # reconstruct approximation coeff's
    coeffs_a = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    rec_a = pywt.waverec(coeffs_a, wavelet_type)[:n]
    wa_dic[f'a{level}'] = rec_a
    
    # convert to numpy array
    return pd.DataFrame(wa_dic).to_numpy()

#############################################################################################

"""
Class the is used for real time adaptive filter
- you can use either LMS or NLMS algo's

functions:

    filtering: applies adaptive filtering to the ECG signals

"""

class real_time_adaptfilter:
    
    def __init__(self, model, filterSize, mu):
        if model.upper() == 'LMS':
            # init LMS filter
            self.filt = pa.filters.FilterLMS(n=filterSize, mu=mu, w='zeros')
        
        elif model.upper() == 'NLMS':
            # initialize NLMS filter
            self.filt = pa.filters.FilterNLMS(n=filterSize, mu=mu, w='zeros')
        
        else:
            raise ValueError(f"Unsupported model '{model}'. Please use 'LMS' or 'NLMS'.")
        
    def filtering(self, desired_signal, input_signal):
        
        # ensures correct shapes for filtering
        desired_signal = desired_signal.flatten()
        y_hat, e, w = self.filt.run(desired_signal, input_signal)

        return y_hat, e, w

#############################################################################################

def wavelet_adapt_filtering(filter_param, 
                            aecg_vec, 
                            mecg_vec, 
                            fs):
    
    # flatten + normalize signals
    aecg_vec = aecg_vec.flatten()
    mecg_vec = mecg_vec.flatten()
    aecg_vec = normalize_ecg(aecg_vec, 0, 5 * fs)
    mecg_vec = normalize_ecg(mecg_vec, 0, 5 * fs)

    # wavelet transform on caregiver ECG
    mecg_stk = wavelet_transform(mecg_vec, 'db4', 4)

    # apply adaptive filtering
    y, e, w = filter_param.filtering(aecg_vec, mecg_stk)

    return y, e, w

def normalize_ecg(ecg, 
                  start, 
                  stop):
    
    # extract segment for normalization
    ecg_segment = ecg[start:stop]
    if len(ecg_segment) == 0:
        scaling = 1.0
    else:
        # calc scaling factor
        scaling = 1.0 / (np.max(ecg_segment) - np.min(ecg_segment) + 1e-8)

    # scale ECG signal
    scaled_ecg = ecg * scaling
    
    
    # subtract mean
    mean_ecg = np.mean(scaled_ecg[start:stop]) if len(ecg_segment) > 0 else 0
    normalized_ecg = scaled_ecg - mean_ecg
    return normalized_ecg

#############################################################################################

# runs the program
if __name__ == "__main__":

    processor = ECGProcessor(fs=250)
    file_path = "recording" # don't include .csv at the end
    
    mixed_ecg, caregiver_ecg = processor.read_data(file_path)

    if mixed_ecg is not None and caregiver_ecg is not None:

        # preprocess signals (notch + band-pass)
        mixed_filtered, caregiver_filtered = processor.preprocess_signals(mixed_ecg, caregiver_ecg)

        # process signals to estimate maternal ECG and extract child ECG
        estimated_maternal, residual, _ = processor.process_signals(mixed_filtered, caregiver_filtered)

        # plot the results
        processor.plot_results(mixed_filtered, caregiver_filtered, estimated_maternal, residual, time_window=(2000, 5000))

    else:
        print("Failed to read ECG data.")

