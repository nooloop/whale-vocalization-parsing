import numpy as np
import soundfile as sf
import pandas as pd
import random
import os
import sklean
import scipy
import bokeh
import librosa
import fastdtw
import IPython.display as ipd
import matplotlib.pyplot as plt

# this function loads audio:

def load_audio(filename):
    
    # read in file and convert to range [-1, 1]:
    
    srate, audio = scipy.io.wavfile.read(filename)
    audio = audio.astype(np.float32) / 32767.0 
    
    # set max to 0.9:
    
    if (len(audio.shape) == 1): 
        audio = (0.9 / max(np.abs(audio)) * audio)
    else: 
        audio[:,0] = (0.9 / max(np.abs(audio[:,0])) * audio[:,0])
        audio[:,1] = (0.9 / max(np.abs(audio[:,1])) * audio[:,1])
        return audio.transpose(), srate
    
    # return audio:
    
    return audio, srate 
	
# this function plots the time-domain representation of an audio file:

def plot_time_domain(audio, srate): 
    p = figure(plot_width=800, plot_height=200, x_axis_label='Time (s)', y_axis_label='Amplitude')
    time = np.linspace(0, len(audio)/srate, num=len(audio))
    p.line(time, audio)
    show(p)
	
# this function plots the frequency-domain representation of an audio file:

def plot_freq_domain(audio, srate):
    f, t, s = scipy.signal.spectrogram(audio, srate)
    s = 10 * np.log10(s + 1e-40)
    p = figure(plot_width=800, plot_height=400, x_axis_label='Time (s)', y_axis_label='Frequency (Hz)')
    p.image(image=[s], x=0, y=0, dw=t[-1], dh=f[-1], palette="Viridis256", level="image")
    show(p)
	
# this function denoises an audio file:

def denoise(
    audio, 
    noise, 
    n_grad_freq   = 3,
    n_grad_time   = 4,
    n_fft         = 2048,
    win_length    = 2048,
    hop_length    = 512,
    n_std_thresh  = 1,
    prop_decrease = 1.0
):
    """
    
    Remove noise from audio based upon a clip containing only noise

    Args:
        audio (array)        : audio to denoise
        noise (array)        : noise sample
        n_grad_freq (int)    : how many frequency channels to smooth over with the mask.
        n_grad_time (int)    : how many time channels to smooth over with the mask.
        n_fft (int)          : number audio of frames between STFT columns.
        win_length (int)     : Each frame of audio is windowed by `window()`. The window will be of length `win_length` and then padded with zeros to match `n_fft`..
        hop_length (int)     : number audio of frames between STFT columns.
        n_std_thresh (int)   : how many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal

    Returns:
        
        (array) The recovered signal with noise subtracted

    """

    # STFT over noise:
    
    stft_noise    = librosa.stft(y = noise, n_fft = n_fft, hop_length = hop_length, win_length = win_length)
    stft_noise_db = librosa.core.amplitude_to_db(np.abs(stft_noise))
    
    # calculate statistics over noise and noise threshold, over frequency axis:
    
    noise_freq_mean = np.mean(stft_noise_db, axis=1)
    noise_freq_std  = np.std(stft_noise_db, axis=1)
    noise_thresh    = noise_freq_mean + noise_freq_std * n_std_thresh
    
    # STFT over signal:

    stft_audio    = librosa.stft(y = audio, n_fft = n_fft, hop_length = hop_length, win_length = win_length)
    stft_audio_db = librosa.core.amplitude_to_db(np.abs(stft_audio))
    
    # calculate value to mask dB to:
    
    mask_gain_dB = np.min(stft_audio_db)
    
    # create a smoothing filter for the mask in time and frequency:
    
    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    
    # calculate the threshold for each frequency/time bin:
    
    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(noise_freq_mean)]),
        np.shape(stft_audio_db)[1],
        axis=0,
    ).T
    
    # mask if the signal is above the threshold:
    
    mask_audio = stft_audio_db < db_thresh

    # convolve the mask with a smoothing filter:
    
    mask_audio = scipy.signal.fftconvolve(mask_audio, smoothing_filter, mode = "same")
    mask_audio = mask_audio * prop_decrease

    # mask the signal:
    
    stft_audio_db_masked = (stft_audio_db * (1 - mask_audio) + np.ones(np.shape(mask_gain_dB)) 
                            * mask_gain_dB * mask_audio)
    
    # mask real:
    
    audio_imag_masked = np.imag(stft_audio) * (1 - mask_audio)
    stft_audio_amp = (librosa.core.db_to_amplitude(stft_audio_db_masked) * np.sign(stft_audio)) + (
        1j * audio_imag_masked
    )

    # recover the signal:
    
    recovered_signal = librosa.istft(stft_audio_amp, hop_length = hop_length, win_length = win_length)

    return recovered_signal
	
# this function computes and visualizes the cross-correlation (or autocorrelation) between two signals:

def correlation(audio_probe, audio_query, srate):
    
    xcorrelation = np.correlate(audio_probe, audio_query, mode = 'full')
    
    p = figure(plot_width=800, plot_height=200, x_axis_label='Delay (s)', y_axis_label='Cross-Correlation')
    delay = np.linspace(0, len(audio_query), num=len(audio_query))
    p.line(delay, xcorrelation)
    show(p)
    
    return xcorrelation
	
# these functions implements different monophonic pitch detection methods:

def pitch_zero_crossings(frame, srate): 
    
    zero_indices   = np.nonzero((frame[1:] >= 0) & (frame[:-1] < 0))[0]
    pitch_estimate = (srate / np.mean(np.diff(indices)))
    
    return pitch_estimate 

def pitch_fft(frame, srate): 
    
    mag            = np.abs(np.fft.fft(frame))
    mag            = mag[0:int(len(mag)/2)]
    pitch_estimate = np.argmax(mag) * (srate / len(frame))
    
    return pitch_estimate 

def pitch_autocorrelation(frame, srate):
    
    xcorrelation   = np.correlate(frame, frame, mode = 'full')
    derivative     = np.diff(xcorrelation[:int(len(xcorrelation)/2)+2])
    peak_indices   = np.nonzero((derivative[:-1] > 0) & (derivative[1:] <= 0))[0] + 1
    peak_values    = xcorrelation[peak_indices]
    peak_indices_sorted = peak_indices[np.argsort(peak_values)[-2:]]
    
    return srate/(peak_indices_sorted[1]-peak_indices_sorted[0])

def pitch_track(signal, hopSize, winSize, extractor): 
    
    offsets = np.arange(0, len(signal), hopSize)
    pitch_track = np.zeros(len(offsets))
    amp_track = np.zeros(len(offsets))
    
    for (m, o) in enumerate(offsets): 
        frame = signal[o:o+winSize] 
        pitch_track[m] = extractor(frame, srate)
        amp_track[m] = np.sqrt(np.mean(np.square(frame)))  

        if (pitch_track[m] > 1500): 
            pitch_track[m] = 0 
    
    return (amp_track, pitch_track)

def sonify(amp_track, pitch_track, srate, hop_size):

    times = np.arange(0.0, float(hop_size * len(pitch_track)) / srate,
                      float(hop_size) / srate)

    # sample locations in time (seconds)                                                      
    sample_times = np.linspace(0, np.max(times), int(np.max(times)*srate-1))

    freq_interpolator = interp1d(times,pitch_track)
    amp_interpolator = interp1d(times,amp_track)
                                                                
    sample_freqs = freq_interpolator(sample_times)
    sample_amps  = amp_interpolator(sample_times)

    audio = np.zeros(len(sample_times));
    T = 1.0 / srate
    phase = 0.0
    
    for i in range(1, len(audio)):
        audio[i] = sample_amps[i] * np.sin(phase)
        phase = phase + (2*np.pi*T*sample_freqs[i])

    return audio
	
# these functions compute the dynamic time warping between two audio signals

def dtw_table(x, y, distance = None):
    
    if distance is None:
        distance = scipy.spatial.distance.euclidean
    nx = len(x)
    ny = len(y)
    table = np.zeros((nx+1, ny+1))
    
    # compute left column separately, i.e. j=0.
    table[1:, 0] = np.inf
        
    # compute top row separately, i.e. i=0.
    table[0, 1:] = np.inf
        
    # Fill in the rest.
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            d = distance(x[i-1], y[j-1])
            table[i, j] = d + min(table[i-1, j], table[i, j-1], table[i-1, j-1])
    return table

def dtw_path(x, y, table):
    
    i = len(x)
    j = len(y)
    path = [(i, j)]
    while i > 0 or j > 0:
        minval = np.inf
        if table[i-1][j-1] < minval:
            minval = table[i-1, j-1]
            step = (i-1, j-1)
        if table[i-1, j] < minval:
            minval = table[i-1, j]
            step = (i-1, j)
        if table[i][j-1] < minval:
            minval = table[i, j-1]
            step = (i, j-1)
        path.insert(0, step)
        i, j = step
    
    return np.array(path)

def plot_dtw(table, path, signal1, signal2):
    
    %matplotlib widget

    plt.figure(figsize = (10, 10))

    # Bottom right plot.
    ax1 = plt.axes([0.2, 0, 0.8, 0.2])
    ax1.imshow(signal1, origin = 'upper', aspect = 'auto', cmap = 'coolwarm')
    ax1.set_xlabel('Signal 1')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_ylim(10)

    # Top left plot.
    ax2 = plt.axes([0, 0.2, 0.20, 0.8])
    ax2.imshow(signal2.T, origin = 'lower', aspect = 'auto', cmap = 'coolwarm')
    ax2.set_ylabel('Signal 2')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_ylim(1)

    # Top right plot.
    ax3 = plt.axes([0.2, 0.2, 0.8, 0.8], sharex = ax1, sharey = ax2)
    ax3.imshow(table.T, aspect = 'auto', origin = 'upper', interpolation = 'nearest', cmap = 'gray')
    ax3.set_xticks([])
    ax3.set_yticks([])

    # Path.
    ax3.plot(path[:,0], path[:,1], 'r')
    
# this function trims a call file based its amplitude track:

def trim(audio, amplitude_track, srate, binSize = 6, hopSize = 256, threshold = 2.5, sensitivity = 5):

    begin_noise_level = np.mean(amplitude_track[:int(len(amplitude_track)/binSize)])
    end_noise_level   = np.mean(amplitude_track[-int(len(amplitude_track)/binSize):])
        
    index_first = None
    index_last  = None
    
    for i in range(len(amplitude_track)):
        if amplitude_track[i] > threshold * begin_noise_level:
            if all(amplitude_track[j] > threshold * begin_noise_level for j in range(i, i + sensitivity + 1)):
                index_first = i
                break
                
    for i in range(len(amplitude_track)):
        if amplitude_track[i] > threshold * end_noise_level:
            if all(amplitude_track[j] > threshold * end_noise_level for j in range(i - sensitivity + 2, i + 1)):
                index_last = i
    
    return audio[index_first * hopSize:(index_last + 10) * hopSize]
    
# this function takes a probe sequence and a query sequence, and locates the probe in the query:

def probe_localization(probe_audio, query_audio, winSize, srate, threshold):
    
    # get correlation between probe and query, and probe and probe:
    
    correlation_pq = np.correlate(probe_audio, query_audio, mode = 'same')[::-1]
    correlation_pp = np.correlate(probe_audio, probe_audio, mode = 'same')
    
    # define localization threshold based on probe-probe correlation:
    
    thresh = threshold * max(correlation_pp)

    offsets = np.arange(0, len(correlation_pq), winSize)
    amp = np.zeros(len(offsets))
    
    for (m, o) in enumerate(offsets): 
        frame = correlation_pq[o:o+winSize] 
        amp[m] = np.max(np.abs(frame))
    
    localizations = np.array([i for i in range(1, len(amp) - 1) 
                              if (amp[i] > amp[i - 1] and amp[i] > amp[i + 1] and amp[i] > thresh)])
    
    return amp, localizations * winSize / srate
    
# these functions measure the similarity between two calls, using a variety of methods:

def similarity_cross_correlation(audio_1, audio_2):
    
    correlation = np.correlate(audio_1, audio_2, mode = "same")
    
    return max(correlation)

def similarity_dynamic_time_warping(audio_1, audio_2):
    
    distance, path = fastdtw(call_1, call_2)
    
    return distance

def similarity_mfcc(audio_1, audio_2, srate):
    
    mfccs_1 = librosa.feature.mfcc(y = audio_1, sr = srate)
    mfccs_2 = librosa.feature.mfcc(y = audio_2, sr = srate)
    
    distance = np.linalg.norm(mfccs_1 - mfccs_2)
    
    return distance
    
# this function takes an array of feature vectors, and clusters them:

def cluster_adaptive_kmeans(data, max_clusters = 50, tolerance = 0.01):
    
    kmeans = sklearn.cluster.Kmeans(n_clusters = 10, random_state = 0).fit(data)
    prev_inertia = kmeans.inertia_
    
    for n_clusters in range(11, max_clusters + 1):
        
        kmeans = KMeans(n_clusters = n_clusters, random_state = 0).fit(data)
        inertia = kmeans.inertia_
        
        if (prev_inertia - inertia) / prev_inertia < tolerance:
            break
        
        prev_inertia = inertia
    
    return kmeans

def cluster_dbscan(data, eps = 0.5, min_samples = 5):
    
    dbscan = sklearn.cluster.DBSCAN(eps = eps, min_samples = min_samples)    
    dbscan.fit(data)
    labels = dbscan.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    return labels, n_clusters_, n_noise_