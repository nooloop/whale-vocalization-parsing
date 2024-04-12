# Whale Vocalization Parsing

This repository contains a set of digital signal processing tools and methods for denoising whale vocalizations and extracting information from these vocalizations. Tools are located in `wvp_tools.py`, and are based on the `numpy`, `scipy`, `matplotlib`, `librosa`, and `bokeh` Python libraries. 

`wvp_tools_and_methods.ipynb` is an interactive Python notebook that implements in a central location these tools and methods for development purposes. It requires that users have access to their own database of whale vocalization tracks in `.wav` form (though adaptation is easily possible to accommodate other audio formats).

## Tools

* Audio loading
* Audio visualization (time- and frequency-domain representations)
* Audio denoising
* Audio pitch extraction (zero-crossing, FFT peak selection, autocorrelation)
* Audio pitch track sonification
* Audio dynamic time warping
* Audio track comparison (cross-correlation, dynamic time warping)
* Audio trimming
* Audio probe localization (time domain, time-frequency domain)
* Audio features exrtaction (spectral centroid, bandwith, etc., MFCCs, chroma, tonnetz)
* Audio clustering (adaptive k-means, DBSCAN)

## Methods

* Detecting all type changes in time (linear regression agains FF estimation)
* Characterizing vocalization variance (DTW, PCA)
* n-gram analysis
