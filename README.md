# Whale Vocalization Parsing

This repository contains a set of digital signal processing tools and methods for denoising whale vocalizations and extracting information from these vocalizations. Tools are located in `whale_tools.py`, and are based on the `numpy`, `scipy`, `matplotlib`, `librosa`, and `bokeh` Python libraries. Methods are located in `whale_methods.py`, which inherits functions from `whale_tools.py`.

`whale_tools_and_methods.ipynb` is an interactive Python notebook that implements in a central location these tools and methods for development purposes. It requires that users have access to their own database of whale vocalization tracks in `.wav` form (though adaptation is easily possible to accommodate other audio formats).

## Tools

* Audio loading
* Audio visualization (time- and frequency-domain representations)
* Audio denoising
* Audio pitch extraction (zero-crossing, FFT peak selection, autocorrelation)
* Audio pitch track sonification
* Audio track comparison (cross-correlation, dynamic time warping)
* Audio trimming

## Methods

* Characterizing vocalization variance
* n-gram analysis
