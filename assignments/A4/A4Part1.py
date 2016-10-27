import sys, os
import numpy as np
from scipy.signal import get_window
from scipy.fftpack import fft, fftshift
import math
import matplotlib
import matplotlib.pyplot as plt
eps = np.finfo(float).eps

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
from dftModel import dftAnal

"""
A4-Part-1: Extracting the main lobe of the spectrum of a window

Write a function that extracts the main lobe of the magnitude spectrum of a window given a window
type and its length (M). The function should return the samples corresponding to the main lobe in
decibels (dB).

To compute the spectrum, take the FFT size (N) to be 8 times the window length (N = 8*M) (For this
part, N need not be a power of 2).

The input arguments to the function are the window type (window) and the length of the window (M).
The function should return a numpy array containing the samples corresponding to the main lobe of
the window.

In the returned numpy array you should include the samples corresponding to both the local minimas
across the main lobe.

The possible window types that you can expect as input are rectangular ('boxcar'), 'hamming' or
'blackmanharris'.

NOTE: You can approach this question in two ways: 1) You can write code to find the indices of the
local minimas across the main lobe. 2) You can manually note down the indices of these local minimas
by plotting and a visual inspection of the spectrum of the window. If done manually, the indices
have to be obtained for each possible window types separately (as they differ across different
window types).

Tip: log10(0) is not well defined, so its a common practice to add a small value such as eps = 1e-16
to the magnitude spectrum before computing it in dB. This is optional and will not affect your answers.
If you find it difficult to concatenate the two halves of the main lobe, you can first center the
spectrum using fftshift() and then compute the indexes of the minimas around the main lobe.


Test case 1: If you run your code using window = 'blackmanharris' and M = 100, the output numpy
array should contain 65 samples.

Test case 2: If you run your code using window = 'boxcar' and M = 120, the output numpy array
should contain 17 samples.

Test case 3: If you run your code using window = 'hamming' and M = 256, the output numpy array
should contain 33 samples.

"""
def my_dft(x, N):
    """
    x - signal
    N - FFT length (doesn't have to be pow 2, does have to be >= len(x))
    """
    M = len(x)
    assert N >= M, \
        'FFT buffer N [{}] must be >= length of signal [{}].'.format(N, M)
    len_x_half1 = int(math.floor((M)/2))
    len_x_half2 = int(math.floor((M+1)/2))
    fft_buffer = np.zeros(N)
    fft_buffer[:len_x_half2] = x[len_x_half1:]
    fft_buffer[N-len_x_half1:] = x[:len_x_half1]
    X = fft(fft_buffer)
    mX = np.absolute(X)
    mX[mX < eps] = eps
    mX = 20*np.log10(mX)
    # mX = mX - np.max(mX)
    pX = np.angle(X)
    return mX, pX


def extractMainLobe(window, M):
    """
    Input:
            window (string): Window type to be used (Either rectangular ('boxcar'), 'hamming' or '
                blackmanharris')
            M (integer): length of the window to be used
    Output:
            The function should return a numpy array containing the main lobe of the magnitude
            spectrum of the window in decibels (dB).
    """
    x = get_window(window, M)         # get the window
    N = 8*M
    mX, pX = my_dft(x, N)
    prev_m = mX[0]
    main_lobe_width = 0
    for ii in range(len(x)):
        m = mX[ii]
        if m <= prev_m:
            main_lobe_width += 1
            prev_m = m
        else:
            break
    right_lobe = mX[:main_lobe_width]
    left_lobe = mX[-(main_lobe_width-1):]
    return np.hstack((left_lobe, right_lobe))


if __name__ == '__main__':
    M = 100
    N = 8*M
    window = 'blackmanharris'
    w = get_window(window, M)
    lobe1 = extractMainLobe(window, M)
    mX1, pX1 = my_dft(w, N)
    print len(lobe1), 65

    M = 120
    N = 8*M
    window = 'boxcar'
    w = get_window(window, M)
    lobe2 = extractMainLobe(window, M)
    mX2, pX2 = my_dft(w, N)
    print len(lobe2), 17

    M = 256
    N = 8*M
    window = 'hamming'
    w = get_window(window, M)
    lobe3 = extractMainLobe(window, M)
    mX3, pX3 = my_dft(w, N)
    print len(lobe3), 33


    # mX, pX = dftAnal(np.ones(M), w, N)
    # mX = mX - np.max(mX)
    # plt.figure()
    # plt.plot(np.arange(N), np.fft.fftshift(mX))
    # plt.figure()
    # plt.plot(np.arange(N), np.fft.fftshift(pX))
    # plt.show()
