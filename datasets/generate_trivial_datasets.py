#!/usr/bin/env python3
"""
Generate some extremely simple time-series datasets that the RNNs should be
able to get 100% classification accuracy on

Positive-slope -- Identify if the slope of a line is positive (2) or negative (1)
Positive-slope-noise -- same but with noise
Positive-sine -- Identify if a sine wave (2) or negative sine wave (1)
Positive-sine-noise -- same but with noise

Freq-low  -- classify low (1) or high (2) frequencies, but relatively low
Freq-high -- classify low (1) or high (2) frequencies, but higher than Freq-low
Freq-low-amp-noise -- ... but noisy vertically
Freq-high-amp-noise -- ... but noisy vertically
Freq-low-freq-noise -- ... but noisy in the frequency
Freq-high-freq-noise -- ... but noisy in the frequency

Phase-0 -- classify low (1) or high (2) frequencies, but with phase shift 0
Phase-90 -- classify low (1) or high (2) frequencies, but with phase shift 90 deg
Phase-0-amp-noise -- ... but noisy vertically
Phase-90-amp-noise -- ... but noisy vertically
Phase-0-phase-noise -- ... but noisy in the phase shift
Phase-90-phase-noise -- ... but noisy in the phase shift

FreqShift-low  -- + (2): 2,6,8 Hz; - (1): 7,9,11 Hz (some noise in freq/amp/phase)
FreqShift-high -- +6 -- +: 8,12,14 Hz; -: 13,15,17 Hz (...)
FreqScale-low  -- +: 2,6,8 Hz; -: 7,9,11 Hz (...)
FreqScale-high -- *2 -- +: 4,12,16 Hz; -: 14,18,22 Hz (...)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def linear(m, b, length=25, minvalue=0, maxvalue=10):
    x = np.arange(minvalue, maxvalue, (maxvalue-minvalue)/length).reshape(-1, 1).astype(np.float32)
    y = m*x + b
    return x, y


def display_xy(x, y):
    plt.figure()
    for i in range(y.shape[1]):
        plt.plot(x, y[:, i])
    plt.show()


def to_pandas(y, labels):
    """
    Note: y is y-axis but actually the data, i.e. "x" in (x,y) ML terminology
    """
    df = pd.DataFrame(y.T)
    df.insert(0, 'class', pd.Series(np.squeeze(labels).astype(np.int32)+1, index=df.index))
    return df


def generate_positive_slope_data(n, display=False, add_noise=False,
        bmin=0.0, bmax=10.0, m_mu=0.0, m_std=0.25):
    """ Positive or negative slope lines """
    m = np.random.normal(m_mu, m_std, (1, n))
    b = np.random.uniform(bmin, bmax, (1, n))
    x, y = linear(m, b)
    labels = m > 0

    if add_noise:
        noise = np.random.normal(0.0, 0.25, (y.shape[0], n))
        y += noise

    if display:
        display_xy(x, y)

    return to_pandas(y, labels)


def sine(m=1.0, b=0.0, f=None, freq_noise=1.0, phase_shift=5.0,
        length=25, mint=0, maxt=10):
    """
    Generate a single or multiple sine waves (multiple if f is list)

    m - scale horizontally
    b - offset vertically
    f - either one frequency, a list of frequencies for each example (see below)
    freq_noise - if not None, frequencies noisy about what is given in f
    phase_shift - how much to randomly offset in time (per frequency, so
        even with freq_noise=None, the two samples of f=[[1,2], [1,2]] will
        look different)
    length - number of samples to generate between mint and maxt
    mint - starting time
    maxt - stopping time

    100-length samples, one of 1 Hz and one of 2 Hz:
        sine(f=[[1], [2]], maxt=1, length=100, freq_noise=None, phase_shift=None)

    One 100-length sample with 1 and 2 Hz frequency components:
        sine(f=[[1,2]], maxt=1, length=100, freq_noise=None, phase_shift=None)

    Same frequency but different phase:
        sine(f=[[1]]*5, maxt=1, length=100, freq_noise=None, phase_shift=1.0)
    """
    multi_freq = isinstance(f, list) or isinstance(f, np.ndarray)

    # Set frequency if desired
    if f is None:
        s = np.array(1.0, dtype=np.float32)
    else:
        f = np.array(f, dtype=np.float32)

        if freq_noise is not None:
            f += np.random.normal(0.0, freq_noise, f.shape)

        s = 2.0*np.pi*f

    x_orig = np.arange(mint, maxt, (maxt-mint)/length).reshape(-1, 1)
    x = x_orig

    if multi_freq:
        x_tile = np.tile(x, f.shape[0])  # e.g. (100,1) to (100,3) if 100 time steps, 3 frequencies
        x_newdim = np.expand_dims(x_tile, 2)  # now (100,3,1) so broadcasting works below
        x = x_newdim

    if phase_shift is None:
        phase_shift = 0
    else:
        if f is None:
            if isinstance(m, np.ndarray):
                shape = m.shape
            elif isinstance(b, np.ndarray):
                shape = b.shape
            else:
                raise NotImplementedError("When using phase shift one of m, b, "
                    "or f must be np.ndarray")
        else:
            # shape = (num_freq, num_examples)
            shape = f.shape

        phase_shift = np.random.normal(0.0, phase_shift, shape)

    y = m*np.sin(s*(x+phase_shift)) + b

    # Sum the extra dimension for multiple frequencies, e.g. from (100,3,2)
    # back to (100,3) if 2 frequencies each
    if multi_freq:
        y = np.sum(y, axis=2)

    return x_orig, y


def generate_positive_sine_data(n, display=False, add_noise=False,
        bmin=0.0, bmax=10.0, m_mu=0.0, m_std=1.0):
    """ Sine wave multiplied by positive or negative number and offset some """
    m = np.random.normal(m_mu, m_std, (1, n))
    b = np.random.uniform(bmin, bmax, (1, n))
    x, y = sine(m=m, b=b)
    labels = m > 0

    if add_noise:
        noise = np.random.normal(0.0, 0.1, (y.shape[0], n))
        y += noise

    if display:
        display_xy(x, y)

    return to_pandas(y, labels)


def generate_freq(n, display=False, amp_noise=False, freq_noise=False,
        fmin=1.0, fmax=2.0):
    """ Sine wave multiplied by positive or negative number and offset some
    Warning: probably recall Nyquist when setting fmax
    """
    freq = np.random.uniform(fmin, fmax, (n, 1))

    if freq_noise:
        freq += np.random.normal(0.0, 1.0, (n, 1))  # on order of freq diffs

    x, y = sine(f=freq, maxt=2, length=2*50)
    labels = freq > (fmax-fmin)/2

    if amp_noise:
        y += np.random.normal(0.0, 0.1, (y.shape[0], n))

    if display:
        display_xy(x, y)

    return to_pandas(y, labels)


def generate_multi_freq(n, pos_f, neg_f, display=False,
        amp_noise=0.1, freq_noise=1.0, phase_shift=5.0,
        sample_freq=50):
    """
    Generate data with different sets of frequencies for +/- classes
    """
    pos_f = np.array(pos_f, dtype=np.float32)
    neg_f = np.array(neg_f, dtype=np.float32)

    # Generate the labels, ~1/2 from + and 1/2 from - classes
    labels = np.random.randint(2, size=n)

    # Get approximately the pos_f/neg_f frequencies for each
    freqs = []

    for label in labels:
        f = neg_f if label == 0 else pos_f
        freqs.append(f)

    freqs = np.array(freqs, dtype=np.float32)

    # Generate time series data
    x, y = sine(f=freqs, maxt=2, length=2*sample_freq, freq_noise=freq_noise,
        phase_shift=phase_shift)

    if amp_noise is not None:
        y += np.random.normal(0.0, amp_noise, (y.shape[0], n))

    if display:
        display_xy(x, y)

    return to_pandas(y, labels)


def save_data(func, fn, display=False):
    """ Use func to create examples that are saved to fn_TRAIN and fn_TEST """
    print(fn)
    func(10000, False).to_csv('trivial/'+fn+'_TRAIN', header=False, index=False)
    func(200, display).to_csv('trivial/'+fn+'_TEST', header=False, index=False)


if __name__ == '__main__':
    if not os.path.exists('trivial'):
        os.makedirs('trivial')

    # Whether to display
    dsp = False

    # For reproducibility
    np.random.seed(0)
    # No noise
    save_data(lambda x, dsp: generate_positive_slope_data(x, display=dsp), 'positive_slope', dsp)
    save_data(lambda x, dsp: generate_positive_sine_data(x, display=dsp), 'positive_sine', dsp)
    # Noisy
    save_data(lambda x, dsp: generate_positive_slope_data(x, add_noise=True, display=dsp), 'positive_slope_noise', dsp)
    save_data(lambda x, dsp: generate_positive_sine_data(x, add_noise=True, display=dsp), 'positive_sine_noise', dsp)
    # No noise - but different y-intercept
    save_data(lambda x, dsp: generate_positive_slope_data(x, bmin=20.0, bmax=30.0, display=dsp), 'positive_slope_low', dsp)
    save_data(lambda x, dsp: generate_positive_sine_data(x, bmin=20.0, bmax=30.0, display=dsp), 'positive_sine_low', dsp)

    # Frequency
    np.random.seed(0)
    save_data(lambda x, dsp: generate_freq(x, fmin=0.1, fmax=0.5, display=dsp), 'freq_low', dsp)
    save_data(lambda x, dsp: generate_freq(x, fmin=1.0, fmax=5.0, display=dsp), 'freq_high', dsp)
    save_data(lambda x, dsp: generate_freq(x, fmin=0.1, fmax=0.5, amp_noise=True, display=dsp), 'freq_low_amp_noise', dsp)
    save_data(lambda x, dsp: generate_freq(x, fmin=1.0, fmax=5.0, amp_noise=True, display=dsp), 'freq_high_amp_noise', dsp)
    save_data(lambda x, dsp: generate_freq(x, fmin=0.1, fmax=0.5, freq_noise=True, display=dsp), 'freq_low_freq_noise', dsp)
    save_data(lambda x, dsp: generate_freq(x, fmin=1.0, fmax=5.0, freq_noise=True, display=dsp), 'freq_high_freq_noise', dsp)
    save_data(lambda x, dsp: generate_freq(x, fmin=0.1, fmax=0.5, amp_noise=True, freq_noise=True, display=dsp), 'freq_low_freqamp_noise', dsp)
    save_data(lambda x, dsp: generate_freq(x, fmin=1.0, fmax=5.0, amp_noise=True, freq_noise=True, display=dsp), 'freq_high_freqamp_noise', dsp)

    # Multiple frequencies
    np.random.seed(0)
    save_data(lambda x, dsp: generate_multi_freq(x, [2, 6, 8], [7, 9, 11], display=dsp), 'freqshift_low', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, [8, 12, 14], [13, 15, 17], display=dsp), 'freqshift_high', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, [2, 6, 8], [7, 9, 11], display=dsp), 'freqscale_low', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, [4, 12, 16], [14, 18, 22], display=dsp), 'freqscale_high', dsp)
