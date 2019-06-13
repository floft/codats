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


def sine(m=1.0, b=0.0, f=None, length=25, mint=0, maxt=10):
    # Set frequency if desired
    if f is None:
        s = 1.0
    else:
        s = 2.0*np.pi*f

    x = np.arange(mint, maxt, (maxt-mint)/length).reshape(-1, 1)
    y = m*np.sin(s*x) + b

    return x, y


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
        fmin=1.0, fmax=2.0, m_mu=0.0, m_std=1.0):
    """ Sine wave multiplied by positive or negative number and offset some

    Warning: fmax should be no more than 1/2*length (by default 25) of time-series
    samples, i.e. no more than 12.5 Hz here.
    """
    freq = np.random.uniform(fmin, fmax, (1, n))

    if freq_noise:
        freq += np.random.normal(0.0, 1.0, (1, n))  # on order of freq diffs

    x, y = sine(f=freq, maxt=1)
    labels = freq > (fmax-fmin)/2  # half + half -

    if amp_noise:
        y += np.random.normal(0.0, 0.1, (y.shape[0], n))

    if display:
        display_xy(x, y)

    return to_pandas(y, labels)


def save_data(func, fn, display=False):
    """ Use func to create examples that are saved to fn_TRAIN and fn_TEST """
    func(10000, False).to_csv('trivial/'+fn+'_TRAIN', header=False, index=False)
    func(200, display).to_csv('trivial/'+fn+'_TEST', header=False, index=False)


if __name__ == '__main__':
    # For reproducibility
    np.random.seed(0)

    if not os.path.exists('trivial'):
        os.makedirs('trivial')

    # Whether to display
    dsp = True

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
    save_data(lambda x, dsp: generate_freq(x, fmin=1.0, fmax=3.0, display=dsp), 'freq_low', dsp)
    save_data(lambda x, dsp: generate_freq(x, fmin=7.0, fmax=9.0, display=dsp), 'freq_high', dsp)
    save_data(lambda x, dsp: generate_freq(x, fmin=1.0, fmax=3.0, amp_noise=True, display=dsp), 'freq_low_amp_noise', dsp)
    save_data(lambda x, dsp: generate_freq(x, fmin=7.0, fmax=9.0, amp_noise=True, display=dsp), 'freq_high_amp_noise', dsp)
    save_data(lambda x, dsp: generate_freq(x, fmin=1.0, fmax=3.0, freq_noise=True, display=dsp), 'freq_low_freq_noise', dsp)
    save_data(lambda x, dsp: generate_freq(x, fmin=7.0, fmax=9.0, freq_noise=True, display=dsp), 'freq_high_freq_noise', dsp)
    save_data(lambda x, dsp: generate_freq(x, fmin=1.0, fmax=3.0, amp_noise=True, freq_noise=True, display=dsp), 'freq_low_freqamp_noise', dsp)
    save_data(lambda x, dsp: generate_freq(x, fmin=7.0, fmax=9.0, amp_noise=True, freq_noise=True, display=dsp), 'freq_high_freqamp_noise', dsp)
