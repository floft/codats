#!/usr/bin/env python3
"""
Generate some extremely simple time-series datasets that the RNNs should be
able to get 100% classification accuracy on

v1:
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

v2 (invertible):

"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def linear(m, b, length=100, minvalue=0, maxvalue=2):
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
        bmin=0.0, bmax=5.0, mmin=-1.0, mmax=1.0):
    """ Positive or negative slope lines """
    m = np.random.uniform(mmin, mmax, (1, n))
    b = np.random.uniform(bmin, bmax, (1, n))
    x, y = linear(m, b)
    labels = m > 0

    if add_noise:
        noise = np.random.normal(0.0, 0.25, (y.shape[0], n))
        y += noise

    if display:
        display_xy(x, y)

    return y, labels


def sine(m=1.0, b=0.0, f=None, amps=None, freq_noise=1.0, phase_shift=5.0,
        length=100, mint=0, maxt=2):
    """
    Generate a single or multiple sine waves (multiple if f is list)

    m - scale horizontally
    b - offset vertically
    f - either one frequency, a list of frequencies for each example (see below)
    amps - if specified, the amplitude of each specified frequency
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
        amps = np.array(1.0, dtype=np.float32)
    else:
        if amps is not None:
            amps = np.array(amps, dtype=np.float32)
        else:
            amps = np.array([1.0]*len(f), dtype=np.float32)

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

    y = m*amps*np.sin(s*(x+phase_shift))

    # Sum the extra dimension for multiple frequencies, e.g. from (100,3,2)
    # back to (100,3) if 2 frequencies each
    #
    # Note: make sure we don't add b in y above before summing, since otherwise
    # it'll be shifted much more than b
    if multi_freq:
        y = np.sum(y, axis=2) + b
    else:
        y += b

    return x_orig, y


def generate_positive_sine_data(n, display=False, add_noise=False,
        bmin=0.0, bmax=5.0, mmin=-1.0, mmax=1.0, f=None):
    """ Sine wave multiplied by positive or negative number and offset some """
    m = np.random.uniform(mmin, mmax, (1, n))
    b = np.random.uniform(bmin, bmax, (1, n))
    x, y = sine(m=m, b=b, f=f, freq_noise=None, phase_shift=None)
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

    return y, labels


def generate_multi_freq(n, pos_f_all, neg_f_all,
        pos_amp_all=None, neg_amp_all=None,
        display=False,
        amp_noise=0.1, freq_noise=1.0, phase_shift=5.0,
        sample_freq=50, duration=2, b=0.0):
    """
    Generate data with different sets of frequencies for +/- classes

    Optionally specify different amplitudes for each of the frequencies. If not,
    then they all have the same amplitude (possibly with amplitude noise if
    amp_noise is not None).

    Note: {pos,neg}_{f,amp} is a 2D list to split frequencies/amplitudes across
    multiple channels. If you only want one channel, then do
        generate_multi_freq(x, [run_f], [walk_f], [run_amp], [walk_amp], ...)
    """
    assert pos_amp_all is None or len(pos_amp_all) == len(pos_f_all), \
        "pos_amp_all must be same length as pos_f_all"
    assert neg_amp_all is None or len(neg_amp_all) == len(neg_f_all), \
        "neg_amp_all must be same length as neg_f_all"

    # Generate the labels, ~1/2 from + and 1/2 from - classes
    labels = np.random.randint(2, size=n)

    # Multi-dimensional data
    data = []

    # For each channel
    for channel in range(len(pos_f_all)):
        pos_f = np.array(pos_f_all[channel], dtype=np.float32)
        neg_f = np.array(neg_f_all[channel], dtype=np.float32)

        if pos_amp_all is not None:
            pos_amp = np.array(pos_amp_all[channel], dtype=np.float32)
        else:
            pos_amp = np.array([1.0]*len(pos_f_all), dtype=np.float32)

        if neg_amp_all is not None:
            neg_amp = np.array(neg_amp_all[channel], dtype=np.float32)
        else:
            neg_amp = np.array([1.0]*len(neg_f_all), dtype=np.float32)

        assert pos_amp is None or len(pos_amp) == len(pos_f), \
            "pos_amp must be same length as pos_f"
        assert neg_amp is None or len(neg_amp) == len(neg_f), \
            "neg_amp must be same length as neg_f"

        # Match sizes by zero padding, otherwise we can't convert to single matrix
        if len(pos_f) > len(neg_f):
            padding = len(pos_f) - len(neg_f)
            neg_f = np.pad(neg_f, (0, padding), 'constant', constant_values=(0.0, 0.0))
            neg_amp = np.pad(neg_amp, (0, padding), 'constant', constant_values=(0.0, 0.0))
        elif len(neg_f) > len(pos_f):
            padding = len(neg_f) - len(pos_f)
            pos_f = np.pad(pos_f, (0, padding), 'constant', constant_values=(0.0, 0.0))
            pos_amp = np.pad(pos_amp, (0, padding), 'constant', constant_values=(0.0, 0.0))

        # Get approximately the pos_f/neg_f frequencies for each
        freqs = []
        amps = []

        for label in labels:
            f = neg_f if label == 0 else pos_f
            amp = neg_amp if label == 0 else pos_amp
            freqs.append(f)
            amps.append(amp)

        freqs = np.array(freqs, dtype=np.float32)
        amps = np.array(amps, dtype=np.float32)
        amps[np.isnan(amps)] = 1.0  # None is NaN, and so just set to 1.0 amplitude

        # Generate time series data
        x, y = sine(b=b, f=freqs, amps=amps, maxt=duration, length=duration*sample_freq,
            freq_noise=freq_noise, phase_shift=phase_shift)

        if amp_noise is not None:
            y += np.random.normal(0.0, amp_noise, (y.shape[0], n))

        if display:
            display_xy(x, y)

        data.append(y)

    # Transpose from [features, time_steps, examples] to
    # [examples, time_steps, features]
    data = np.array(data, dtype=np.float32).T

    # Make labels 1-indexed
    labels += 1

    return data, labels


def rotate2d(x, degrees):
    """
    Rotate the 2D data in x a certain number of degrees (clockwise, if each
    point in x is (x,y))

    If x is a single point (i.e. x is something like (x,y)) then it's only
    rotates that point. However, more useful is if x is a time series, where the
    values to be rotated (i.e. the feature dimension) is last. For example, pass
    x with shape: [examples, time_steps, num_features] where num_features = 2
    (since this is a 2D rotation matrix).

    Note: if you want counterclockwise, do a left-multiply instead of a
    right-multiply

    See:
    https://en.wikipedia.org/wiki/Rotation_matrix
    https://scipython.com/book/chapter-6-numpy/examples/creating-a-rotation-matrix-in-numpy/
    """
    theta = np.radians(degrees)
    c = np.cos(theta)
    s = np.sin(theta)
    rotation_matrix = np.array(((c, -s), (s, c)))
    return np.dot(x, rotation_matrix)


def rotate2d_data(data, labels, degrees):
    """ rotate2d but only rotates data and directly passes through labels """
    return rotate2d(data, degrees), labels


def save_data_file(values, labels, filename):
    """
    To be compatible with UCR dataset format with 1D data (univariate), commas
    separate label first then all the data with one example on each line.
    However, to support multivariate data, features for each time step are
    delimitated by semicolons.

    Example:
      univariate: label,timestep1,timestep2,timestep3,...
      multivariate: label,ts1f1;ts1f2;ts1f3,fs2f1;fs2f2;ts2f3,...
    """
    with open(filename, "w") as f:
        for i, x in enumerate(values):
            y = labels[i]
            s = str(y) + ","

            # If only one feature, we don't have the extra dimension
            if len(x.shape) == 1:
                s += ",".join([str(v) for v in x])
            elif len(x.shape) == 2:
                for j, time_step in enumerate(x):
                    s += ";".join([str(v) for v in time_step])

                    if j != len(x) - 1:
                        s += ","
            else:
                raise NotImplementedError(
                    "only support shapes [examples, time_steps]"
                    " or [examples, time_steps, features]")

            f.write(s+"\n")


def save_data(func, fn, display=False):
    """ Use func to create examples that are saved to fn_TRAIN and fn_TEST """
    print(fn)
    save_data_file(*func(10000, False), 'trivial/'+fn+'_TRAIN')
    save_data_file(*func(2000, display), 'trivial/'+fn+'_TEST')

    # Using pandas, but doesn't work with multi-dimensional data
    # to_pandas(*func(10000, False)).to_csv('trivial/'+fn+'_TRAIN', header=False, index=False)
    # to_pandas(*func(2000, display)).to_csv('trivial/'+fn+'_TEST', header=False, index=False)


if __name__ == '__main__':
    if not os.path.exists('trivial'):
        os.makedirs('trivial')

    # Whether to display
    dsp = False

    # # For reproducibility
    # np.random.seed(0)
    # # No noise
    # save_data(lambda x, dsp: generate_positive_slope_data(x, display=dsp), 'positive_slope', dsp)
    # save_data(lambda x, dsp: generate_positive_sine_data(x, display=dsp), 'positive_sine', dsp)
    # # Noisy
    # save_data(lambda x, dsp: generate_positive_slope_data(x, add_noise=True, display=dsp), 'positive_slope_noise', dsp)
    # save_data(lambda x, dsp: generate_positive_sine_data(x, add_noise=True, display=dsp), 'positive_sine_noise', dsp)
    # # No noise - but different y-intercept
    # save_data(lambda x, dsp: generate_positive_slope_data(x, bmin=20.0, bmax=30.0, display=dsp), 'positive_slope_low', dsp)
    # save_data(lambda x, dsp: generate_positive_sine_data(x, bmin=20.0, bmax=30.0, display=dsp), 'positive_sine_low', dsp)

    # # Frequency
    # np.random.seed(0)
    # save_data(lambda x, dsp: generate_freq(x, fmin=0.1, fmax=0.5, display=dsp), 'freq_low', dsp)
    # save_data(lambda x, dsp: generate_freq(x, fmin=1.0, fmax=5.0, display=dsp), 'freq_high', dsp)
    # save_data(lambda x, dsp: generate_freq(x, fmin=0.1, fmax=0.5, amp_noise=True, display=dsp), 'freq_low_amp_noise', dsp)
    # save_data(lambda x, dsp: generate_freq(x, fmin=1.0, fmax=5.0, amp_noise=True, display=dsp), 'freq_high_amp_noise', dsp)
    # save_data(lambda x, dsp: generate_freq(x, fmin=0.1, fmax=0.5, freq_noise=True, display=dsp), 'freq_low_freq_noise', dsp)
    # save_data(lambda x, dsp: generate_freq(x, fmin=1.0, fmax=5.0, freq_noise=True, display=dsp), 'freq_high_freq_noise', dsp)
    # save_data(lambda x, dsp: generate_freq(x, fmin=0.1, fmax=0.5, amp_noise=True, freq_noise=True, display=dsp), 'freq_low_freqamp_noise', dsp)
    # save_data(lambda x, dsp: generate_freq(x, fmin=1.0, fmax=5.0, amp_noise=True, freq_noise=True, display=dsp), 'freq_high_freqamp_noise', dsp)

    # # Multiple frequencies
    # # TODO maybe remove/decrease frequency/amplitude noise and reduce domain shifts
    # np.random.seed(0)
    # save_data(lambda x, dsp: generate_multi_freq(x, [2, 6, 8], [7, 9, 11], display=dsp), 'freqshift_low', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, [8, 12, 14], [13, 15, 17], display=dsp), 'freqshift_high', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, [2, 6, 8], [7, 9, 11], display=dsp), 'freqscale_low', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, [4, 12, 16], [14, 18, 22], display=dsp), 'freqscale_high', dsp)

    # #
    # # Invertible (for directly evaluating mapping)
    # #
    # # jumping mean - 1 with some overlap, 2 with no overlap
    # np.random.seed(0)
    # save_data(lambda x, dsp: generate_positive_slope_data(x, bmin=0.0, bmax=5.0, display=dsp), 'line1low', dsp)
    # save_data(lambda x, dsp: generate_positive_slope_data(x, bmin=2.5, bmax=7.5, display=dsp), 'line1high', dsp)
    # save_data(lambda x, dsp: generate_positive_slope_data(x, bmin=0.0, bmax=5.0, display=dsp), 'line2low', dsp)
    # save_data(lambda x, dsp: generate_positive_slope_data(x, bmin=5.0, bmax=10.0, display=dsp), 'line2high', dsp)

    # save_data(lambda x, dsp: generate_positive_sine_data(x, bmin=0.0, bmax=5.0, display=dsp), 'sine1low', dsp)
    # save_data(lambda x, dsp: generate_positive_sine_data(x, bmin=2.5, bmax=7.5, display=dsp), 'sine1high', dsp)
    # save_data(lambda x, dsp: generate_positive_sine_data(x, bmin=0.0, bmax=5.0, display=dsp), 'sine2low', dsp)
    # save_data(lambda x, dsp: generate_positive_sine_data(x, bmin=5.0, bmax=10.0, display=dsp), 'sine2high', dsp)

    # # see if network works better with higher-frequency data
    # save_data(lambda x, dsp: generate_positive_sine_data(x, bmin=0.0, bmax=5.0, f=10.0, display=dsp), 'sine3low', dsp)
    # save_data(lambda x, dsp: generate_positive_sine_data(x, bmin=2.5, bmax=7.5, f=10.0, display=dsp), 'sine3high', dsp)
    # save_data(lambda x, dsp: generate_positive_sine_data(x, bmin=0.0, bmax=5.0, f=10.0, display=dsp), 'sine4low', dsp)
    # save_data(lambda x, dsp: generate_positive_sine_data(x, bmin=5.0, bmax=10.0, f=10.0, display=dsp), 'sine4high', dsp)

    # # slope scaling - 1 with small scaling, 2 with more scaling
    # np.random.seed(0)
    # save_data(lambda x, dsp: generate_positive_slope_data(x, mmin=-1.0, mmax=1.0, display=dsp), 'lineslope1low', dsp)
    # save_data(lambda x, dsp: generate_positive_slope_data(x, mmin=-1.5, mmax=1.5, display=dsp), 'lineslope1high', dsp)
    # save_data(lambda x, dsp: generate_positive_slope_data(x, mmin=-1.0, mmax=1.0, display=dsp), 'lineslope2low', dsp)
    # save_data(lambda x, dsp: generate_positive_slope_data(x, mmin=-2.0, mmax=2.0, display=dsp), 'lineslope2high', dsp)

    # save_data(lambda x, dsp: generate_positive_sine_data(x, mmin=-1.0, mmax=1.0, display=dsp), 'sineslope1low', dsp)
    # save_data(lambda x, dsp: generate_positive_sine_data(x, mmin=-1.5, mmax=1.5, display=dsp), 'sineslope1high', dsp)
    # save_data(lambda x, dsp: generate_positive_sine_data(x, mmin=-1.0, mmax=1.0, display=dsp), 'sineslope2low', dsp)
    # save_data(lambda x, dsp: generate_positive_sine_data(x, mmin=-2.0, mmax=2.0, display=dsp), 'sineslope2high', dsp)

    # save_data(lambda x, dsp: generate_positive_sine_data(x, mmin=-1.0, mmax=1.0, f=10.0, display=dsp), 'sineslope3low', dsp)
    # save_data(lambda x, dsp: generate_positive_sine_data(x, mmin=-1.5, mmax=1.5, f=10.0, display=dsp), 'sineslope3high', dsp)
    # save_data(lambda x, dsp: generate_positive_sine_data(x, mmin=-1.0, mmax=1.0, f=10.0, display=dsp), 'sineslope4low', dsp)
    # save_data(lambda x, dsp: generate_positive_sine_data(x, mmin=-2.0, mmax=2.0, f=10.0, display=dsp), 'sineslope4high', dsp)

    #
    # Classification problem:
    # Walking (negative) vs. running (positive)
    #
    run_f = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    run_amp = np.array([0.75, 0.56, 0.38, 0.19, 0.08, 0.04], dtype=np.float32)
    walk_f = np.array([1, 2, 4], dtype=np.float32)
    walk_amp = np.array([0.5, 0.25, 0.06], dtype=np.float32)

    # Since we want 1D data and I modified generate_multi_freq to support
    # multi-dimensional data, but use numpy arrays so we can still do
    # addition/multiplication on these
    run_f = np.expand_dims(run_f, axis=0)
    run_amp = np.expand_dims(run_amp, axis=0)
    walk_f = np.expand_dims(walk_f, axis=0)
    walk_amp = np.expand_dims(walk_amp, axis=0)

    # Frequency shift
    np.random.seed(0)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None, None), 'freqshift_a', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f+0.0, walk_f+0.0, run_amp, walk_amp, dsp, None, None, None), 'freqshift_b0', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f+3.8, walk_f+3.8, run_amp, walk_amp, dsp, None, None, None), 'freqshift_b1', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f+7.6, walk_f+7.6, run_amp, walk_amp, dsp, None, None, None), 'freqshift_b2', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f+11.4, walk_f+11.4, run_amp, walk_amp, dsp, None, None, None), 'freqshift_b3', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f+15.2, walk_f+15.2, run_amp, walk_amp, dsp, None, None, None), 'freqshift_b4', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f+19.0, walk_f+19.0, run_amp, walk_amp, dsp, None, None, None), 'freqshift_b5', dsp)

    save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 'freqshift_phase_a', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f+0.0, walk_f+0.0, run_amp, walk_amp, dsp, None, None), 'freqshift_phase_b0', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f+3.8, walk_f+3.8, run_amp, walk_amp, dsp, None, None), 'freqshift_phase_b1', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f+7.6, walk_f+7.6, run_amp, walk_amp, dsp, None, None), 'freqshift_phase_b2', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f+11.4, walk_f+11.4, run_amp, walk_amp, dsp, None, None), 'freqshift_phase_b3', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f+15.2, walk_f+15.2, run_amp, walk_amp, dsp, None, None), 'freqshift_phase_b4', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f+19.0, walk_f+19.0, run_amp, walk_amp, dsp, None, None), 'freqshift_phase_b5', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f+22.8, walk_f+22.8, run_amp, walk_amp, dsp, None, None), 'freqshift_phase_b6', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f+26.6, walk_f+26.6, run_amp, walk_amp, dsp, None, None), 'freqshift_phase_b7', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f+30.4, walk_f+30.4, run_amp, walk_amp, dsp, None, None), 'freqshift_phase_b8', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f+34.2, walk_f+34.2, run_amp, walk_amp, dsp, None, None), 'freqshift_phase_b9', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f+38.0, walk_f+38.0, run_amp, walk_amp, dsp, None, None), 'freqshift_phase_b10', dsp)

    # Frequency scale (shift looks identical, but scaled horizontally)
    np.random.seed(0)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None, None), 'freqscale_a', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f*1.000, walk_f*1.000, run_amp, walk_amp, dsp, None, None, None), 'freqscale_b0', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f*1.633, walk_f*1.633, run_amp, walk_amp, dsp, None, None, None), 'freqscale_b1', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f*2.266, walk_f*2.266, run_amp, walk_amp, dsp, None, None, None), 'freqscale_b2', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f*2.900, walk_f*2.900, run_amp, walk_amp, dsp, None, None, None), 'freqscale_b3', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f*3.533, walk_f*3.533, run_amp, walk_amp, dsp, None, None, None), 'freqscale_b4', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f*4.166, walk_f*4.166, run_amp, walk_amp, dsp, None, None, None), 'freqscale_b5', dsp)

    save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 'freqscale_phase_a', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f*1.000, walk_f*1.000, run_amp, walk_amp, dsp, None, None), 'freqscale_phase_b0', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f*1.633, walk_f*1.633, run_amp, walk_amp, dsp, None, None), 'freqscale_phase_b1', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f*2.266, walk_f*2.266, run_amp, walk_amp, dsp, None, None), 'freqscale_phase_b2', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f*2.900, walk_f*2.900, run_amp, walk_amp, dsp, None, None), 'freqscale_phase_b3', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f*3.533, walk_f*3.533, run_amp, walk_amp, dsp, None, None), 'freqscale_phase_b4', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f*4.166, walk_f*4.166, run_amp, walk_amp, dsp, None, None), 'freqscale_phase_b5', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f*4.800, walk_f*4.800, run_amp, walk_amp, dsp, None, None), 'freqscale_phase_b6', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f*5.433, walk_f*5.433, run_amp, walk_amp, dsp, None, None), 'freqscale_phase_b7', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f*6.067, walk_f*6.067, run_amp, walk_amp, dsp, None, None), 'freqscale_phase_b8', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f*6.700, walk_f*6.700, run_amp, walk_amp, dsp, None, None), 'freqscale_phase_b9', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f*7.333, walk_f*7.333, run_amp, walk_amp, dsp, None, None), 'freqscale_phase_b10', dsp)

    # Jumping mean
    # np.random.seed(0)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None, None, b=0.0), 'jumpmean_a', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None, None, b=0.0), 'jumpmean_b0', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None, None, b=2.0), 'jumpmean_b1', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None, None, b=4.0), 'jumpmean_b2', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None, None, b=8.0), 'jumpmean_b3', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None, None, b=16.0), 'jumpmean_b4', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None, None, b=32.0), 'jumpmean_b5', dsp)

    # save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None, b=0.0), 'jumpmean_phase_a', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None, b=0.0), 'jumpmean_phase_b0', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None, b=2.0), 'jumpmean_phase_b1', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None, b=4.0), 'jumpmean_phase_b2', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None, b=8.0), 'jumpmean_phase_b3', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None, b=16.0), 'jumpmean_phase_b4', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None, b=32.0), 'jumpmean_phase_b5', dsp)

    #
    # 2D classification problem:
    # Walking (negative) vs. running (positive)
    #
    # Same as 1D but split every-other frequency components/amplitudes across
    # x and y, starting with first frequency for x
    #
    run_f_x = np.array([1, 3, 5], dtype=np.float32)
    run_amp_x = np.array([0.75, 0.38, 0.08], dtype=np.float32)
    run_f_y = np.array([2, 4, 6], dtype=np.float32)
    run_amp_y = np.array([0.56, 0.19, 0.04], dtype=np.float32)

    walk_f_x = np.array([1, 4], dtype=np.float32)
    walk_amp_x = np.array([0.5, 0.06], dtype=np.float32)
    walk_f_y = np.array([2], dtype=np.float32)
    walk_amp_y = np.array([0.25], dtype=np.float32)

    run_f = [run_f_x, run_f_y]
    run_amp = [run_amp_x, run_amp_y]
    walk_f = [walk_f_x, walk_f_y]
    walk_amp = [walk_amp_x, walk_amp_y]

    # Linear transform (random/fixed), rotating from 0 degrees to 180 degrees (e.g. 2D accelerometer)
    np.random.seed(0)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 'rotate_phase_a', dsp)
    save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 0), 'rotate_phase_b0', dsp)
    save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 18), 'rotate_phase_b1', dsp)
    save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 36), 'rotate_phase_b2', dsp)
    save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 54), 'rotate_phase_b3', dsp)
    save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 72), 'rotate_phase_b4', dsp)
    save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 90), 'rotate_phase_b5', dsp)
    save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 108), 'rotate_phase_b6', dsp)
    save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 126), 'rotate_phase_b7', dsp)
    save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 144), 'rotate_phase_b8', dsp)
    save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 162), 'rotate_phase_b9', dsp)
    save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 180), 'rotate_phase_b10', dsp)
