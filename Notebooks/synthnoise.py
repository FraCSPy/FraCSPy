import itertools
import matplotlib.pyplot as plt
import numpy as np
import pylops
import random
from scipy.signal import filtfilt, butter



def add_whitegaussian_noise(d, sc=0.5):
    """ Add white gaussian noise to data patch

    Parameters
    ----------
    d: np.array [y,x]
        Data to add noise to
    sc: float
        noise scaling value

    Returns
    -------
        d+n: np.array
            Created noisy data
        n: np.array
            Additive noise
    """

    n = np.random.normal(size=d.shape)

    return d + (n * sc), n


def add_bandlimited_noise(d, lc=2, hc=80, sc=0.5):
    """ Add bandlimited noise to data patch

    Parameters
    ----------
    d: np.array [y,x]
        Data to add noise to
    lc: float
        Low cut for bandpass
    hc: float
        High cut for bandpass
    sc: float
        Noise scaling value

    Returns
    -------
        d+n: np.array
            Created noisy data
        n: np.array
            Additive noise
    """
    n = band_limited_noise(size=d.shape, lowcut=lc, highcut=hc)

    return d + (n * sc), n


def add_spatiotemporal_noise(d, sc=0.5, npix=5):
    sigman = 1
    nnr = filtfilt(np.ones(npix) / npix, 1,
                   filtfilt(np.ones(npix) / npix,
                            1,
                            np.random.normal(0, sigman, (d.shape[0], d.shape[1])).T,
                            method='gust').T,
                   method='gust')
    n = nnr / np.max(abs(nnr))

    return d + (n * sc), n


def add_asym_spatiotemporal_noise(d, sc=0.5, npix_x=5, npix_y=3):
    sigman = 1
    nnr = filtfilt(np.ones(npix_x) / npix_x, 1,
                   filtfilt(np.ones(npix_y) / npix_y,
                            1,
                            np.random.normal(0, sigman, (d.shape[0], d.shape[1])).T,
                            method='gust').T,
                   method='gust')
    n = nnr / np.max(abs(nnr))

    return d + (n * sc), n


def add_trace_wise_noise(d,
                         num_noisy_traces,
                         noisy_trace_value,
                         num_realisations,
                         additive=True,
                         bandpassed_noise=None,
                         ):
    """ Add trace-wise noise to data patch

    Parameters
    ----------
    d: np.array [shot,y,x]
        Data to add noise to
    num_noisy_traces: int
        Number of noisy traces to add to shots
    noisy_trace_value: int
        Value of noisy traces
    num_realisations: int
        Number of repeated applications per shot

    Returns
    -------
        alldata: np.array
            Created noisy data
    """

    alldata = []
    for k in range(len(d)):
        clean = d[k]
        data = np.ones([num_realisations, d.shape[1], d.shape[2]])
        for i in range(len(data)):
            corr = np.random.randint(0, d.shape[2], num_noisy_traces)
            data[i] = clean.copy()
            noise_rel = np.zeros([data.shape[1],data.shape[2]])
            # print(data.shape)
            # print(noise_rel.shape)

            for c in corr:
                if bandpassed_noise:
                    noise_rel[:, c] = butter_bandpass_filter(np.random.normal(size=len(data[i, :, c])),
                                                                    lowcut=bandpassed_noise[0],
                                                                    highcut=bandpassed_noise[1],
                                                                    fs=500) * noisy_trace_value
                else:
                    noise_rel[:, c] = np.ones([data.shape[1]]) * noisy_trace_value

            if additive:
                data[i] = data[i] + noise_rel
            else:
                data[i,:,corr] = noise_rel[:,corr].T
        alldata.append(data)

    alldata = np.array(alldata)
    alldata = alldata.reshape(num_realisations * d.shape[0], d.shape[1], d.shape[2])
    # print(alldata.shape)

    return alldata


def add_linear_noise(d, x_range, t_range, theta, sc=0.1):
    par = {
        "ox": x_range[0],
        "dx": 1,
        "nx": x_range[1] - x_range[0],
        "oy": -100,
        "dy": 2,
        "ny": 101,
        "ot": 0,
        "dt": 0.004,
        "nt": t_range[1] - t_range[0],
        "f0": 20,
        "nfmax": 210,
    }

    # Create axis
    t, t2, x, y = pylops.utils.seismicevents.makeaxis(par)

    # Create wavelet
    wav = pylops.utils.wavelets.ricker(np.arange(41) * par["dt"], f0=par["f0"])[0]

    v = 100
    t0 = np.arange(-10, t.max(), step=0.1)
    theta = [theta for _ in range(len(t0))]
    amp = [1 for _ in range(len(t0))]

    mlin, n = pylops.utils.seismicevents.linear2d(x, t, v, t0, theta, amp, wav)
    print(n.shape)

    # Add filter over to increase/decrease amplitudes in places
    _, nfilt = add_spatiotemporal_noise(n, sc=0.2, npix=5)
    n = nfilt * n

    # Add noise
    noisydata = d.copy()
    noisydata[t_range[0]:t_range[1], x_range[0]:x_range[1]] = (n.T * sc) + noisydata[t_range[0]:t_range[1],
                                                                           x_range[0]:x_range[1]]

    return noisydata, n


def add_rotated_linear_noise(d, sc=0.1 ):
    nt, nx = d.shape[0], d.shape[1]

    wav, _, wav_c = pylops.utils.wavelets.ricker(0.008 * np.arange(36), f0=25)  # length=69
    ln = np.zeros([nx, nt])
    for i in range(len(ln)):
        ln[i] = np.hstack([np.zeros([64]),
                           wav[:-1],
                           np.zeros([256])])[10 + int(np.floor(i / 4)):int(np.floor(i / 4)) + 266]

    #Make multiple arrivals within window
    si = 160
    dur = 64
    ln_crp = np.fliplr(ln).T[si:si+dur]
    ln_mr = np.tile(ln_crp.T,8).T
    ln_mr_overlap = ln_mr + np.vstack([ln_mr[int(dur/2):,:], np.zeros([int(dur/2),nx])])

    # Add to data
    n_si = np.random.randint(16)
    n = ln_mr_overlap[n_si:n_si+nt]
    noisydata = d + sc*n

    return noisydata, n


def add_rig_noise(d, sc=0.3):
    nt, nx = d.shape[0], d.shape[1]
    
    # Make complex wavelet
    wav, _, wav_c = pylops.utils.wavelets.ricker(0.008*np.arange(36), f0=50)  # length=69
    wav = np.convolve(wav, np.random.normal(0,1, 2500))

    # Get linear arrivals
    mlin, n = pylops.utils.seismicevents.linear2d(x=np.arange(0, nx)*25, 
                                              t=np.arange(0, nt)*0.008, 
                                              v=[1500], t0=[0], theta=[90], amp=[1], wav=wav)
    
    noisydata = d + sc*n.T

    return noisydata, n


def butter_bandpass(lowcut, highcut, fs, order=5):
    """ Bandpass filter

    Parameters
    ----------
    lowcut: int
        Low cut for bandpass
    highcut: int
        High cut for bandpass
    fs: int
        Sampling frequency
    order: int
        Filter order

    Returns
    -------
        b : np.array
            The numerator coefficient vector of the filter
        a : np.array
            The denominator coefficient vector of the filter
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """ Apply bandpass filter to trace

    Parameters
    ----------
    data: np.array [1D]
        Data onto which to apply bp filter
    lowcut: int
        Low cut for bandpass
    highcut: int
        High cut for bandpass
    fs: int
        Sampling frequency
    order: int
        Filter order

    Returns
    -------
        y : np.array
            Bandpassed data
    """

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def array_bp(data, lowcut, highcut, fs, order=5):
    """ Apply bandpass filter to array of traces

    Parameters
    ----------
    data: np.array [2D]
        Data onto which to apply bp filter
    lowcut: int
        Low cut for bandpass
    highcut: int
        High cut for bandpass
    fs: int
        Sampling frequency
    order: int
        Filter order

    Returns
    -------
        bp : np.array [2D]
            Bandpassed data
    """
    bp = np.vstack([butter_bandpass_filter(data[:, ix], lowcut, highcut, fs, order)
                    for ix in range(data.shape[1])])

    return bp


def band_limited_noise(size, lowcut, highcut, fs=250):
    """ Generate bandlimited noise

    Parameters
    ----------
    size: tuple
        Size of array on which to create the noise
    lowcut: int
        Low cut for bandpass
    highcut: int
        High cut for bandpass
    fs: int
        Sampling frequency

    Returns
    -------
        bpnoise : np.array
            Bandpassed noise
    """

    basenoise = np.random.normal(size=size)
    # Pad top and bottom due to filter effects
    basenoise_pad = np.vstack([np.zeros([50, size[1]]), basenoise, np.zeros([50, size[1]])])
    # Bandpass base noise
    bpnoise = array_bp(basenoise_pad, lowcut, highcut, fs, order=5)[:, 50:-50]

    return bpnoise.T






