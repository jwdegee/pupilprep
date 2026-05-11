
import numpy as np
import scipy as sp
import pandas as pd
import mne
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from IPython import embed as shell

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(style='ticks', font='Arial', font_scale=1, rc={
    'axes.linewidth': 0.25, 
    'axes.labelsize': 7, 
    'axes.titlesize': 7, 
    'xtick.labelsize': 6, 
    'ytick.labelsize': 6, 
    'legend.fontsize': 6, 
    'xtick.major.width': 0.25, 
    'ytick.major.width': 0.25,
    'text.color': 'Black',
    'axes.labelcolor':'Black',
    'xtick.color':'Black',
    'ytick.color':'Black',} )
sns.plotting_context()
sns.set_palette("tab10")

def _double_gamma(params, x):
    a1 = params['a1']
    sh1 = params['sh1']
    sc1 = params['sc1']
    a2 = params['a2']
    sh2 = params['sh2']
    sc2 = params['sc2']
    return a1 * sp.stats.gamma.pdf(x, sh1, loc=0.0, scale = sc1) + a2 * sp.stats.gamma.pdf(x, sh2, loc=0.0, scale=sc2)

def _butter_lowpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = sp.signal.butter(order, [high], btype='lowpass')
    return b, a

def _butter_lowpass_filter(data, highcut, fs, order=5):
    b, a = _butter_lowpass(highcut, fs, order=order)
    y = sp.signal.filtfilt(b, a, data)
    return y

def _butter_highpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = sp.signal.butter(order, [low], btype='highpass')
    return b, a

def _butter_highpass_filter(data, lowcut, fs, order=5):
    b, a = _butter_highpass(lowcut, fs, order=order)
    y = sp.signal.filtfilt(b, a, data)
    return y

def diff(series):
    """
    Python implementation of matlab's diff function
    """
    return series[1:] - series[:-1]

def smooth(x, window_len):
    """
    Python implementation of matlab's smooth function
    """

    if window_len < 3:
        return x

    # Window length must be odd
    if window_len%2 == 0:
        window_len += 1

    window_len = int(window_len)
    w = np.ones(window_len)
    y = np.convolve(w, x, mode='valid') / len(w)
    y = np.hstack((x[:window_len//2], y, x[len(x)-window_len//2:]))

    for i in range(0, window_len//2):
        y[i] = np.sum(y[0 : i+i]) / ((2*i) + 1)

    for i in range(len(x)-window_len//2, len(x)):
        y[i] = np.sum(y[i - (len(x) - i - 1) : i + (len(x) - i - 1)]) / ((2*(len(x) - i - 1)) + 1)

    return y

def detect_blinks_pupil_noise(pupil, fs):
    """
    Function to find blinks and return blink onset and offset indices
    Adapted from: R. Hershman, A. Henik, and N. Cohen, “A novel blink detection method based on pupillometry noise,” Behav. Res. Methods, vol. 50, no. 1, pp. 107–114, 2018.

    Input:
        pupil               : [numpy array/list] of pupil size data for left/right eye
        sampling_freq       : [float] sampling frequency of eye tracking hardware (default = 1000 Hz)
        concat              : [boolean] concatenate close blinks/missing trials or not. See R. Hershman et. al. for more information
        concat_gap_interval : [float] interval between successive missing samples/blinks to concatenate
    Output:
        blinks              : [dictionary] {"blink_onset", "blink_offset"} containing numpy array/list of blink onset and offset indices
    """
    sampling_interval = 1000 // fs
    concat_gap_interval = 100

    blink_onset = []
    blink_offset = []
    blinks = {"blink_onset": blink_onset, "blink_offset": blink_offset}

    pupil = np.asarray(pupil)
    missing_data = np.array(pupil == 0, dtype="float32")
    difference = diff(missing_data)

    blink_onset = np.where(difference == 1)[0]
    blink_offset = np.where(difference == -1)[0] + 1

    length_blinks = len(blink_offset) + len(blink_onset)


    # Edge Case 1: there are no blinks
    if (length_blinks == 0):
        return blinks


    # Edge Case 2: the data starts with a blink. In this case, blink onset will be defined as the first missing value.
    """
        Two possible situations may cause this:
            i.  starts with a blink but does not end with a blink ---> len(blink_onset) < len(blink_offset)
            ii. starts with a blink and ends with a blink          ---> len(blink_onset) == len(blink_offset) and (blink_onset[0] == blink_offset[0])
    """
    if ((len(blink_onset) < len(blink_offset)) or ((len(blink_onset) == len(blink_offset)) and (blink_onset[0] > blink_offset[0]))) and pupil[0] == 0:
        blink_onset = np.hstack((0, blink_onset))


    # Edge Case 3: the data ends with a blink. In this case, blink offset will be defined as the last missing sample
    """
        Two possible situations may cause this:
            i.  ends with a blink but does not start with a blink ---> len(blink_offset) < len(blink_onset)
            ii. ends with a blink and starts with a blink          ---> Already handled "start with blink" in Edge case 2 so it reduces to i (previous case)
    """
    if (len(blink_offset) < len(blink_onset)) and pupil[-1] == 0:
        blink_offset = np.hstack((blink_offset, len(pupil) - 1))

    # Smoothing the data in order to increase the difference between the measurement noise and the eyelid signal.
    ms_4_smoothing = 10
    samples2smooth = ms_4_smoothing // sampling_interval
    smooth_pupil = np.array(smooth(pupil, samples2smooth), dtype='float32')

    smooth_pupil[np.where(smooth_pupil == 0)[0]] = float('nan')
    smooth_pupil_diff = diff(smooth_pupil)

    """
    Finding values <=0 and >=0 in order to find monotonically increasing and decreasing sections of smoothened pupil data

            Eg. a =     [2, 1, 2, 8, 7, 6, 5, 4, 4, 0, 0, 0, 0, 0, 3, 3, 3, 8, 9, 10, 2, 3, 10]
                                  ----------------  S           E  =================
            diff(a)=   [-1  1  6 -1 -1 -1 -1  0 -4  0  0  0  0  3  0  0  5  1  1  -8  1  7]

    monotonically_dec = [T  F  F  T  T  T  T  T  T  T  T  T  T  F  T  T  F  F  F   T  F  F]   (T=True, F=False)
    monotonically_dec = [F  T  T  F  F  F  F  T  F  T  T  T  T  T  T  T  T  T  T   F  T  T]

    ---> The monotonically decreasing sequence before the blink is underlined with -- and the monotonically increasing sequence after the blink with ==
    ---> S : denotes the initially detected onset of blink
    ---> E : denotes the initially detected offset of blink

    >> Looking at diff(a), all values in the montonically decreasing sequence should be <= 0 and those included in the monotonically increasing sequence >= 0
    >> Hence, by moving left from the initially detected onset while T(True) values are encountered in monotonically_dec we can update the onset to the start of monotonically_dec seq
    >> By moving right from the initially detected offset while T(True) values are encountered in monotonically_inc we can update the offset to the end of monotonically_inc seq + 1
    """
    monotonically_dec = smooth_pupil_diff <= 0
    monotonically_inc = smooth_pupil_diff >= 0

    # Finding correct blink onsets and offsets using monotonically increasing and decreasing arrays
    for i in range(len(blink_onset)):
        # Edge Case 2: If data starts with blink we do not update it and let starting blink index be 0
        if blink_onset[i] != 0:
            j = blink_onset[i] - 1
            while j > 0 and monotonically_dec[j] == True:
                j -= 1
            blink_onset[i] = j + 1

        # Edge Case 3: If data ends with blink we do not update it and let ending blink index be the last index of the data
        if blink_offset[i] != len(pupil) - 1:
            j = blink_offset[i]
            while j < len(monotonically_inc) and monotonically_inc[j] == True:
                j += 1
            blink_offset[i] = j

    # Removing duplications (in case of consecutive sets): [a, b, b, c] => [a, c] or if inter blink interval is less than concat_gap_interval
    c = np.empty((len(blink_onset) + len(blink_offset),), dtype=blink_onset.dtype)
    c[0::2] = blink_onset
    c[1::2] = blink_offset
    c = list(c)

    i = 1
    while i<len(c)-1:
        if c[i+1] - c[i] <= concat_gap_interval:
            c[i:i+2] = []
        else:
            i += 2

    temp = np.reshape(c, (-1, 2), order='C')

    """
    Multplied by sampling interval in order to give onset and offset in real time (milliseconds) by factoring in sampling rate of device used
    '+ sampling_interval' because the output should be in real time and as python indexing starts at 0 instead of 1, this is the standardising factor
    NOTE:edit the lines below to only temp[:, 0] and temp[:, 1] in case you are interested in the indices of blinks and not realtime values
    """
    #blinks["blink_onset"] = (temp[:, 0] * sampling_interval) + sampling_interval
    #blinks["blink_offset"] = (temp[:, 1] * sampling_interval) + sampling_interval

    # RONY 2023-01-19: WE FOLLOW THE "NOTE" ORIGINALLY WRITTEN, AND RETURN THE **INDICES** BASED ON THE RECOMMENDATION:
    blinks["blink_onset"] = temp[:,0]
    blinks["blink_offset"] = temp[:,1]

    return temp[:,0], temp[:,1]

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def detect_blinks_pupil_slope(pupil, fs, concat_gap_interval=0.5, z_threshold=10):

    concat_gap_interval = int(concat_gap_interval*fs)

    pupil = moving_average(pupil, int(0.1*fs))
    pupil_diff = np.diff(pupil)
    pupil_diff[0:concat_gap_interval] = np.median(pupil_diff)
    pupil_diff[-concat_gap_interval:] = np.median(pupil_diff)
    pupil_diff_z = (pupil_diff-np.mean(pupil_diff)) / np.std(pupil_diff)

    # find threshold crossings:
    ind = np.diff(np.concatenate((np.array([False]), pupil_diff_z <= -z_threshold), dtype=int))
    blink_onsets = np.where(ind == 1)[0]
    ind = np.diff(np.concatenate((np.array([False]), pupil_diff_z >= z_threshold), dtype=int))
    blink_offsets = np.where(ind == -1)[0] + 1

    # remove ones that are close together:
    if len(blink_onsets)>0:
        blink_onsets = blink_onsets[np.concatenate((np.array([True]), np.diff(blink_onsets)>(0.500*fs)))]
    if len(blink_offsets)>0:
        blink_offsets = blink_offsets[np.concatenate((np.array([True]), np.diff(blink_offsets)>(0.500*fs)))]
    
    # only keep sets:
    ind_onsets = np.zeros(len(blink_onsets), dtype=bool)
    ind_offsets = np.zeros(len(blink_offsets), dtype=bool)
    for i, onset in enumerate(blink_onsets):
        for offset in blink_offsets:
            if ((offset-onset)>0)&((offset-onset)<(0.500*fs)):
                ind_onsets[i] = True
    for i, offset in enumerate(blink_offsets):
        for onset in blink_onsets:
            if ((offset-onset)>0)&((offset-onset)<(0.500*fs)):
                ind_offsets[i] = True
    blink_onsets = blink_onsets[ind_onsets]
    blink_offsets = blink_offsets[ind_offsets]

    # Removing duplications (in case of consecutive sets): [a, b, b, c] => [a, c] or if inter blink interval is less than concat_gap_interval
    c = np.empty((len(blink_onsets) + len(blink_offsets),), dtype=blink_onsets.dtype)
    c[0::2] = blink_onsets
    c[1::2] = blink_offsets
    c = list(c)

    i = 1
    while i<len(c)-1:
        if c[i+1] - c[i] <= concat_gap_interval:
            c[i:i+2] = []
        else:
            i += 2

    temp = np.reshape(c, (-1, 2), order='C')

    return temp[:,0], temp[:,1]

def set_custon_blink_annotations(et, onsets, offsets):

    # load data:
    time = et.times

    # delete existing 'BAD_blink' annotations:
    et.annotations.delete(np.where(et.annotations.description == 'BAD_blink'))

    # convert indices to times:
    onsets = time[onsets]
    offsets = time[offsets]

    # add to annotations:
    for onset, offset in zip(onsets, offsets):
        et.annotations.append(onset=onset, duration=offset-onset,
                                description='BAD_blink', 
                                ch_names=[et.ch_names])
    
    return et

def regress_xy(df):

    # combine regressors:
    regs = []
    regs_titles = []
    regs.append(df['xpos_int'].values)
    regs_titles.append('x')
    regs.append(df['ypos_int'].values)
    regs_titles.append('y')
    print([r.shape for r in regs])

    # GLM:
    design_matrix = np.matrix(np.vstack([reg for reg in regs])).T
    betas = np.array(((design_matrix.T * design_matrix).I * design_matrix.T) * np.matrix(df['pupil_int'].values).T).ravel()
    explained = np.sum(np.vstack([betas[i]*regs[i] for i in range(len(betas))]), axis=0)
    rsq = sp.stats.pearsonr(df['pupil_int'].values, explained)[0]**2
    print('explained variance = {}%'.format(round(rsq*100,2)))

    # cleaned-up time series:
    df['pupil_int'] = (df['pupil_int'] - explained) + df['pupil_int'].mean()

def regress_blinks(df, events, interval=7, regress_blinks=True, regress_sacs=True, fs=1000):

    ''' 
    This function results from Knapen et al. (2016). There, pupil responses to blinks were extracted 
    from the pupil signal using least squares deconvolution and fitting a (double (for blinks)) gamma density functions. So here, 
    a gamma density function is created with the estimates from that paper which is then used as a kernel to convolve
    with a matrix in which the time points of blink ends and saccade ends are described. The result is used as a regressor
    to be applied to the pupil data of the according times.

    Alternatively, it would also be possible to estimate the pupil response in the current data set first and then use the resulting values in this function.
    '''
    
    # only regress out blinks within these limits:
    early_cutoff = 25
    late_cutoff = interval

    # params:
    x = np.linspace(0, interval, int(interval * fs), endpoint=False)
    standard_blink_parameters = {'a1':-0.604, 'sh1':8.337, 'sc1':0.115, 'a2':0.419, 'sh2':15.433, 'sc2':0.178}
    blink_kernel = _double_gamma(standard_blink_parameters, x)
    standard_sac_parameters = {'a1':-0.175, 'sh1': 6.451, 'sc1':0.178, 'a2':0.0, 'sh2': 1, 'sc2': 1}
    sac_kernel = _double_gamma(standard_sac_parameters, x)

    # create blink regressor:
    blink_ends = (events.loc[events['description']=='blink', 'onset'].values +
                  events.loc[events['description']=='blink', 'duration'].values)
    blink_ends = blink_ends[(blink_ends > early_cutoff) & (blink_ends < (df['time'].iloc[-1]-late_cutoff))]
    if blink_ends.size == 0:
        blink_ends = np.array([0], dtype=int)
    else:
        blink_ends = blink_ends.astype(int)
    blink_ends_ind = np.array(df['time'].searchsorted(blink_ends).ravel())
    blink_reg = np.zeros(df.shape[0])
    blink_reg[blink_ends_ind] = 1
    blink_reg_conv = sp.signal.fftconvolve(blink_reg, blink_kernel, 'full')[:-(len(blink_kernel)-1)] #fftconvolve uses fast fourier transformation for a fast convolution

    # create saccade regressor:
    sac_ends = (events.loc[events['description']=='saccade', 'onset'].values +
                events.loc[events['description']=='saccade', 'duration'].values)
    sac_ends = sac_ends[(sac_ends > early_cutoff) & (sac_ends < (df['time'].iloc[-1]-late_cutoff))]
    if sac_ends.size == 0:
        sac_ends = np.array([0], dtype=int)
    else:
        sac_ends = sac_ends.astype(int)
    sac_ends_ind = np.array(df['time'].searchsorted(sac_ends).ravel())
    sac_reg = np.zeros(df.shape[0])
    sac_reg[sac_ends_ind] = 1
    sac_reg_conv = sp.signal.fftconvolve(sac_reg, sac_kernel, 'full')[:-(len(sac_kernel)-1)]

    # combine regressors:
    regs = []
    regs_titles = []
    if regress_blinks:
        regs.append(blink_reg_conv)
        regs_titles.append('blink')
    if regress_sacs:
        regs.append(sac_reg_conv)
        regs_titles.append('saccade')
    print([r.shape for r in regs])

    # GLM:
    design_matrix = np.matrix(np.vstack([reg for reg in regs])).T
    betas = np.array(((design_matrix.T * design_matrix).I * design_matrix.T) * np.matrix(df['pupil_int_bp'].values).T).ravel()
    explained = np.sum(np.vstack([betas[i]*regs[i] for i in range(len(betas))]), axis=0)
    rsq = sp.stats.pearsonr(df['pupil_int_bp'].values, explained)[0]**2
    print('explained variance = {}%'.format(round(rsq*100,2)))

    # cleaned-up time series:
    df['pupil_int_bp_clean'] = df['pupil_int_bp'] - explained
    df['pupil_int_lp_clean'] = df['pupil_int_bp_clean'] + (df['pupil_int_lp']-df['pupil_int_bp'])

def temporal_filter(df, measure, fs=15, hp=0.01, lp=6.0, order=3):
    df['{}_lp'.format(measure)] = _butter_lowpass_filter(data=df[measure], highcut=lp, fs=fs, order=order)
    df['{}_bp'.format(measure)] = _butter_highpass_filter(data=df[measure], lowcut=hp, fs=fs, order=order) - (df[measure] - df['{}_lp'.format(measure)])

def psc(df, measure):
    df['{}_psc'.format(measure)] = (df[measure] - df[measure].median()) / df[measure].median() * 100

def fraction(df, measure):
    df['{}_frac'.format(measure)] = df[measure] / np.percentile(df[measure], 99.5)

def slope(df, measure, hp=2.0, fs=15, order=3):
    slope = np.concatenate((np.array([0]), np.diff(df[measure]))) * fs
    slope = _butter_lowpass_filter(slope, highcut=hp, fs=fs, order=order)
    df['{}_slope'.format(measure)] = slope

def preprocess_pupil(filename, params):
    
    # load pupil data:
    raw_et = mne.io.read_raw_eyelink(filename)
    df_raw = raw_et.to_data_frame()
    df_raw.columns = [c.split('_')[0] for c in df_raw.columns]
    fs = raw_et.info['sfreq']

    # custom blink detection based on pupil noise:
    blink_detection_noise = False
    if blink_detection_noise:
        pupil = raw_et.get_data()[np.array([c == 'pupil' for c in raw_et.get_channel_types()]),:].ravel()
        onsets, offsets = detect_blinks_pupil_noise(pupil, fs)
        if len(onsets) > 0:
            raw_et = set_custon_blink_annotations(raw_et, onsets, offsets)

    # interpolate blinks:
    print('interpolating {} Eyelink blinks'.format((raw_et.annotations.description == 'BAD_blink').sum()))
    et = mne.preprocessing.eyetracking.interpolate_blinks(
                    raw_et, buffer=(0.2, 0.2), interpolate_gaze=True)

    # custom blink detection based on pupil slope:
    blink_detection_slope = True
    if blink_detection_slope:
        pupil = et.get_data()[np.array([c == 'pupil' for c in raw_et.get_channel_types()]),:].ravel()
        onsets_slope, offsets_slope = detect_blinks_pupil_slope(pupil, fs, z_threshold=params['slope_z_threshold'])
        if len(onsets_slope) > 0:
            et = set_custon_blink_annotations(et, onsets_slope, offsets_slope)

    # interpolate blinks:
    print('interpolating {} additional blinks'.format((et.annotations.description == 'BAD_blink').sum()))
    et = mne.preprocessing.eyetracking.interpolate_blinks(
                    et, buffer=(0.2, 0.2), interpolate_gaze=True)

    # get events:
    events = et.annotations.to_data_frame()
    events['onset'] = et.annotations.onset

    # get in right shape:
    df = et.to_data_frame()
    df.columns = [c.split('_')[0]+'_int' for c in df.columns]
    df = df.loc[:,[c for c in df.columns if not 'time' in c]]
    df = pd.concat((df_raw, df), axis=1)

    # regress xy:
    if params['regress_xy']:
        regress_xy(df=df)
    
    # temporal filter:
    temporal_filter(df=df, measure='pupil_int', 
                    hp=params['hp'], lp=params['lp'], 
                    order=params['order'], fs=fs)
    
    # regress out pupil responses to blinks and saccades:
    regress_blinks(df=df, events=events, interval=7,
                   regress_blinks=params['regress_blinks'],
                   regress_sacs=params['regress_sacs'], fs=fs)

    # percent signal change:
    psc(df=df, measure='pupil_int_lp_clean')
    psc(df=df, measure='pupil_int_lp')
    psc(df=df, measure='pupil_int')

    # figures:
    fig1 = plt.figure(figsize=(8,4))
    ax = fig1.add_subplot(211)
    plt.plot(df['time'], df['pupil'])
    plt.plot(df['time'], df['pupil_int'])
    plt.plot(df['time'], df['pupil_int_lp'])
    blinks = events.loc[(events['description']=='BAD_blink'), 'onset'].values
    for b in blinks:
        plt.axvspan(b-0.05, b+0.1, color='r', alpha=0.2)
    ax = fig1.add_subplot(212)
    plt.plot(df['time'], df['pupil_int_lp_psc'])
    plt.tight_layout()
    sns.despine()

    fig2 = plt.figure(figsize=(12,12))
    if len(onsets_slope) > 0:
        for i in range(16):
            ax = fig2.add_subplot(4,4,i+1)
            try:
                plt.plot(df['time'].iloc[onsets_slope[i]-400:offsets_slope[i]+400],
                        pupil[onsets_slope[i]-400:offsets_slope[i]+400])
                plt.plot(df['time'].iloc[onsets_slope[i]-400:offsets_slope[i]+400],
                        df['pupil_int'].iloc[onsets_slope[i]-400:offsets_slope[i]+400])
            except:
                pass
        plt.tight_layout()
        sns.despine()
    
    return df, events, fs, [fig1, fig2]

def preprocess_pupillabs(df, df_blinks, params):
    
    # load pupil data:
    fs = int(1/df['time'].diff().median())

    # interpolate blinks:
    for measure in ['pupil_left', 'pupil_right', 'eyelid_left', 'eyelid_right']:
        df['{}_int'.format(measure)] = df[measure]
        for i in range(df_blinks.shape[0]):
            blink_start = df_blinks['time_start'].iloc[i] - 0.1
            blink_end = df_blinks['time_end'].iloc[i] + 0.1
            ind = (df['time']>blink_start)&(df['time']<blink_end)
            df.loc[ind,'{}_int'.format(measure)] = np.linspace(df.loc[ind,measure].iloc[0], 
                                                    df.loc[ind,measure].iloc[-1], sum(ind))

    # # custom blink detection based on pupil slope:
    # blink_detection_slope = True
    # if blink_detection_slope:
    #     pupil = df['pupil_left_int'].values
    #     onsets_slope, offsets_slope = detect_blinks_pupil_slope(pupil, fs, z_threshold=params['slope_z_threshold'])
    #     if len(onsets_slope) > 0:
    #         et = set_custon_blink_annotations(et, onsets_slope, offsets_slope)

    # regress xy:
    if params['regress_xy']:
        regress_xy(df=df)
    
    # temporal filter:
    for measure in ['pupil_left_int', 'pupil_right_int', 'eyelid_left_int', 'eyelid_right_int']:
        temporal_filter(df=df, measure=measure, 
                        hp=params['hp'], lp=params['lp'], 
                        order=params['order'], fs=fs)

    # # regress out pupil responses to blinks and saccades:
    # regress_blinks(df=df, events=events, interval=7,
    #                regress_blinks=params['regress_blinks'],
    #                regress_sacs=params['regress_sacs'], fs=fs)

    # percent signal change:
    # psc(df=df, measure='pupil_int_lp_clean')
    for measure in ['pupil_left_int', 'pupil_right_int', 'eyelid_left_int', 'eyelid_right_int',
                    'pupil_left_int_lp', 'pupil_right_int_lp', 'eyelid_left_int_lp', 'eyelid_right_int_lp']:
        psc(df=df, measure=measure)

    # snr:
    left_snr = df['pupil_left_int'].mean()/df['pupil_left_int'].std()
    right_snr = df['pupil_right_int'].mean()/df['pupil_right_int'].std()

    # figures:
    fig1 = plt.figure(figsize=(12,4))

    ax = fig1.add_subplot(221)
    plt.plot(df['time'], df['pupil_left'])
    plt.plot(df['time'], df['pupil_left_int'])
    plt.plot(df['time'], df['pupil_left_int_lp'])
    plt.title('SNR={}'.format(round(left_snr, 2)))
    for i in range(df_blinks.shape[0]):
        plt.axvspan(df_blinks['time_start'].iloc[i], df_blinks['time_end'].iloc[i], color='r', alpha=0.2)
    ax = fig1.add_subplot(222)
    plt.plot(df['time'], df['pupil_left_int_lp_psc'])
    ax = fig1.add_subplot(223)
    plt.plot(df['time'], df['eyelid_left'])
    plt.plot(df['time'], df['eyelid_left_int'])
    plt.plot(df['time'], df['eyelid_left_int_lp'])
    # plt.title('SNR={}'.format(round(left_snr, 2)))
    for i in range(df_blinks.shape[0]):
        plt.axvspan(df_blinks['time_start'].iloc[i], df_blinks['time_end'].iloc[i], color='r', alpha=0.2)
    ax = fig1.add_subplot(224)
    plt.plot(df['time'], df['eyelid_left_int_lp_psc'])
    plt.tight_layout()
    sns.despine()

    fig2 = plt.figure(figsize=(12,4))
    ax = fig2.add_subplot(221)
    plt.plot(df['time'], df['pupil_right'])
    plt.plot(df['time'], df['pupil_right_int'])
    plt.plot(df['time'], df['pupil_right_int_lp'])
    plt.title('SNR={}'.format(round(right_snr, 2)))
    for i in range(df_blinks.shape[0]):
        plt.axvspan(df_blinks['time_start'].iloc[i], df_blinks['time_end'].iloc[i], color='r', alpha=0.2)
    ax = fig2.add_subplot(222)
    plt.plot(df['time'], df['pupil_right_int_lp_psc'])
    ax = fig2.add_subplot(223)
    plt.plot(df['time'], df['eyelid_right'])
    plt.plot(df['time'], df['eyelid_right_int'])
    plt.plot(df['time'], df['eyelid_right_int_lp'])
    # plt.title('SNR={}'.format(round(left_snr, 2)))
    for i in range(df_blinks.shape[0]):
        plt.axvspan(df_blinks['time_start'].iloc[i], df_blinks['time_end'].iloc[i], color='r', alpha=0.2)
    ax = fig2.add_subplot(224)
    plt.plot(df['time'], df['eyelid_right_int_lp_psc'])
    plt.tight_layout()
    sns.despine()
    
    return df, fs, [fig1, fig2]
