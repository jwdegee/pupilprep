
import numpy as np
import scipy as sp
import pandas as pd
import mne

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

def interpolate_blinks(raw_et, buffer=0.2):

    # interpolate blinks round 1:
    raw_et = mne.preprocessing.eyetracking.interpolate_blinks(
                    raw_et, buffer=buffer, interpolate_gaze=True)

    # detect additional blinks:
    fs = raw_et.info['sfreq']
    time = raw_et.times
    pupil = raw_et.get_data()[np.array([c == 'pupil' for c in raw_et.get_channel_types()]),:].ravel()
    pupil_diff = np.diff(pupil) * fs
    pupil_diff_z = pupil_diff / np.std(pupil_diff)
    blinks_starts = np.array(pupil_diff_z<-6, dtype=int)
    blinks_ends = np.array(pupil_diff_z>6, dtype=int)
    if sum(blinks_starts)>0:
        blink_start_inds = np.where((np.diff(blinks_starts)==1))[0]
        blink_start_times = time[blink_start_inds]
        blink_start_times_selected = blink_start_times[np.concatenate((np.array([True]),np.diff(blink_start_times)>0.5))]
        blink_end_inds = np.where((np.diff(blinks_ends)==1))[0]
        blink_end_times = time[blink_end_inds]
        blink_end_times_selected = np.zeros(len(blink_start_times_selected))
        for i, b in enumerate(blink_start_times_selected):
            b_end_times = blink_end_times[(blink_end_times>b)&(blink_end_times<(b+1))]
            if len(b_end_times) == 0:
                blink_start_times_selected[i] = blink_start_times_selected[i]-0.1
                blink_end_times_selected[i] = blink_start_times_selected[i] + 0.3
            else:
                blink_end_times_selected[i] = max(b_end_times)
        blink_durations = blink_end_times_selected - blink_start_times_selected

        # add to annotations:
        for onset, dur in zip(blink_start_times_selected, blink_durations):
            raw_et.annotations.append(onset=onset, duration=dur,
                                    description='BAD_blink', 
                                    ch_names=[raw_et.ch_names])
        
        # interpolate blinks round 2:
        raw_et = mne.preprocessing.eyetracking.interpolate_blinks(
            raw_et, buffer=buffer, interpolate_gaze=True)
    
    return raw_et

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

    # blink interpolation:
    et = interpolate_blinks(raw_et, buffer=0.2)

    # get events:
    events = et.annotations.to_data_frame()
    events['onset'] = et.annotations.onset

    # get in right shape:
    df = et.to_data_frame()
    df.columns = [c.split('_')[0]+'_int' for c in df.columns]
    df = df.loc[:,[c for c in df.columns if not 'time' in c]]
    df = pd.concat((df_raw, df), axis=1)

    # don't start or end with NaN
    df.loc[df['pupil_int']==0, 'pupil_int'] = np.NaN
    columns = ['pupil_int', 'xpos_int', 'ypos_int']
    df[columns] = df[columns].ffill(axis=0)
    df[columns] = df[columns].bfill(axis=0)
    
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

    return df, events, fs
