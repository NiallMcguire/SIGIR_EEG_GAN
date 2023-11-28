import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from numpy import zeros, floor, log10, log, mean, array, sqrt, vstack, cumsum, ones, log2, std
from scipy.signal import butter, lfilter
from hurst import compute_Hc, random_walk
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
import math
from itertools import zip_longest
import pywt


class FeatureExtraction:

    def __init__(self):
        pass

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def bandpower(self, data, sf, band, window_sec=None, relative=False):
        """Compute the average power of the signal x in a specific frequency band.
        Parameters
        ----------
        data : 1d-array
            Input signal in the time-domain.
        sf : float
            Sampling frequency of the data.
        band : list
            Lower and upper frequencies of the band of interest.
        window_sec : float
            Length of each window in seconds.
            If None, window_sec = (1 / min(band)) * 2
        relative : boolean
            If True, return the relative power (= divided by the total power of the signal).
            If False (default), return the absolute power.
        Return
        ------
        bp : float
            Absolute or relative band power."""
        from scipy.signal import welch
        from scipy.integrate import simps
        band = np.asarray(band)
        low, high = band

        # Define window length
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = (2 / low) * sf

        # Compute the modified periodogram (Welch)
        freqs, psd = welch(data, sf, nperseg=nperseg)

        # Frequency resolution
        freq_res = freqs[1] - freqs[0]

        # Find closest indices of band in frequency vector
        idx_band = np.logical_and(freqs >= low, freqs <= high)

        # Integral approximation of the spectrum using Simpson's rule.
        bp = simps(psd[idx_band], dx=freq_res)

        if relative:
            bp /= simps(psd, dx=freq_res)
        return bp

    """Three morphological features were extracted to describe morphological characteristics of a single-channel signal
        Kunjira. K """

    # Curve length
    def curve_lenght(self, data):

        scurve = pd.Series(data)
        total_ab = (abs(scurve.diff())).sum()
        Clenght = np.divide(total_ab, (len(scurve) - 1))
        return Clenght

    # Number of peaks
    def no_peaks(self, data):
        speaks = pd.Series(data)
        Npeaks = np.divide((pow(speaks, 2) - (speaks.shift(-1) * speaks.shift(1))).sum(), (len(speaks) - 2))
        return Npeaks

    # Average nonlinear energy
    def average_nonlinear_energy(self, data):
        sav = pd.Series(data)
        AvNE = np.divide(((np.maximum(0, np.sign(sav.shift(2) - sav.shift(1)) - np.sign(sav.shift(1) - sav))).sum()), 2)
        return AvNE

    # Approximate Entropy
    def ApEn(self, U, m, r):
        U = np.array(U)
        N = U.shape[0]

        def _phi(m):
            z = N - m + 1.0
            x = np.array([U[i:i + m] for i in range(int(z))])
            X = np.repeat(x[:, np.newaxis], 1, axis=2)
            C = np.sum(np.absolute(x - X).max(axis=2) <= r, axis=0) / z
            return np.log(C).sum() / z

        return abs(_phi(m + 1) - _phi(m))

    """"
        A four-level discrete wavelet transform (DWT) decomposition was applied
        delta (0-4 Hz)
        theta (4-8 Hz)
        alpha (8-16 Hz)
        beta (16-32 Hz)
        gamma (32-64 Hz)
        To further decrease feature dimensionality, a measure of wavelet
        coefficients called wavelet entropy (WE) are employed.
    """""
    """"Ref :Gregory R. Lee, Ralf Gommers, Filip Wasilewski, Kai Wohlfahrt, Aaron O’Leary (2019).
        PyWavelets: A Python package for wavelet analysis.
        Journal of Open Source Software, 4(36), 1237, https://doi.org/10.21105/joss.01237.
    """""

    for family in pywt.families():
        print("%s family: " % family + ', '.join(pywt.wavelist(family)))

    def DWT(self, x):
        coeffs = pywt.wavedec(x, 'db5', mode='sym', level=4)  # in this experiment, we do a four-level DWT decomposition
        cA4, cD4, cD3, cD2, cD1 = coeffs
        y = pywt.waverec(coeffs, 'db5', mode='sym')
        return coeffs

    def WE(self, coeffs):
        """wavelet entropy (WE) is employed to indicate the degree of multifrequency signal order/disorder in the signals
        Kunjira K."""
        Ej = []
        for c in coeffs:
            for i in c:
                sm = sum(pow(i, 2) for i in c)
            Ej.append(sm)

        Etot = sum(Ej)
        pj = np.divide(Ej, Etot)

        ln_pj = []
        for i in pj:
            ln = np.array(math.log(i))
            ln_pj.append(ln)

        lnpj = np.array(ln_pj)
        WE = -((np.multiply(pj, lnpj)).sum())
        return WE

    def create_feature_set(self, start, end, df):

        rslt_df = df.loc[(df['time']) > start]
        rslt_df2 = rslt_df.loc[(df['time']) < end]

        Track_Features = {}

        # Defining parameterv
        sf = 500  # sampling frequency per second, TODO: adjust it if your data have a differnt frequency (Hz)
        window_sec = 0.79  # secs
        sampls = sf * window_sec  # number of samples that we are going to use in each window. It is a sliging window side
        sampls_sti = rslt_df2.shape[0]  # samples in each stimuli interval;song (sample from start time to end time)

        n_window = sampls_sti / sampls  # number of window
        sampls_window = sampls_sti / n_window  # samples in each sliding window for feature extraction

        C = 41  # Number of EEG channels

        all_slide = []
        Window_Time_Stamp = []
        for i in range(0, int(n_window), 1):
            slide = rslt_df2.iloc[0 + (i * int(sampls_window)):int(sampls_window) + (i * int(sampls_window)),
                    1:C + 1]  # sliding window side is calculated form sf*window_sec and shift sf*window_sec sample (no overlapping)
            Window_Start = ((0 + (i * int(sampls_window))) + (start * sf)) / sf
            # print("Window Start :", Window_Start)
            Window_Time_Stamp.append(Window_Start)
            all_slide.append(slide)


        # ***************************************************************************************#
        # extract AR coefficient using p=2, got 2 feature from 64 channels
        '''
        AR = []
        for index, item in enumerate(all_slide):  # calculate for each time slices
            data5 = pd.DataFrame(item)
            AR_Ch = []
            for name, values in data5.iteritems():
                model = AutoReg(values, 2)
                result = model.fit()
                AR_Ch.append(result.params)
            AR.append(AR_Ch)
        ar = np.array(AR)

        ar_copy = ar
        x = np.delete(ar_copy, 0, axis=2)
        AR_6 = x.reshape(int(n_window),128) # A side of sliding window is 500 and shift 500
        AR_6 = AR_6.tolist()
        '''

        # ***************************************************************************************#
        # alpha [0.5-4], beta [4-8], theta [8-13], delta [13-30],gamma [30-40]

        '''
        lowcut5 = [0.5, 4, 8, 13, 30]
        highcut5 = [4, 8, 13, 30, 40]

        all_band5 = []
        for i, j in zip_longest(lowcut5, highcut5):
            band5 = [i, j]
            BPW5_ = []
            for index, item in enumerate(all_slide):  # calculate for each time slices
                data1 = pd.DataFrame(item)
                BPW5_Ch = []
                for name, values in data1.iteritems():  # step through columns
                    BPW5_Ch.append(self.bandpower(values, sf, band5, window_sec))
                BPW5_.append(BPW5_Ch)
            all_band5.append(BPW5_)

        # ***************************************************************************************#
        # Extract Approximate Entropy,  Hurst Exponent, Mean, Standard deviation, Skewness, Kurtosis features

        Hurst = []
        ApEntropy = []
        Mean = []
        SD = []
        Skew = []
        Kur = []
        Curve = []
        Peaks = []
        AveEn = []

        for index, item in enumerate(all_slide):  # calculate for each time slices
            data2 = pd.DataFrame(item)

            ApEntropy4BL_Ch = []
            Hurst4BL_Ch = []
            Mean4BL_Ch = []
            SD4BL_Ch = []
            Skew4BL_Ch = []
            Kur4BL_Ch = []
            Curve4BL_Ch = []
            Peaks4BL_Ch = []
            AveEn4BL_Ch = []

            for name, values in data2.iteritems():
                ApEntropy4BL_Ch.append(self.ApEn(values, 2, 3))  # m=2,r=3

                series = values
                H, c, data = compute_Hc(series, simplified=False)
                Hurst4BL_Ch.append(H)

                Mean4BL_Ch.append(mean(values))
                SD4BL_Ch.append(std(values))
                Skew4BL_Ch.append(skew(values))
                Kur4BL_Ch.append(kurtosis(values))
                Curve4BL_Ch.append(self.curve_lenght(values))
                Peaks4BL_Ch.append(self.no_peaks(values))
                AveEn4BL_Ch.append(self.average_nonlinear_energy(values))

            ApEntropy.append(ApEntropy4BL_Ch)
            Hurst.append(Hurst4BL_Ch)
            Mean.append(Mean4BL_Ch)
            SD.append(SD4BL_Ch)
            Skew.append(Skew4BL_Ch)
            Kur.append(Kur4BL_Ch)
            Curve.append(Curve4BL_Ch)
            Peaks.append(Peaks4BL_Ch)
            AveEn.append(AveEn4BL_Ch)
        
        '''

        # ***************************************************************************************#

        # Based on Wang S. paper for Statistical and Morphological calculations,
        # we extract features from four frequency bands:
        # theta (4–8 Hz),alpha (8–13 Hz),beta (13–25 Hz), and low gamma (25–40 Hz)

        lowcut = [4, 8, 13, 25]
        highcut = [8, 13, 25, 40]

        data_bandpass = []
        for i, j in zip_longest(lowcut, highcut):
            lohi = []
            for index, item in enumerate(all_slide):  # calculate for each time slices
                values = item
                y = self.butter_bandpass_filter(values, i, j, sf, order=6)
                lohi.append(y)
            data_bandpass.append(lohi)

        # ***************************************************************************************#
        # Extract Approximate Entropy, Hurst Exponent, Mean, Standard deviation, Skewness, Kurtosis features

        ApEntropy_4band = []
        Hurst_4band = []
        Mean_4band = []
        # Var_4band = []
        SD_4band = []
        Skew_4band = []
        Kur_4band = []
        Curve_4band = []
        Peaks_4band = []
        AveEn_4band = []
        for index, item in enumerate(data_bandpass):  # 4 psd
            data_sliding = item

            # calculate features
            ApEntropy_ = []
            #Hurst_ = []
            Mean_ = []
            # Var_ = []
            SD_ = []
            Skew_ = []
            Kur_ = []
            Curve_ = []
            Peaks_ = []
            AveEn_ = []

            for index, item in enumerate(data_sliding):  # 147 sliding
                data = pd.DataFrame(item)

                ApEntropy_Ch = []
                #Hurst_Ch = []
                Mean_Ch = []
                # Var_Ch = []
                SD_Ch = []
                Skew_Ch = []
                Kur_Ch = []
                Curve_Ch = []
                Peaks_Ch = []
                AveEn_Ch = []

                for name, values in data.iteritems():  # step through columns

                    ApEntropy_Ch.append(self.ApEn(values, 2, 3))  # m=2,r=3

                    series = values
                    #H, c, data = compute_Hc(series, simplified=True)
                    #Hurst_Ch.append(H)

                    Mean_Ch.append(mean(values))
                    # Var_Ch.append(item.var(axis=0))
                    SD_Ch.append(std(values))
                    Skew_Ch.append(skew(values))
                    Kur_Ch.append(kurtosis(values))
                    Curve_Ch.append(self.curve_lenght(values))
                    Peaks_Ch.append(self.no_peaks(values))
                    AveEn_Ch.append(self.average_nonlinear_energy(values))

                ApEntropy_.append(ApEntropy_Ch)
                #Hurst_.append(Hurst_Ch)
                Mean_.append(Mean_Ch)
                # Var_.append(Var_Ch)
                SD_.append(SD_Ch)
                Skew_.append(Skew_Ch)
                Kur_.append(Kur_Ch)
                Curve_.append(Curve_Ch)
                Peaks_.append(Peaks_Ch)
                AveEn_.append(AveEn_Ch)

            ApEntropy_4band.append(ApEntropy_)
            #Hurst_4band.append(Hurst_)
            Mean_4band.append(Mean_)
            # Var_4band.append(Var_Ch)
            SD_4band.append(SD_)
            Skew_4band.append(Skew_)
            Kur_4band.append(Kur_)
            Curve_4band.append(Curve_)
            Peaks_4band.append(Peaks_)
            AveEn_4band.append(AveEn_)

        # ***************************************************************************************#
        # PSD band from non-overlapped 2Hz
        '''
        all_band = []
        for p in range(4, 40, 2):
            band = [p, p + 2]
            BPW_ = []
            for index, item in enumerate(all_slide):  # calculate for each time slices
                data = pd.DataFrame(item)
                BPW_Ch = []
                for name, values in data.iteritems():  # step through columns
                    BPW_Ch.append(self.bandpower(values, sf, band, window_sec))
                BPW_.append(BPW_Ch)
            all_band.append(BPW_)
        '''

        # ***************************************************************************************#

        '''
        Wavelet_ = []
        for index, item in enumerate(all_slide):  # calculate for each time slices
            data = pd.DataFrame(item)
            Wavelet_Ch = []
            for name, values in data.iteritems():  # step through columns
                values = values
                Wavelet = self.WE(self.DWT(values))
                Wavelet_Ch.append(Wavelet)
            Wavelet_.append(Wavelet_Ch)
        '''

        # ***************************************************************************************#

        #Track_Features['Mean4BL'] = Mean
        Track_Features['Mean'] = Mean_4band
        #Track_Features['SD4BL'] = SD
        Track_Features['SD'] = SD_4band
        #Track_Features['Skew4BL'] = Skew
        Track_Features['Skewness'] = Skew_4band
        #Track_Features['Kur4BL'] = Kur
        Track_Features['Kurtosis'] = Kur_4band
        #Track_Features['Curve4BL'] = Curve
        Track_Features['Curve_length'] = Curve_4band
        #Track_Features['Peaks4BL'] = Peaks
        Track_Features['No_peaks'] = Peaks_4band
        #Track_Features['AveEn4BL'] = AveEn
        Track_Features['Average_nonlinear_enegy'] = AveEn_4band
        #Track_Features['ApEntropy4BL'] = ApEntropy
        #Track_Features['ApEntropy'] = ApEntropy_4band
        #Track_Features['Hurst4BL'] = Hurst
        #Track_Features['Hurst'] = Hurst_4band
        #Track_Features['AR4BL'] = AR_6

        #Track_Features['WE'] = Wavelet_
        #Track_Features['all_band'] = all_band
        #Track_Features['all_band5'] = all_band5
        Track_Features['Window_Time_Stamp'] = Window_Time_Stamp

        # file_end = file_end.replace(".csv", ".json")
        # "Mean", "Mean4BL", "SD", "SD4BL", "Skewness", "Skew4BL", "Kurtosis", "Kur4BL", "Curve_length", "Curve4BL", "No_peaks", "Peaks4BL", "Average_nonlinear_enegy", "AveEn4BL", "ApEntropy", "ApEntropy4BL", "Hurst", "Hurst4BL", "WE", "all_band", "all_band5"

        # import json
        # json = json.dumps(Track_Features)

        # f = open(fr'C:\Users\gxb18167\PycharmProjects\ACAIN\STEW Dataset\Test\features_{file_end}', "w")
        # f.write(json)
        # f.close()

        return Track_Features

    import pandas as pd
    def create_feature_df(self, Feature_name, dict):

        if Feature_name != "WE":
            df_list = []
            #print(Feature_name)
            for x in range(0, len(dict[Feature_name])):
                feature = dict[Feature_name][x]
                Feature = Feature_name + str(x)

                df = pd.DataFrame(
                    columns=['1' + Feature, '2' + Feature, '3' + Feature, '4' + Feature, '5' + Feature, '6' + Feature,
                             '7' + Feature, '8' + Feature, '9' + Feature,
                             '10' + Feature, '11' + Feature, '12' + Feature, '13' + Feature, '14' + Feature,
                             '15' + Feature,
                             '16' + Feature, '17' + Feature, '18' + Feature,
                             '19' + Feature, '20' + Feature, '21' + Feature, '22' + Feature, '23' + Feature,
                             '24' + Feature, "25" + Feature, "26" + Feature, "27" + Feature, "28" + Feature, "29" + Feature
                             , "30" + Feature, "31" + Feature, "32" + Feature, "33" + Feature, "34" + Feature, "35" + Feature
                             , "36" + Feature, "37" + Feature, "38" + Feature, "39" + Feature, "40" + Feature, "41" + Feature])
                for x in range(0, len(feature)):
                    row_data = feature[x]
                    df.loc[len(df)] = row_data
                df_list.append(df)
            combined_df = pd.concat(df_list, axis=1)
            return combined_df

        else:
            Feature = Feature_name
            df = pd.DataFrame(
                columns=['1' + Feature, '2' + Feature, '3' + Feature, '4' + Feature, '5' + Feature, '6' + Feature,
                         '7' + Feature, '8' + Feature, '9' + Feature,
                         '10' + Feature, '11' + Feature, '12' + Feature, '13' + Feature, '14' + Feature,
                         '15' + Feature,
                         '16' + Feature, '17' + Feature, '18' + Feature,
                         '19' + Feature, '20' + Feature, '21' + Feature, '22' + Feature, '23' + Feature,
                         '24' + Feature, "25" + Feature, "26" + Feature, "27" + Feature, "28" + Feature, "29" + Feature
                    , "30" + Feature, "31" + Feature, "32" + Feature, "33" + Feature, "34" + Feature, "35" + Feature
                    , "36" + Feature, "37" + Feature, "38" + Feature, "39" + Feature, "40" + Feature, "41" + Feature])

            FeatureData = dict[Feature]
            FeatureData = FeatureData.T
            for x in range(0, len(FeatureData)):
                row_data = FeatureData[x]
                df.loc[len(df)] = row_data
            return df

    def format_features(self, Feature_Dict):
        import json

        Features = ['theta1', 'theta2', 'alpha1', 'alpha2', 'beta1', 'beta2', 'gamma1', 'gamma2']
        # Features = ["all_band5"]
        df_list = []

        for x in Features:
            df = self.create_feature_df(x, Feature_Dict)
            df_list.append(df)
        combined_df = pd.concat(df_list, axis=1)


        return combined_df



