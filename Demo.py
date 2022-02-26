import numpy as np
from scipy import signal
import pandas as pd
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

def train_eeg():

    file_id = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']
    predict_label = []
    trial_id = [i for i in range(1, 1601)]
    for i in range(20):
        train_data = np.load('../input/hnubci2022dataset/train_data_' + file_id[i] + '.npy')
        train_label = np.load('../input/hnubci2022dataset/train_label_' + file_id[i] + '.npy')
        test_data = np.load('../input/hnubci2022dataset/test_data_' + file_id[i] + '.npy')

        b, a = signal.butter(8, [8 / 125, 30 / 125], 'bandpass')
        train_data = signal.filtfilt(b, a, train_data)
        test_data = signal.filtfilt(b, a, test_data)

        csp = CSP(n_components=20, reg=None, log=False, norm_trace=False)
        svm = SVC(kernel='linear')
        clf = Pipeline([('CSP', csp), ('SVM', svm)])
        clf.fit(train_data, train_label)
        result = clf.predict(test_data)
        predict_label.extend(result)

    dataframe = pd.DataFrame({'TrialId': trial_id,
                              'Label': predict_label})
    dataframe.to_csv("sample_submission.csv", index=False, sep=',')

if __name__ == '__main__':
    train_eeg()
