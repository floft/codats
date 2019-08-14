"""
Datasets

Load the desired datasets into memory so we can write them to tfrecord files
in generate_tfrecords.py
"""
import os
import re
import io
import zipfile
import rarfile  # pip install rarfile
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_enum("normalize", "meanstd", ["none", "minmax", "meanstd"], "How to normalize data")


# For normalization
def calc_normalization(x, method):
    """
    Calculate zero mean unit variance normalization statistics

    We calculate separate mean/std or min/max statistics for each
    feature/channel, the default (-1) for BatchNormalization in TensorFlow and
    I think makes the most sense. If we set axis=0, then we end up with a
    separate statistic for each time step and feature, and then we can get odd
    jumps between time steps. Though, we get shape problems when setting axis=2
    in numpy, so instead we reshape/transpose.
    """
    # from (10000,100,1) to (1,100,10000)
    x = x.T
    # from (1,100,10000) to (1,100*10000)
    x = x.reshape((x.shape[0], -1))
    # then we compute statistics over axis=1, i.e. along 100*10000 and end up
    # with 1 statistic per channel (in this example only one)

    if method == "meanstd":
        values = (np.mean(x, axis=1), np.std(x, axis=1))
    elif method == "minmax":
        values = (np.min(x, axis=1), np.max(x, axis=1))
    else:
        raise NotImplementedError("unsupported normalization method")

    return method, values


def calc_normalization_jagged(x, method):
    """ Same as calc_normalization() except works for arrays of varying-length
    numpy arrays

    x should be: [
        np.array([example 1 time steps, example 1 features]),
        np.array([example 2 time steps, example 2 features]),
        ...
    ] where the # time steps can differ between examples.
    """
    num_features = x[0].shape[1]
    features = [None for x in range(num_features)]

    for example in x:
        transpose = example.T

        for i, feature_values in enumerate(transpose):
            if features[i] is None:
                features[i] = feature_values
            else:
                features[i] = np.concatenate([features[i], feature_values], axis=0)

    if method == "meanstd":
        values = (np.array([np.mean(x) for x in features], dtype=np.float32),
            np.array([np.std(x) for x in features], dtype=np.float32))
    elif method == "minmax":
        values = (np.array([np.min(x) for x in features], dtype=np.float32),
            np.array([np.max(x) for x in features], dtype=np.float32))
    else:
        raise NotImplementedError("unsupported normalization method")

    return method, values


def apply_normalization(x, normalization, epsilon=1e-5):
    """ Apply zero mean unit variance normalization statistics """
    method, values = normalization

    if method == "meanstd":
        mean, std = values
        x = (x - mean) / (std + epsilon)
    elif method == "minmax":
        minx, maxx = values
        x = (x - minx) / (maxx - minx + epsilon) - 0.5

    x[np.isnan(x)] = 0

    return x


def apply_normalization_jagged(x, normalization, epsilon=1e-5):
    """ Same as apply_normalization() except works for arrays of varying-length
    numpy arrays """
    normalized = []

    for example in x:
        normalized.append(apply_normalization(example, normalization, epsilon))

    return normalized


class Dataset:
    """
    Base class for datasets

    class Something(Dataset):
        num_classes = 2
        class_labels = ["class1", "class2"]
        window_size = 250
        window_overlap = False

        def __init__(self, *args, **kwargs):
            super().__init__(Something.num_classes, Something.class_labels,
                Something.window_size, Something.window_overlap,
                *args, **kwargs)

        def process(self, data, labels):
            ...
            return super().process(data, labels)

        def load(self):
            ...
            return train_data, train_labels, test_data, test_labels

    Also, add to the datasets={"something": Something, ...} dictionary below.
    """
    def __init__(self, num_classes, class_labels, window_size, window_overlap,
            feature_names=None, test_percent=0.2):
        """
        Initialize dataset

        Must specify num_classes and class_labels (the names of the classes).

        For example,
            Dataset(num_classes=2, class_labels=["class1", "class2"])

        This calls load() to get the data, process() to normalize, convert to
        float, etc.

        At the end, look at dataset.{train,test}_{data,labels}
        """
        # Sanity checks
        assert num_classes == len(class_labels), \
            "num_classes != len(class_labels)"

        # Set parameters
        self.num_classes = num_classes
        self.class_labels = class_labels
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.feature_names = feature_names
        self.test_percent = test_percent

        # Load the dataset
        train_data, train_labels, test_data, test_labels = self.load()

        if train_data is not None and train_labels is not None:
            self.train_data, self.train_labels = \
                self.process(train_data, train_labels)
        else:
            self.train_data = None
            self.train_labels = None

        if test_data is not None and test_labels is not None:
            self.test_data, self.test_labels = \
                self.process(test_data, test_labels)
        else:
            self.test_data = None
            self.test_labels = None

    def load(self):
        raise NotImplementedError("must implement load() for Dataset class")

    def download_dataset(self, files_to_download, url):
        """
        Download url/file for file in files_to_download
        Returns: the downloaded filenames for each of the files given
        """
        downloaded_files = []

        for f in files_to_download:
            downloaded_files.append(tf.keras.utils.get_file(
                fname=f, origin=url+"/"+f))

        return downloaded_files

    def process(self, data, labels):
        """ Perform conversions, etc. If you override,
        you should `return super().process(data, labels)` to make sure these
        options are handled. """
        # TODO
        return data, labels

    def one_hot(self, y, index_one=False):
        """ One-hot encode y if not already 2D """
        squeezed = np.squeeze(y)

        if len(squeezed.shape) < 2:
            if index_one:
                y = np.eye(self.num_classes, dtype=np.float32)[squeezed.astype(np.int32) - 1]
            else:
                y = np.eye(self.num_classes, dtype=np.float32)[squeezed.astype(np.int32)]
        else:
            y = y.astype(np.float32)
            assert squeezed.shape[1] == self.num_classes, "y.shape[1] != num_classes"

        return y

    def create_windows_x(self, x, window_size, overlap):
        """
        Concatenate along dim-1 to meet the desired window_size. We'll skip any
        windows that reach beyond the end. Only process x (saves memory).

        Two options (examples for window_size=5):
            Overlap - e.g. window 0 will be a list of examples 0,1,2,3,4 and the
                label of example 4; and window 1 will be 1,2,3,4,5 and the label of
                example 5
            No overlap - e.g. window 0 will be a list of examples 0,1,2,3,4 and the
                label of example 4; and window 1 will be 5,6,7,8,9 and the label of
                example 9
        """
        x = np.expand_dims(x, axis=1)

        # No work required if the window size is 1, only part required is
        # the above expand dims
        if window_size == 1:
            return x

        windows_x = []
        i = 0

        while i < len(x)-window_size:
            window_x = np.expand_dims(np.concatenate(x[i:i+window_size], axis=0), axis=0)
            windows_x.append(window_x)

            # Where to start the next window
            if overlap:
                i += 1
            else:
                i += window_size

        return np.vstack(windows_x)

    def create_windows_y(self, y, window_size, overlap):
        """
        Concatenate along dim-1 to meet the desired window_size. We'll skip any
        windows that reach beyond the end. Only process y (saves memory).

        Two options (examples for window_size=5):
            Overlap - e.g. window 0 will be a list of examples 0,1,2,3,4 and the
                label of example 4; and window 1 will be 1,2,3,4,5 and the label of
                example 5
            No overlap - e.g. window 0 will be a list of examples 0,1,2,3,4 and the
                label of example 4; and window 1 will be 5,6,7,8,9 and the label of
                example 9
        """
        # No work required if the window size is 1
        if window_size == 1:
            return y

        windows_y = []
        i = 0

        while i < len(y)-window_size:
            window_y = y[i+window_size-1]
            windows_y.append(window_y)

            # Where to start the next window
            if overlap:
                i += 1
            else:
                i += window_size

        return np.hstack(windows_y)

    def create_windows(self, x, y, window_size, overlap):
        """ Split time-series data into windows """
        x = self.create_windows_x(x, window_size, overlap)
        y = self.create_windows_y(y, window_size, overlap)
        return x, y

    def train_test_split(self, x, y, random_state=42):
        """
        Split x and y data into train/test sets

        Warning: train_test_split() is from sklearn but self.train_test_split()
        is this function, which is what you should use.
        """
        x_train, x_test, y_train, y_test = train_test_split(x, y,
            test_size=self.test_percent, stratify=y, random_state=random_state)
        return x_train, y_train, x_test, y_test

    def label_to_int(self, label_name):
        """ e.g. Bathe to 0 """
        return self.class_labels.index(label_name)

    def int_to_label(self, label_index):
        """ e.g. Bathe to 0 """
        return self.class_labels[label_index]


class UTDataBase(Dataset):
    """ Base class for loading the wrist or pocket UT-Data-Complex datasets """
    invertible = False
    num_classes = 13
    class_labels = ["walk", "stand", "jog", "sit", "bike", "upstairs",
        "downstairs", "type", "write", "coffee", "talk", "smoke", "eat"]
    window_size = 250  # 5s of data sampled at 50Hz
    window_overlap = False
    feature_names = [
        "acc_x", "acc_y", "acc_z",
        "lacc_x", "lacc_y", "lacc_z",
        "gyr_x", "gyr_y", "gyr_z",
        "mag_x", "mag_y", "mag_z",
    ]

    def __init__(self, utdata_domain, *args, **kwargs):
        self.utdata_domain = utdata_domain
        super().__init__(UTDataBase.num_classes, UTDataBase.class_labels,
            UTDataBase.window_size, UTDataBase.window_overlap,
            UTDataBase.feature_names, *args, **kwargs)

    def download(self):
        (dataset_fp,) = self.download_dataset(["ut-data-complex.rar"],
            "https://www.utwente.nl/en/eemcs/ps/dataset-folder/")
        return dataset_fp

    def get_file_in_zip(self, archive, filename):
        """ Read one file out of the already-open zip file """
        with archive.open(filename) as fp:
            contents = fp.read()
        return contents

    def parse_csv(self, content):
        """ Load a CSV file, convert data columns to float and label to int """
        lines = content.decode("utf-8").strip().split("\n")
        data = []
        labels = []

        for line in lines:
            timestamp, acc_x, acc_y, acc_z, lacc_x, lacc_y, lacc_z, \
                gyr_x, gyr_y, gyr_z, mag_x, mag_y, mag_z, label = \
                line.split(",")

            acc_x = float(acc_x)
            acc_y = float(acc_y)
            acc_z = float(acc_z)
            lacc_x = float(lacc_x)
            lacc_y = float(lacc_y)
            lacc_z = float(lacc_z)
            gyr_x = float(gyr_x)
            gyr_y = float(gyr_y)
            gyr_z = float(gyr_z)
            mag_x = float(mag_x)
            mag_y = float(mag_y)
            mag_z = float(mag_z)
            label = int(label) - 11111  # make in range [0,12]

            data.append([acc_x, acc_y, acc_z, lacc_x, lacc_y, lacc_z,
                gyr_x, gyr_y, gyr_z, mag_x, mag_y, mag_z])
            labels.append(label)

        return data, labels

    def load_file(self, filename):
        """ Load RAR file, get CSV data from it, convert to numpy arrays """
        with rarfile.RarFile(filename, "r") as archive:
            filelist = archive.namelist()

            for f in filelist:
                if "UT_Data_Complex/" in f:
                    folder, filename = os.path.split(
                        f.replace("UT_Data_Complex/", ""))

                    if self.utdata_domain+".csv" in filename:
                        contents = self.get_file_in_zip(archive, f)
                        data, labels = self.parse_csv(contents)

        x = np.vstack(data).astype(np.float32)
        y = np.hstack(labels).astype(np.float32)

        return x, y

    def load(self):
        # Load data
        dataset_fp = self.download()
        all_data, all_labels = self.load_file(dataset_fp)
        # Split time-series data into windows
        x_windows, y_windows = self.create_windows(all_data, all_labels,
            self.window_size, self.window_overlap)
        # Split into train/test sets
        train_data, train_labels, test_data, test_labels = \
            self.train_test_split(x_windows, y_windows)

        return train_data, train_labels, test_data, test_labels

    def process(self, data, labels):
        """ Normalize, one-hot encode labels """
        #data = (data - 127.5) / 127.5  # TODO normalize
        labels = self.one_hot(labels)
        return super().process(data, labels)


class UTDataWrist(UTDataBase):
    def __init__(self, *args, **kwargs):
        super().__init__("wrist", *args, **kwargs)


class UTDataPocket(UTDataBase):
    def __init__(self, *args, **kwargs):
        super().__init__("pocket", *args, **kwargs)


class UnivariateCSVBase(Dataset):
    """ Base class for loading UCR-like univarate datasets """
    invertible = False
    feature_names = ["univariate"]

    def __init__(self, train_filename, test_filename, num_classes, class_labels,
            *args, **kwargs):
        self.train_filename = train_filename
        self.test_filename = test_filename
        super().__init__(num_classes, class_labels, None, None,
            UnivariateCSVBase.feature_names, *args, **kwargs)

    def load_file(self, filename):
        """
        Load CSV files in UCR time-series data format

        Load a time-series dataset. This is set up to load data in the format of the
        UCR time-series datasets (http://www.cs.ucr.edu/~eamonn/time_series_data/).
        Or, see the generate_trivial_datasets.py for a trivial dataset.

        Returns:
            data - numpy array with data of shape (num_examples, num_features)
            labels - numpy array with labels of shape: (num_examples, 1)
        """
        df = pd.read_csv(filename, header=None)
        df_data = df.drop(0, axis=1).values.astype(np.float32)
        df_labels = df.loc[:, df.columns == 0].values.astype(np.uint8)
        return df_data, df_labels

    def load(self):
        train_data, train_labels = self.load_file(self.train_filename)
        test_data, test_labels = self.load_file(self.test_filename)

        return train_data, train_labels, test_data, test_labels

    def process(self, data, labels):
        """ Normalize, one-hot encode labels
        Note: UCR datasets are index-one """
        # For if we only have one feature
        # [examples, time_steps] --> [examples, time_steps, 1]
        if len(data.shape) < 3:
            data = np.expand_dims(data, axis=2)

        labels = self.one_hot(labels, index_one=True)
        return super().process(data, labels)


class MultivariateCSVBase(Dataset):
    """ Base class for loading UCR-like multivariate datasets, where we have
    x and y rather than just one feature """
    invertible = False
    feature_names = ["x", "y"]

    def __init__(self, train_filename, test_filename, num_classes, class_labels,
            *args, **kwargs):
        self.train_filename = train_filename
        self.test_filename = test_filename
        super().__init__(num_classes, class_labels, None, None,
            MultivariateCSVBase.feature_names, *args, **kwargs)

    def load_file(self, filename):
        """
        Load CSV files in UCR time-series data format but with semicolons
        delimiting the features

        Returns:
            data - numpy array with data of shape (num_examples, time_steps, num_features)
            labels - numpy array with labels of shape: (num_examples, 1)
        """
        with open(filename, "r") as f:
            data = []
            labels = []

            for line in f:
                parts = line.split(",")
                assert len(parts) >= 2, "must be at least a label and a data value"
                label = int(parts[0])
                values_str = parts[1:]
                values = []

                for value in values_str:
                    features_str = value.split(";")
                    features = [float(v) for v in features_str]
                    values.append(features)

                labels.append(label)
                data.append(values)

        data = np.array(data, dtype=np.float32)
        labels = np.expand_dims(np.array(labels, dtype=np.int32), axis=1)

        return data, labels

    def load(self):
        train_data, train_labels = self.load_file(self.train_filename)
        test_data, test_labels = self.load_file(self.test_filename)

        return train_data, train_labels, test_data, test_labels

    def process(self, data, labels):
        """ Normalize, one-hot encode labels
        Note: UCR datasets are index-one """
        assert len(data.shape) == 3, \
            "multivariate data should be of shape [examples, time_steps, 2]"
        labels = self.one_hot(labels, index_one=True)
        return super().process(data, labels)


class uWaveBase(Dataset):
    """
    uWave Gesture dataset

    See: https://zhen-wang.appspot.com/rice/projects_uWave.html

    Either split on days or users:
      - If users: pass days=None, users=[1,2,3,4,5,6,7,8]
      - If days: pass days=[1,2,3,4,5,6,7], users=None
      - To get all data: days=None, users=None
    (or specify any subset of those users/days)
    """
    invertible = False
    feature_names = ["accel_x", "accel_y", "accel_z"]

    def __init__(self, days, users, num_classes, class_labels,
            *args, **kwargs):
        self.days = days
        self.users = users
        super().__init__(num_classes, class_labels, None, None,
            uWaveBase.feature_names, *args, **kwargs)

    def download(self):
        (dataset_fp,) = self.download_dataset(["uWaveGestureLibrary.zip"],
            "https://zhen-wang.appspot.com/rice/files/uwave")
        return dataset_fp

    def get_file_in_archive(self, archive, filename):
        """ Read one file out of the already-open zip/rar file """
        with archive.open(filename) as fp:
            contents = fp.read()
        return contents

    def parse_example(self, filename, content):
        """ Load file containing a single example """
        # Get data
        lines = content.decode("utf-8").strip().split("\n")
        data = []

        for line in lines:
            x, y, z = line.split(" ")

            x = float(x)
            y = float(y)
            z = float(z)

            data.append([x, y, z])

        data = np.array(data, dtype=np.float32)

        # Get label from filename
        # Note: there's at least one without a repeat_index, so make it optional
        matches = re.findall(r"[0-9]+-[0-9]*", filename)
        assert len(matches) == 1, \
            "Filename should be in format X_Template_Acceleration#-#.txt but is " \
            + filename + " instead"
        parts = matches[0].split("-")
        assert len(parts) == 2, "Match should be tuple of (gesture index, repeat index)"
        gesture_index, repeat_index = parts
        # The label is the gesture index
        label = int(gesture_index)

        return data, label

    def load_rar(self, filename):
        """ Load RAR file containing examples from one user for one day """
        data = []
        labels = []

        with rarfile.RarFile(filename, "r") as archive:
            filelist = archive.namelist()

            for f in filelist:
                if ".txt" in f and "Acceleration" in f:
                    contents = self.get_file_in_archive(archive, f)
                    new_data, new_label = self.parse_example(f, contents)
                    data.append(new_data)
                    labels.append(new_label)

        return data, labels

    def load_zip(self, filename):
        """ Load ZIP file containing all the RAR files """
        data = []
        labels = []

        with zipfile.ZipFile(filename, "r") as archive:
            filelist = archive.namelist()

            for f in filelist:
                if ".rar" in f:
                    matches = re.findall(r"[0-9]+", f)
                    assert len(matches) == 2, "should be 2 numbers in .rar filename"
                    user, day = matches
                    user = int(user)
                    day = int(day)

                    # Skip data we don't want
                    if self.users is not None:
                        if user not in self.users:
                            print("Skipping user", user)
                            continue

                    if self.days is not None:
                        if day not in self.days:
                            print("Skipping day", day)
                            continue

                    print("Processing user", user, "day", day)
                    contents = self.get_file_in_archive(archive, f)
                    new_data, new_labels = self.load_rar(io.BytesIO(contents))
                    data += new_data
                    labels += new_labels

        # Zero pad (appending zeros) to make all the same shape
        # for uwave_all, we know the max max([x.shape[0] for x in data]) = 315
        # and expand the dimensions to [1, time_steps, num_features] so we can
        # vstack them properly
        #data = [np.expand_dims(self.pad_to(d, 315), axis=0) for d in data]

        #x = np.vstack(data).astype(np.float32)
        y = np.hstack(labels).astype(np.float32)

        return data, y

    def pad_to(self, data, desired_length):
        """ Pad to the desired length """
        current_length = data.shape[0]
        assert current_length <= desired_length, "Cannot shrink size by padding"
        return np.pad(data, [(0, desired_length - current_length), (0, 0)],
                mode="constant", constant_values=0)

    def load(self):
        # Load data
        dataset_fp = self.download()
        x, y = self.load_zip(dataset_fp)
        # Split into train/test sets
        train_data, train_labels, test_data, test_labels = \
            self.train_test_split(x, y)

        # Normalize here since we know which data is train vs. test and we have
        # to normalize before we zero pad or the zero padding messes up the
        # mean calculation a lot
        #
        # TODO we'll normalize this twice, once here and once later on when
        # generating the tfrecord files.... but, it's the same normalization, so
        # it doesn't really matter, but it's a waste of time
        if FLAGS.normalize != "none":
            normalization = calc_normalization_jagged(train_data, FLAGS.normalize)
            train_data = apply_normalization_jagged(train_data, normalization)
            test_data = apply_normalization_jagged(test_data, normalization)

        # Then zero-pad to be the right length
        train_data = np.vstack([np.expand_dims(self.pad_to(d, 315), axis=0)
            for d in train_data]).astype(np.float32)
        test_data = np.vstack([np.expand_dims(self.pad_to(d, 315), axis=0)
            for d in test_data]).astype(np.float32)

        return train_data, train_labels, test_data, test_labels

    def process(self, data, labels):
        """ One-hot encode labels
        Note: uWave classes are index-one """
        # Check we have data in [examples, time_steps, 3]
        assert len(data.shape) == 3, "should shape [examples, time_steps, 3]"
        assert data.shape[2] == 3, "should have 3 features"

        labels = self.one_hot(labels, index_one=True)
        return super().process(data, labels)


class SleepBase(Dataset):
    """
    Loads sleep RF data files in datasets/RFSleep.zip/*.npy

    Notes:
      - RF data is 30 seconds of data sampled at 25 samples per second, thus
        750 samples. For each of these sets of 750 samples there is a stage
        label.
      - The RF data is complex, so we'll split the complex 5 features into
        the 5 real and then 5 imaginary components to end up with 10 features.
    """
    invertible = False
    feature_names = ["real1", "real2", "real3", "real4", "real5",
        "imag1", "imag2", "imag3", "imag4", "imag5"]

    def __init__(self, days, users, num_classes, class_labels,
            *args, **kwargs):
        self.days = days
        self.users = users
        super().__init__(num_classes, class_labels, None, None,
            SleepBase.feature_names, *args, **kwargs)

    def download(self):
        (dataset_fp,) = self.download_dataset(["uWaveGestureLibrary.zip"],
            "https://zhen-wang.appspot.com/rice/files/uwave")
        return dataset_fp

    def get_file_in_archive(self, archive, filename):
        """ Read one file out of the already-open zip/rar file """
        with archive.open(filename) as fp:
            contents = fp.read()
        return contents

    def process_examples(self, filename, fp):
        d = np.load(fp, allow_pickle=True).item()
        day = int(filename.replace(".npy", ""))
        user = d["subject"]

        # Skip data we don't want
        if self.days is not None:
            if day not in self.days:
                print("Skipping day", day)
                return None, None

        if self.users is not None:
            if user not in self.users:
                print("Skipping user", user)
                return None, None

        print("Processing user", user, "day", day)

        stage_labels = d["stage"]
        rf = d["rf"]

        # Split 5 complex features into 5 real and 5 imaginary, i.e.
        # now we have 10 features
        rf = np.vstack([np.real(rf), np.imag(rf)])

        assert stage_labels.shape[0]*750 == rf.shape[-1], \
            "If stage labels is of shape (n) then rf should be of shape (5, 750n)"

        # Reshape and transpose into desired format
        x = np.transpose(np.reshape(rf, (rf.shape[0], -1, stage_labels.shape[0])))

        # Drop those that have a label other than 0-5 (sleep stages) since
        # label 6 means "no signal" and 9 means "error"
        no_error = stage_labels < 6
        x = x[no_error]
        stage_labels = stage_labels[no_error]

        assert x.shape[0] == stage_labels.shape[0], \
            "Incorrect first dimension of x (not length of stage labels)"
        assert x.shape[1] == 750, \
            "Incorrect second dimension of x (not 750)"
        assert x.shape[2] == 10, \
            "Incorrect third dimension of x (not 10)"

        return x, stage_labels

    def load_file(self, filename):
        """ Load ZIP file containing all the .npy files """
        if not os.path.exists(filename):
            print("Download unencrypted "+filename+" into the current directory")

        data = []
        labels = []

        with zipfile.ZipFile(filename, "r") as archive:
            filelist = archive.namelist()

            for f in filelist:
                if ".npy" in f:
                    contents = self.get_file_in_archive(archive, f)
                    x, label = self.process_examples(f, io.BytesIO(contents))

                    if x is not None and label is not None:
                        data.append(x)
                        labels.append(label)

        x = np.vstack(data).astype(np.float32)
        y = np.hstack(labels).astype(np.float32)

        return x, y

    def load(self):
        x, y = self.load_file("RFSleep_unencrypted.zip")
        train_data, train_labels, test_data, test_labels = \
            self.train_test_split(x, y)

        return train_data, train_labels, test_data, test_labels

    def process(self, data, labels):
        """ One-hot encode labels """
        labels = self.one_hot(labels, index_one=False)
        return super().process(data, labels)


class UciHarBase(Dataset):
    """
    Loads human activity recognition data files in datasets/UCI HAR Dataset.zip

    Download from:
    https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
    """
    invertible = False
    feature_names = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z",
    ]
    num_classes = 6
    class_labels = [
        "walking", "walking_upstairs", "walking_downstairs",
        "sitting", "standing", "laying",
    ]

    def __init__(self, users, *args, **kwargs):
        self.users = users
        super().__init__(UciHarBase.num_classes, UciHarBase.class_labels,
            None, None, UciHarBase.feature_names, *args, **kwargs)

    def download(self):
        (dataset_fp,) = self.download_dataset(["UCI%20HAR%20Dataset.zip"],
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00240")
        return dataset_fp

    def get_file_in_archive(self, archive, filename):
        """ Read one file out of the already-open zip/rar file """
        with archive.open(filename) as fp:
            contents = fp.read()
        return contents

    def get_feature(self, content):
        """
        Read the space-separated, example on each line file

        Returns 2D array with dimensions: [num_examples, num_time_steps]
        """
        lines = content.decode("utf-8").strip().split("\n")
        features = []

        for line in lines:
            features.append([float(v) for v in line.strip().split()])

        return features

    def get_data(self, archive, name):
        """ To shorten duplicate code for name=train or name=test cases """
        def get_data_single(f):
            return self.get_feature(self.get_file_in_archive(archive,
                "UCI HAR Dataset/"+f))

        data = [
            get_data_single(name+"/Inertial Signals/body_acc_x_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/body_acc_y_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/body_acc_z_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/body_gyro_x_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/body_gyro_y_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/body_gyro_z_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/total_acc_x_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/total_acc_y_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/total_acc_z_"+name+".txt"),
        ]

        labels = get_data_single(name+"/y_"+name+".txt")

        subjects = get_data_single(name+"/subject_"+name+".txt")

        data = np.array(data, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        # Squeeze so we can easily do selection on this later on
        subjects = np.squeeze(np.array(subjects, dtype=np.float32))

        # Transpose from [features, examples, time_steps] to
        # [examples, time_steps (128), features (9)]
        data = np.transpose(data, axes=[1, 2, 0])

        return data, labels, subjects

    def load_file(self, filename):
        """ Load ZIP file containing all the .txt files """
        with zipfile.ZipFile(filename, "r") as archive:
            train_data, train_labels, train_subjects = self.get_data(archive, "train")
            test_data, test_labels, test_subjects = self.get_data(archive, "test")

        all_data = np.vstack([train_data, test_data]).astype(np.float32)
        all_labels = np.vstack([train_labels, test_labels]).astype(np.float32)
        all_subjects = np.hstack([train_subjects, test_subjects]).astype(np.float32)

        # All data if no selection
        if self.users is None:
            return all_data, all_labels

        # Otherwise, select based on the desired users
        data = []
        labels = []

        for user in self.users:
            selection = all_subjects == user
            data.append(all_data[selection])
            labels.append(all_labels[selection])

        x = np.vstack(data).astype(np.float32)
        y = np.vstack(labels).astype(np.float32)

        # print("Selected data:", self.users)
        # print(x.shape, y.shape)

        return x, y

    def load(self):
        dataset_fp = self.download()
        x, y = self.load_file(dataset_fp)
        train_data, train_labels, test_data, test_labels = \
            self.train_test_split(x, y)

        return train_data, train_labels, test_data, test_labels

    def process(self, data, labels):
        """ One-hot encode labels """
        labels = self.one_hot(labels, index_one=True)
        return super().process(data, labels)


def make_trivial_negpos(filename_prefix):
    """ make a -/+ dataset object, since we have a bunch of these """
    class Trivial(UnivariateCSVBase):
        invertible = False
        num_classes = 2
        class_labels = ["negative", "positive"]

        def __init__(self, *args, **kwargs):
            super().__init__(
                "trivial/"+filename_prefix+"_TRAIN",
                "trivial/"+filename_prefix+"_TEST",
                Trivial.num_classes,
                Trivial.class_labels,
                *args, **kwargs)

    return Trivial


def make_trivial_lowhigh(filename_prefix):
    """ make a low/high dataset object, since we have a bunch of these """
    class Trivial(UnivariateCSVBase):
        invertible = False
        num_classes = 2
        class_labels = ["low", "high"]

        def __init__(self, *args, **kwargs):
            super().__init__(
                "trivial/"+filename_prefix+"_TRAIN",
                "trivial/"+filename_prefix+"_TEST",
                Trivial.num_classes,
                Trivial.class_labels,
                *args, **kwargs)

    return Trivial


def make_trivial_negpos_invertible(filename_prefix):
    """ make a -/+ dataset object, since we have a bunch of these """
    class Trivial(UnivariateCSVBase):
        invertible = True
        num_classes = 2
        class_labels = ["negative", "positive"]

        def __init__(self, *args, **kwargs):
            super().__init__(
                "trivial/"+filename_prefix+"_TRAIN",
                "trivial/"+filename_prefix+"_TEST",
                Trivial.num_classes,
                Trivial.class_labels,
                *args, **kwargs)

    return Trivial


def make_trivial_lowhigh_invertible(filename_prefix):
    """ make a low/high dataset object, since we have a bunch of these """
    class Trivial(UnivariateCSVBase):
        invertible = True
        num_classes = 2
        class_labels = ["low", "high"]

        def __init__(self, *args, **kwargs):
            super().__init__(
                "trivial/"+filename_prefix+"_TRAIN",
                "trivial/"+filename_prefix+"_TEST",
                Trivial.num_classes,
                Trivial.class_labels,
                *args, **kwargs)

    return Trivial


def make_trivial2d(filename_prefix):
    """ make a -/+ dataset object """
    class Trivial(MultivariateCSVBase):
        invertible = False
        num_classes = 2
        class_labels = ["negative", "positive"]

        def __init__(self, *args, **kwargs):
            super().__init__(
                "trivial/"+filename_prefix+"_TRAIN",
                "trivial/"+filename_prefix+"_TEST",
                Trivial.num_classes,
                Trivial.class_labels,
                *args, **kwargs)

    return Trivial


def make_uwave(days=None, users=None):
    """ Make uWave dataset split on either days or users """
    class uWaveGestures(uWaveBase):
        invertible = False
        num_classes = 8
        class_labels = list(range(num_classes))

        def __init__(self, *args, **kwargs):
            super().__init__(
                days, users,
                uWaveGestures.num_classes,
                uWaveGestures.class_labels,
                *args, **kwargs)

    return uWaveGestures


def make_sleep(days=None, users=None):
    """ Make RF sleep dataset split on either days or users """
    class SleepDataset(SleepBase):
        invertible = False
        num_classes = 6
        class_labels = ["Awake", "N1", "N2", "N3", "Light N2", "REM"]

        def __init__(self, *args, **kwargs):
            super().__init__(
                days, users,
                SleepDataset.num_classes,
                SleepDataset.class_labels,
                *args, **kwargs)

    return SleepDataset


def make_ucihar(users=None):
    """ Make UCI HAR dataset split on users """
    class UciHarDataset(UciHarBase):
        def __init__(self, *args, **kwargs):
            super().__init__(users, *args, **kwargs)

    return UciHarDataset


# List of datasets
datasets = {
    "utdata_wrist": UTDataWrist,
    "utdata_pocket": UTDataPocket,
    "uwave_all": make_uwave(),
    "uwave_days_first": make_uwave(days=[1, 2, 3]),
    "uwave_days_second": make_uwave(days=[5, 6, 7]),
    # Note: in the paper, call these "participants" not "users"
    "uwave_users_first": make_uwave(users=[1, 2, 3, 4]),
    "uwave_users_second": make_uwave(users=[5, 6, 7, 8]),
    # From user X to user Y
    "uwave_1": make_uwave(users=[1]),
    "uwave_2": make_uwave(users=[2]),
    "uwave_3": make_uwave(users=[3]),
    "uwave_4": make_uwave(users=[4]),
    "uwave_5": make_uwave(users=[5]),
    "uwave_6": make_uwave(users=[6]),
    "uwave_7": make_uwave(users=[7]),
    "uwave_8": make_uwave(users=[8]),

    "sleep_all": make_sleep(),
    # Split users randomly since if first/second half then there's a large
    # difference in amount of data
    # import random; a = list(range(0, 26)); random.shuffle(a)
    # First and second: a[:13], a[13:]
    "sleep_users_first": make_sleep(users=[21, 15, 25, 19, 8, 23, 4, 12, 10, 13, 0, 9, 3]),
    "sleep_users_second": make_sleep(users=[17, 16, 6, 2, 20, 18, 1, 24, 22, 7, 5, 11, 14]),

    # The users of the original train/test sets
    "ucihar_train": make_ucihar(users=[1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]),
    "ucihar_test": make_ucihar(users=[2, 4, 9, 10, 12, 13, 18, 20, 24]),
    # Split the users in half
    "ucihar_first": make_ucihar(users=[17, 6, 22, 9, 19, 20, 14, 15, 16, 10, 26, 23, 7, 4, 24]),
    "ucihar_second": make_ucihar(users=[11, 3, 1, 30, 13, 28, 12, 21, 27, 25, 18, 5, 8, 29, 2]),
    # From user X to user Y
    "ucihar_1": make_ucihar(users=[1]),
    "ucihar_2": make_ucihar(users=[2]),
    "ucihar_3": make_ucihar(users=[3]),
    "ucihar_4": make_ucihar(users=[4]),
    "ucihar_5": make_ucihar(users=[5]),
    "ucihar_6": make_ucihar(users=[6]),
    "ucihar_7": make_ucihar(users=[7]),
    "ucihar_8": make_ucihar(users=[8]),
    "ucihar_9": make_ucihar(users=[9]),
    "ucihar_10": make_ucihar(users=[10]),
    "ucihar_11": make_ucihar(users=[11]),
    "ucihar_12": make_ucihar(users=[12]),
    "ucihar_13": make_ucihar(users=[13]),
    "ucihar_14": make_ucihar(users=[14]),
    "ucihar_15": make_ucihar(users=[15]),
    "ucihar_16": make_ucihar(users=[16]),
    "ucihar_17": make_ucihar(users=[17]),
    "ucihar_18": make_ucihar(users=[18]),
    "ucihar_19": make_ucihar(users=[19]),
    "ucihar_20": make_ucihar(users=[20]),
    "ucihar_21": make_ucihar(users=[21]),
    "ucihar_22": make_ucihar(users=[22]),
    "ucihar_23": make_ucihar(users=[23]),
    "ucihar_24": make_ucihar(users=[24]),
    "ucihar_25": make_ucihar(users=[25]),
    "ucihar_26": make_ucihar(users=[26]),
    "ucihar_27": make_ucihar(users=[27]),
    "ucihar_28": make_ucihar(users=[28]),
    "ucihar_29": make_ucihar(users=[29]),
    "ucihar_30": make_ucihar(users=[30]),


    # "positive_slope": make_trivial_negpos("positive_slope"),
    # "positive_slope_low": make_trivial_negpos("positive_slope_low"),
    # "positive_slope_noise": make_trivial_negpos("positive_slope_noise"),
    # "positive_sine": make_trivial_negpos("positive_sine"),
    # "positive_sine_low": make_trivial_negpos("positive_sine_low"),
    # "positive_sine_noise": make_trivial_negpos("positive_sine_noise"),
    # "freq_low": make_trivial_lowhigh("freq_low"),
    # "freq_high": make_trivial_lowhigh("freq_high"),
    # "freq_low_amp_noise": make_trivial_lowhigh("freq_low_amp_noise"),
    # "freq_high_amp_noise": make_trivial_lowhigh("freq_high_amp_noise"),
    # "freq_low_freq_noise": make_trivial_lowhigh("freq_low_freq_noise"),
    # "freq_high_freq_noise": make_trivial_lowhigh("freq_high_freq_noise"),
    # "freq_low_freqamp_noise": make_trivial_lowhigh("freq_low_freqamp_noise"),
    # "freq_high_freqamp_noise": make_trivial_lowhigh("freq_high_freqamp_noise"),
    # "freqshift_low": make_trivial_lowhigh("freqshift_low"),
    # "freqshift_high": make_trivial_lowhigh("freqshift_high"),
    # "freqscale_low": make_trivial_lowhigh("freqscale_low"),
    # "freqscale_high": make_trivial_lowhigh("freqscale_high"),
    # "line1low": make_trivial_negpos_invertible("line1low"),
    # "line1high": make_trivial_negpos_invertible("line1high"),
    # "line2low": make_trivial_negpos_invertible("line2low"),
    # "line2high": make_trivial_negpos_invertible("line2high"),
    # "sine1low": make_trivial_negpos_invertible("sine1low"),
    # "sine1high": make_trivial_negpos_invertible("sine1high"),
    # "sine2low": make_trivial_negpos_invertible("sine2low"),
    # "sine2high": make_trivial_negpos_invertible("sine2high"),
    # "sine3low": make_trivial_negpos_invertible("sine3low"),
    # "sine3high": make_trivial_negpos_invertible("sine3high"),
    # "sine4low": make_trivial_negpos_invertible("sine4low"),
    # "sine4high": make_trivial_negpos_invertible("sine4high"),

    # "lineslope1low": make_trivial_negpos_invertible("lineslope1low"),
    # "lineslope1high": make_trivial_negpos_invertible("lineslope1high"),
    # "lineslope2low": make_trivial_negpos_invertible("lineslope2low"),
    # "lineslope2high": make_trivial_negpos_invertible("lineslope2high"),
    # "sineslope1low": make_trivial_negpos_invertible("sineslope1low"),
    # "sineslope1high": make_trivial_negpos_invertible("sineslope1high"),
    # "sineslope2low": make_trivial_negpos_invertible("sineslope2low"),
    # "sineslope2high": make_trivial_negpos_invertible("sineslope2high"),
    # "sineslope3low": make_trivial_negpos_invertible("sineslope3low"),
    # "sineslope3high": make_trivial_negpos_invertible("sineslope3high"),
    # "sineslope4low": make_trivial_negpos_invertible("sineslope4low"),
    # "sineslope4high": make_trivial_negpos_invertible("sineslope4high"),

    # "freqshift_a": make_trivial_negpos("freqshift_a"),
    # "freqshift_b0": make_trivial_negpos("freqshift_b0"),
    # "freqshift_b1": make_trivial_negpos("freqshift_b1"),
    # "freqshift_b2": make_trivial_negpos("freqshift_b2"),
    # "freqshift_b3": make_trivial_negpos("freqshift_b3"),
    # "freqshift_b4": make_trivial_negpos("freqshift_b4"),
    # "freqshift_b5": make_trivial_negpos("freqshift_b5"),
    "freqshift_phase_a": make_trivial_negpos("freqshift_phase_a"),
    "freqshift_phase_b0": make_trivial_negpos("freqshift_phase_b0"),
    "freqshift_phase_b1": make_trivial_negpos("freqshift_phase_b1"),
    "freqshift_phase_b2": make_trivial_negpos("freqshift_phase_b2"),
    "freqshift_phase_b3": make_trivial_negpos("freqshift_phase_b3"),
    "freqshift_phase_b4": make_trivial_negpos("freqshift_phase_b4"),
    "freqshift_phase_b5": make_trivial_negpos("freqshift_phase_b5"),
    "freqshift_phase_b6": make_trivial_negpos("freqshift_phase_b6"),
    "freqshift_phase_b7": make_trivial_negpos("freqshift_phase_b7"),
    "freqshift_phase_b8": make_trivial_negpos("freqshift_phase_b8"),
    "freqshift_phase_b9": make_trivial_negpos("freqshift_phase_b9"),
    "freqshift_phase_b10": make_trivial_negpos("freqshift_phase_b10"),

    # "freqscale_a": make_trivial_negpos("freqscale_a"),
    # "freqscale_b0": make_trivial_negpos("freqscale_b0"),
    # "freqscale_b1": make_trivial_negpos("freqscale_b1"),
    # "freqscale_b2": make_trivial_negpos("freqscale_b2"),
    # "freqscale_b3": make_trivial_negpos("freqscale_b3"),
    # "freqscale_b4": make_trivial_negpos("freqscale_b4"),
    # "freqscale_b5": make_trivial_negpos("freqscale_b5"),
    "freqscale_phase_a": make_trivial_negpos("freqscale_phase_a"),
    "freqscale_phase_b0": make_trivial_negpos("freqscale_phase_b0"),
    "freqscale_phase_b1": make_trivial_negpos("freqscale_phase_b1"),
    "freqscale_phase_b2": make_trivial_negpos("freqscale_phase_b2"),
    "freqscale_phase_b3": make_trivial_negpos("freqscale_phase_b3"),
    "freqscale_phase_b4": make_trivial_negpos("freqscale_phase_b4"),
    "freqscale_phase_b5": make_trivial_negpos("freqscale_phase_b5"),
    "freqscale_phase_b6": make_trivial_negpos("freqscale_phase_b6"),
    "freqscale_phase_b7": make_trivial_negpos("freqscale_phase_b7"),
    "freqscale_phase_b8": make_trivial_negpos("freqscale_phase_b8"),
    "freqscale_phase_b9": make_trivial_negpos("freqscale_phase_b9"),
    "freqscale_phase_b10": make_trivial_negpos("freqscale_phase_b10"),

    "freqscaleshift_phase_a": make_trivial_negpos("freqscaleshift_phase_a"),
    "freqscaleshift_phase_b0": make_trivial_negpos("freqscaleshift_phase_b0"),
    "freqscaleshift_phase_b1": make_trivial_negpos("freqscaleshift_phase_b1"),
    "freqscaleshift_phase_b2": make_trivial_negpos("freqscaleshift_phase_b2"),
    "freqscaleshift_phase_b3": make_trivial_negpos("freqscaleshift_phase_b3"),
    "freqscaleshift_phase_b4": make_trivial_negpos("freqscaleshift_phase_b4"),
    "freqscaleshift_phase_b5": make_trivial_negpos("freqscaleshift_phase_b5"),
    "freqscaleshift_phase_b6": make_trivial_negpos("freqscaleshift_phase_b6"),
    "freqscaleshift_phase_b7": make_trivial_negpos("freqscaleshift_phase_b7"),
    "freqscaleshift_phase_b8": make_trivial_negpos("freqscaleshift_phase_b8"),
    "freqscaleshift_phase_b9": make_trivial_negpos("freqscaleshift_phase_b9"),
    "freqscaleshift_phase_b10": make_trivial_negpos("freqscaleshift_phase_b10"),

    # "jumpmean_a": make_trivial_negpos("jumpmean_a"),
    # "jumpmean_b0": make_trivial_negpos("jumpmean_b0"),
    # "jumpmean_b1": make_trivial_negpos("jumpmean_b1"),
    # "jumpmean_b2": make_trivial_negpos("jumpmean_b2"),
    # "jumpmean_b3": make_trivial_negpos("jumpmean_b3"),
    # "jumpmean_b4": make_trivial_negpos("jumpmean_b4"),
    # "jumpmean_b5": make_trivial_negpos("jumpmean_b5"),
    # "jumpmean_phase_a": make_trivial_negpos("jumpmean_phase_a"),
    # "jumpmean_phase_b0": make_trivial_negpos("jumpmean_phase_b0"),
    # "jumpmean_phase_b1": make_trivial_negpos("jumpmean_phase_b1"),
    # "jumpmean_phase_b2": make_trivial_negpos("jumpmean_phase_b2"),
    # "jumpmean_phase_b3": make_trivial_negpos("jumpmean_phase_b3"),
    # "jumpmean_phase_b4": make_trivial_negpos("jumpmean_phase_b4"),
    # "jumpmean_phase_b5": make_trivial_negpos("jumpmean_phase_b5"),

    # "rotate_phase_a": make_trivial2d("rotate_phase_a"),
    # "rotate_phase_b0": make_trivial2d("rotate_phase_b0"),
    # "rotate_phase_b1": make_trivial2d("rotate_phase_b1"),
    # "rotate_phase_b2": make_trivial2d("rotate_phase_b2"),
    # "rotate_phase_b3": make_trivial2d("rotate_phase_b3"),
    # "rotate_phase_b4": make_trivial2d("rotate_phase_b4"),
    # "rotate_phase_b5": make_trivial2d("rotate_phase_b5"),
    # "rotate_phase_b6": make_trivial2d("rotate_phase_b6"),
    # "rotate_phase_b7": make_trivial2d("rotate_phase_b7"),
    # "rotate_phase_b8": make_trivial2d("rotate_phase_b8"),
    # "rotate_phase_b9": make_trivial2d("rotate_phase_b9"),
    # "rotate_phase_b10": make_trivial2d("rotate_phase_b10"),

    "rotate2_phase_a": make_trivial2d("rotate2_phase_a"),
    "rotate2_phase_b0": make_trivial2d("rotate2_phase_b0"),
    "rotate2_phase_b1": make_trivial2d("rotate2_phase_b1"),
    "rotate2_phase_b2": make_trivial2d("rotate2_phase_b2"),
    "rotate2_phase_b3": make_trivial2d("rotate2_phase_b3"),
    "rotate2_phase_b4": make_trivial2d("rotate2_phase_b4"),
    "rotate2_phase_b5": make_trivial2d("rotate2_phase_b5"),
    "rotate2_phase_b6": make_trivial2d("rotate2_phase_b6"),
    "rotate2_phase_b7": make_trivial2d("rotate2_phase_b7"),
    "rotate2_phase_b8": make_trivial2d("rotate2_phase_b8"),
    "rotate2_phase_b9": make_trivial2d("rotate2_phase_b9"),
    "rotate2_phase_b10": make_trivial2d("rotate2_phase_b10"),

    # "rotate2_noise_a": make_trivial2d("rotate2_noise_a"),
    # "rotate2_noise_b0": make_trivial2d("rotate2_noise_b0"),
    # "rotate2_noise_b1": make_trivial2d("rotate2_noise_b1"),
    # "rotate2_noise_b2": make_trivial2d("rotate2_noise_b2"),
    # "rotate2_noise_b3": make_trivial2d("rotate2_noise_b3"),
    # "rotate2_noise_b4": make_trivial2d("rotate2_noise_b4"),
    # "rotate2_noise_b5": make_trivial2d("rotate2_noise_b5"),
    # "rotate2_noise_b6": make_trivial2d("rotate2_noise_b6"),
    # "rotate2_noise_b7": make_trivial2d("rotate2_noise_b7"),
    # "rotate2_noise_b8": make_trivial2d("rotate2_noise_b8"),
    # "rotate2_noise_b9": make_trivial2d("rotate2_noise_b9"),
    # "rotate2_noise_b10": make_trivial2d("rotate2_noise_b10"),

    "freqshiftrotate_phase_a": make_trivial2d("freqshiftrotate_phase_a"),
    "freqshiftrotate_phase_b0": make_trivial2d("freqshiftrotate_phase_b0"),
    "freqshiftrotate_phase_b1": make_trivial2d("freqshiftrotate_phase_b1"),
    "freqshiftrotate_phase_b2": make_trivial2d("freqshiftrotate_phase_b2"),
    "freqshiftrotate_phase_b3": make_trivial2d("freqshiftrotate_phase_b3"),
    "freqshiftrotate_phase_b4": make_trivial2d("freqshiftrotate_phase_b4"),
    "freqshiftrotate_phase_b5": make_trivial2d("freqshiftrotate_phase_b5"),
    "freqshiftrotate_phase_b6": make_trivial2d("freqshiftrotate_phase_b6"),
    "freqshiftrotate_phase_b7": make_trivial2d("freqshiftrotate_phase_b7"),
    "freqshiftrotate_phase_b8": make_trivial2d("freqshiftrotate_phase_b8"),
    "freqshiftrotate_phase_b9": make_trivial2d("freqshiftrotate_phase_b9"),
    "freqshiftrotate_phase_b10": make_trivial2d("freqshiftrotate_phase_b10"),

    "freqscalerotate_phase_a": make_trivial2d("freqscalerotate_phase_a"),
    "freqscalerotate_phase_b0": make_trivial2d("freqscalerotate_phase_b0"),
    "freqscalerotate_phase_b1": make_trivial2d("freqscalerotate_phase_b1"),
    "freqscalerotate_phase_b2": make_trivial2d("freqscalerotate_phase_b2"),
    "freqscalerotate_phase_b3": make_trivial2d("freqscalerotate_phase_b3"),
    "freqscalerotate_phase_b4": make_trivial2d("freqscalerotate_phase_b4"),
    "freqscalerotate_phase_b5": make_trivial2d("freqscalerotate_phase_b5"),
    "freqscalerotate_phase_b6": make_trivial2d("freqscalerotate_phase_b6"),
    "freqscalerotate_phase_b7": make_trivial2d("freqscalerotate_phase_b7"),
    "freqscalerotate_phase_b8": make_trivial2d("freqscalerotate_phase_b8"),
    "freqscalerotate_phase_b9": make_trivial2d("freqscalerotate_phase_b9"),
    "freqscalerotate_phase_b10": make_trivial2d("freqscalerotate_phase_b10"),

    "freqscaleshiftrotate_phase_a": make_trivial2d("freqscaleshiftrotate_phase_a"),
    "freqscaleshiftrotate_phase_b0": make_trivial2d("freqscaleshiftrotate_phase_b0"),
    "freqscaleshiftrotate_phase_b1": make_trivial2d("freqscaleshiftrotate_phase_b1"),
    "freqscaleshiftrotate_phase_b2": make_trivial2d("freqscaleshiftrotate_phase_b2"),
    "freqscaleshiftrotate_phase_b3": make_trivial2d("freqscaleshiftrotate_phase_b3"),
    "freqscaleshiftrotate_phase_b4": make_trivial2d("freqscaleshiftrotate_phase_b4"),
    "freqscaleshiftrotate_phase_b5": make_trivial2d("freqscaleshiftrotate_phase_b5"),
    "freqscaleshiftrotate_phase_b6": make_trivial2d("freqscaleshiftrotate_phase_b6"),
    "freqscaleshiftrotate_phase_b7": make_trivial2d("freqscaleshiftrotate_phase_b7"),
    "freqscaleshiftrotate_phase_b8": make_trivial2d("freqscaleshiftrotate_phase_b8"),
    "freqscaleshiftrotate_phase_b9": make_trivial2d("freqscaleshiftrotate_phase_b9"),
    "freqscaleshiftrotate_phase_b10": make_trivial2d("freqscaleshiftrotate_phase_b10"),
}


# Get datasets
def load(name, *args, **kwargs):
    """ Load a dataset based on the name (must be one of datasets.names()) """
    assert name in datasets.keys(), "Name specified not in datasets.names()"
    return datasets[name](*args, **kwargs)


def load_da(source_name, target_name, *args, **kwargs):
    """ Load two datasets (source and target) but perform necessary conversions
    to make them compatable for adaptation (i.e. same size, channels, etc.).
    Names must be in datasets.names()."""

    # No conversions, resizes, etc.
    source_dataset = load(source_name, *args, **kwargs)

    if target_name is not None:
        target_dataset = load(target_name, *args, **kwargs)
    else:
        target_dataset = None

    return source_dataset, target_dataset


# Get names
def names():
    """
    Returns list of all the available datasets to load with datasets.load(name)
    """
    return list(datasets.keys())


def main(argv):
    sd, td = load_da("utdata_wrist", "utdata_pocket")

    print("Source")
    print(sd.train_data, sd.train_labels)
    print(sd.train_data.shape, sd.train_labels.shape)
    print(sd.test_data, sd.test_labels)
    print(sd.test_data.shape, sd.test_labels.shape)
    print("Target")
    print(td.train_data, td.train_labels)
    print(td.train_data.shape, td.train_labels.shape)
    print(td.test_data, td.test_labels)
    print(td.test_data.shape, td.test_labels.shape)


if __name__ == "__main__":
    app.run(main)
