"""
Datasets

Load the desired datasets into memory so we can write them to tfrecord files
in generate_tfrecords.py
"""
import os
import re
import io
import zipfile
import tarfile
import rarfile  # pip install rarfile
import scipy.io
import numpy as np
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
            return train_data, train_labels, train_domain, \
                test_data, test_labels, test_domain

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
        train_data, train_labels, self.train_domain, \
            test_data, test_labels, self.test_domain = self.load()

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
        return data, labels

    def train_test_split(self, x, y, domain, random_state=42):
        """
        Split x and y data into train/test sets

        Warning: train_test_split() is from sklearn but self.train_test_split()
        is this function, which is what you should use.
        """
        x_train, x_test, y_train, y_test, domain_train, domain_test = \
            train_test_split(x, y, domain, test_size=self.test_percent,
            stratify=y, random_state=random_state)
        return x_train, y_train, x_test, y_test, domain_train, domain_test

    def get_file_in_archive(self, archive, filename):
        """ Read one file out of the already-open zip/rar file """
        with archive.open(filename) as fp:
            contents = fp.read()
        return contents

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

    def pad_to(self, data, desired_length):
        """
        Pad the number of time steps to the desired length

        Accepts data in one of two formats:
            - shape: (time_steps, features) -> (desired_length, features)
            - shape: (batch_size, time_steps, features) ->
                (batch_size, desired_length, features)
        """
        if len(data.shape) == 2:
            current_length = data.shape[0]
            assert current_length <= desired_length, "Cannot shrink size by padding"
            return np.pad(data, [(0, desired_length - current_length), (0, 0)],
                    mode="constant", constant_values=0)
        elif len(data.shape) == 3:
            current_length = data.shape[1]
            assert current_length <= desired_length, "Cannot shrink size by padding"
            return np.pad(data, [(0, 0), (0, desired_length - current_length), (0, 0)],
                    mode="constant", constant_values=0)
        else:
            raise NotImplementedError("pad_to requires 2 or 3-dim data")

    def label_to_int(self, label_name):
        """ e.g. Bathe to 0 """
        return self.class_labels.index(label_name)

    def int_to_label(self, label_index):
        """ e.g. Bathe to 0 """
        return self.class_labels[label_index]


def one_to_n(n):
    """ Return [1, 2, 3, ..., n] """
    return list(range(1, n+1))


def calc_domains(users, is_target):
    """
    Returns [list of domains...] based on the number of users and
    if this is the target or not
    """
    if is_target:
        assert len(users) == 1, "cannot have more than one target"
        return [0]

    return one_to_n(len(users))


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

    def __init__(self, days, users, num_classes, class_labels, target,
            *args, **kwargs):
        self.days = days
        self.users = users
        self.domains = calc_domains(days or users, target)
        super().__init__(num_classes, class_labels, None, None,
            uWaveBase.feature_names, *args, **kwargs)

    def download(self):
        (dataset_fp,) = self.download_dataset(["uWaveGestureLibrary.zip"],
            "https://zhen-wang.appspot.com/rice/files/uwave")
        return dataset_fp

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
        domain = []

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
                            #print("Skipping user", user)
                            continue

                    if self.days is not None:
                        if day not in self.days:
                            #print("Skipping day", day)
                            continue

                    #print("Processing user", user, "day", day)
                    contents = self.get_file_in_archive(archive, f)
                    new_data, new_labels = self.load_rar(io.BytesIO(contents))
                    data += new_data
                    labels += new_labels

                    # Which domain this is for
                    if self.users is not None:
                        domain_index = self.users.index(user)
                    elif self.days is not None:
                        domain_index = self.days.index(day)
                    else:
                        domain_index = 0

                    domain += [self.domains[domain_index]]*len(new_labels)

        # Zero pad (appending zeros) to make all the same shape
        # for uwave_all, we know the max max([x.shape[0] for x in data]) = 315
        # and expand the dimensions to [1, time_steps, num_features] so we can
        # vstack them properly
        #data = [np.expand_dims(self.pad_to(d, 315), axis=0) for d in data]

        #x = np.vstack(data).astype(np.float32)
        y = np.hstack(labels).astype(np.float32)
        domain = np.hstack(domain).astype(np.float32)

        return data, y, domain

    def load(self):
        # Load data
        dataset_fp = self.download()
        x, y, domain = self.load_zip(dataset_fp)
        # Split into train/test sets
        train_data, train_labels, test_data, test_labels, train_domain, test_domain = \
            self.train_test_split(x, y, domain)

        # Normalize here since we know which data is train vs. test and we have
        # to normalize before we zero pad or the zero padding messes up the
        # mean calculation a lot
        if FLAGS.normalize != "none":
            normalization = calc_normalization_jagged(train_data, FLAGS.normalize)
            train_data = apply_normalization_jagged(train_data, normalization)
            test_data = apply_normalization_jagged(test_data, normalization)

        # Then zero-pad to be the right length
        train_data = np.vstack([np.expand_dims(self.pad_to(d, 315), axis=0)
            for d in train_data]).astype(np.float32)
        test_data = np.vstack([np.expand_dims(self.pad_to(d, 315), axis=0)
            for d in test_data]).astype(np.float32)

        return train_data, train_labels, train_domain, \
            test_data, test_labels, test_domain

    def process(self, data, labels):
        """ uWave classes are index-one """
        # Check we have data in [examples, time_steps, 3]
        assert len(data.shape) == 3, "should shape [examples, time_steps, 3]"
        assert data.shape[2] == 3, "should have 3 features"

        # Index one
        labels = labels - 1
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

    def __init__(self, days, users, num_classes, class_labels, target,
            *args, **kwargs):
        self.days = days
        self.users = users
        self.domains = calc_domains(days or users, target)
        super().__init__(num_classes, class_labels, None, None,
            SleepBase.feature_names, *args, **kwargs)

    def download(self):
        (dataset_fp,) = self.download_dataset(["uWaveGestureLibrary.zip"],
            "https://zhen-wang.appspot.com/rice/files/uwave")
        return dataset_fp

    def process_examples(self, filename, fp):
        d = np.load(fp, allow_pickle=True).item()
        day = int(filename.replace(".npy", ""))
        user = d["subject"]

        # Skip data we don't want
        if self.days is not None:
            if day not in self.days:
                #print("Skipping day", day)
                return None, None, None

        if self.users is not None:
            if user not in self.users:
                #print("Skipping user", user)
                return None, None, None

        #print("Processing user", user, "day", day)

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

        # Which domain this is for
        if self.users is not None:
            domain_index = self.users.index(user)
        elif self.days is not None:
            domain_index = self.days.index(day)
        else:
            domain_index = 0

        domain = [self.domains[domain_index]]*len(stage_labels)

        return x, stage_labels, domain

    def load_file(self, filename):
        """ Load ZIP file containing all the .npy files """
        if not os.path.exists(filename):
            print("Download unencrypted "+filename+" into the current directory")

        data = []
        labels = []
        domains = []

        with zipfile.ZipFile(filename, "r") as archive:
            filelist = archive.namelist()

            for f in filelist:
                if ".npy" in f:
                    contents = self.get_file_in_archive(archive, f)
                    x, label, domain = self.process_examples(f, io.BytesIO(contents))

                    if x is not None and label is not None and domain is not None:
                        data.append(x)
                        labels.append(label)
                        domains.append(domain)

        x = np.vstack(data).astype(np.float32)
        y = np.hstack(labels).astype(np.float32)
        domains = np.hstack(domains).astype(np.float32)

        return x, y, domains

    def load(self):
        x, y, domain = self.load_file("RFSleep_unencrypted.zip")
        train_data, train_labels, test_data, test_labels, train_domain, test_domain = \
            self.train_test_split(x, y, domain)

        return train_data, train_labels, train_domain, \
            test_data, test_labels, test_domain


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

    def __init__(self, users, target, *args, **kwargs):
        self.users = users
        self.domains = calc_domains(users, target)
        super().__init__(UciHarBase.num_classes, UciHarBase.class_labels,
            None, None, UciHarBase.feature_names, *args, **kwargs)

    def download(self):
        (dataset_fp,) = self.download_dataset(["UCI%20HAR%20Dataset.zip"],
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00240")
        return dataset_fp

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
        labels = np.squeeze(np.array(labels, dtype=np.float32))
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
        all_labels = np.hstack([train_labels, test_labels]).astype(np.float32)
        all_subjects = np.hstack([train_subjects, test_subjects]).astype(np.float32)

        # All data if no selection
        if self.users is None:
            raise NotImplementedError("currently you must select a subset of users")
            # TODO create domain list for all users
            #return all_data, all_labels

        # Otherwise, select based on the desired users
        data = []
        labels = []
        domains = []

        for user in self.users:
            selection = all_subjects == user
            data.append(all_data[selection])
            current_labels = all_labels[selection]
            labels.append(current_labels)

            domain_index = self.users.index(user)
            domains.append([self.domains[domain_index]]*len(current_labels))

        x = np.vstack(data).astype(np.float32)
        y = np.hstack(labels).astype(np.float32)
        domains = np.hstack(domains).astype(np.float32)

        # print("Selected data:", self.users)
        # print(x.shape, y.shape, domains.shape)

        return x, y, domains

    def load(self):
        dataset_fp = self.download()
        x, y, domain = self.load_file(dataset_fp)
        train_data, train_labels, test_data, test_labels, train_domain, test_domain = \
            self.train_test_split(x, y, domain)

        return train_data, train_labels, train_domain, \
            test_data, test_labels, test_domain

    def process(self, data, labels):
        # Index one
        labels = labels - 1
        return super().process(data, labels)


class UciHHarBase(Dataset):
    """
    Loads Heterogeneity Human Activity Recognition (HHAR) dataset
    http://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition
    """
    invertible = False
    feature_names = [
        "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z",
    ]
    num_classes = 6
    class_labels = [
        "bike", "sit", "stand", "walk", "stairsup", "stairsdown",
    ]  # we throw out "null"
    window_size = 128  # to be relatively similar to HAR
    window_overlap = False

    def __init__(self, users, target, *args, **kwargs):
        self.users = users
        self.domains = calc_domains(users, target)
        super().__init__(UciHHarBase.num_classes, UciHHarBase.class_labels,
            UciHHarBase.window_size, UciHHarBase.window_overlap,
            UciHHarBase.feature_names, *args, **kwargs)

    def download(self):
        (dataset_fp,) = self.download_dataset(["Activity%20recognition%20exp.zip"],
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00344/")
        return dataset_fp

    def read_file(self, content):
        """ Read the CSV file """
        lines = content.decode("utf-8").strip().split("\n")
        data_x = []
        data_label = []
        data_subject = []
        users = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

        for line in lines:
            index, arrival, creation, x, y, z, user, \
                model, device, label = line.strip().split(",")

            # Skip the header (can't determine user if invalid)
            if index == "Index":
                continue

            user = users.index(user)  # letter --> number

            # Skip users we don't care about and data without a label
            if user in self.users and label != "null":
                #index = int(index)
                #arrival = float(arrival)
                #creation = float(creation)
                x = float(x)
                y = float(y)
                z = float(z)
                label = self.class_labels.index(label)  # name --> number

                data_x.append((x, y, z))
                data_label.append(label)
                data_subject.append(user)

        data_x = np.array(data_x, dtype=np.float32)
        data_label = np.array(data_label, dtype=np.float32)
        data_subject = np.array(data_subject, dtype=np.float32)

        return data_x, data_label, data_subject

    def get_data(self, archive, name):
        # In their paper, looks like they only did either accelerometer or
        # gyroscope, not aligning them by the creation timestamp. For them the
        # accelerometer data worked better, so we'll just use that for now.
        return self.read_file(self.get_file_in_archive(archive,
                "Activity recognition exp/"+name+"_accelerometer.csv"))

    def load_file(self, filename):
        """ Load ZIP file containing all the .txt files """
        with zipfile.ZipFile(filename, "r") as archive:
            # For now just use phone data since the positions may differ too much
            all_data, all_labels, all_subjects = self.get_data(archive, "Phones")

            # phone_data, phone_labels, phone_subjects = self.get_data(archive, "Phone")
            # watch_data, watch_labels, watch_subjects = self.get_data(archive, "Watch")

        # all_data = np.vstack([phone_data, watch_data]).astype(np.float32)
        # all_labels = np.hstack([phone_labels, watch_labels]).astype(np.float32)
        # all_subjects = np.hstack([phone_subjects, watch_subjects]).astype(np.float32)

        # Otherwise, select based on the desired users
        data = []
        labels = []
        domains = []

        for user in self.users:
            # Load this user's data
            selection = all_subjects == user
            current_data = all_data[selection]
            current_labels = all_labels[selection]
            assert len(current_labels) > 0, "Error: no data for user "+str(user)

            # Split into windows
            current_data, current_labels = self.create_windows(current_data,
                current_labels, self.window_size, self.window_overlap)

            # Save
            data.append(current_data)
            labels.append(current_labels)

            domain_index = self.users.index(user)
            domains.append([self.domains[domain_index]]*len(current_labels))

        x = np.vstack(data).astype(np.float32)
        y = np.hstack(labels).astype(np.float32)
        domains = np.hstack(domains).astype(np.float32)

        # print("Selected data:", self.users)
        # print(x.shape, y.shape, domains.shape)

        return x, y, domains

    def load(self):
        dataset_fp = self.download()
        x, y, domain = self.load_file(dataset_fp)
        train_data, train_labels, test_data, test_labels, train_domain, test_domain = \
            self.train_test_split(x, y, domain)

        return train_data, train_labels, train_domain, \
            test_data, test_labels, test_domain


class UciHmBase(Dataset):
    """
    Loads sEMG for Basic Hand movements dataset
    http://archive.ics.uci.edu/ml/datasets/sEMG+for+Basic+Hand+movements
    """
    invertible = False
    feature_names = [
        "ch1", "ch2",
    ]
    num_classes = 6
    class_labels = [
        "spher", "tip", "palm", "lat", "cyl", "hook",
    ]
    window_size = 500  # 500 Hz, so 1 second
    window_overlap = False  # Note: using np.hsplit, so this has no effect

    def __init__(self, users, target, split=True, pad=True, subsample=True,
            *args, **kwargs):
        self.split = split
        # Only apply if split=False
        self.pad = pad
        self.subsample = subsample

        self.users = users
        self.domains = calc_domains(users, target)
        super().__init__(UciHmBase.num_classes, UciHmBase.class_labels,
            UciHmBase.window_size, UciHmBase.window_overlap,
            UciHmBase.feature_names, *args, **kwargs)

    def download(self):
        (dataset_fp,) = self.download_dataset(["sEMG_Basic_Hand_movements_upatras.zip"],
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00313/")
        return dataset_fp

    def get_data(self, archive, filename):
        """ Open .mat file in zip file, then load contents"""
        with archive.open(filename) as fp:
            mat = scipy.io.loadmat(fp)

            data_x = []
            data_label = []

            for label_index, label in enumerate(self.class_labels):
                # Concatenate the channels
                data = []

                for channel in self.feature_names:
                    data.append(mat[label+"_"+channel])

                # Reshape from (2, 30, 3000) to (30, 3000, 2) for 30 examples,
                # 3000 time steps, and 2 features
                data = np.array(data, dtype=np.float32).transpose([1, 2, 0])

                # Split from 3000 or 2500 time steps into 6 or 5 non-overlapping
                # windows of 500 samples. Duplicate the label 6 or 5 times,
                # respectively.
                if self.split:
                    assert data.shape[1] in [2500, 3000], "data.shape[1] should " \
                        "be 2500 or 3000, but is "+str(data.shape[1])
                    num_windows = data.shape[1]//self.window_size
                    data_x += np.hsplit(data, num_windows)
                else:
                    # Since we have to concatenate multiple domains, pad with
                    # zeros to get them to all be the same number of time steps
                    if self.pad:
                        data = self.pad_to(data, 3000)

                    # It's really slow with 3000 samples, and 500 Hz is probably
                    # overkill, so subsample
                    if self.subsample:
                        data = data[:, ::6, :]  # 3000 -> 500, fewer samples

                    data_x.append(data)
                    num_windows = 1

                data_label += [label_index]*len(data)*num_windows

            data_x = np.vstack(data_x).astype(np.float32)
            data_label = np.array(data_label, dtype=np.float32)

            return data_x, data_label

    def load_file(self, filename):
        """
        Load desired participants' data

        Numbering:
            0 - Database 1/female_1.mat
            1 - Database 1/female_2.mat
            2 - Database 1/female_3.mat
            3 - Database 1/male_1.mat
            4 - Database 1/male_2.mat
            5 - Database 2/male_day_{1,2,3}.mat
        """
        data = []
        labels = []
        domains = []

        with zipfile.ZipFile(filename, "r") as archive:
            for user in self.users:
                # Load this user's data
                if user != 5:
                    gender = "female" if user < 3 else "male"
                    index = user+1 if user < 3 else user-2
                    current_data, current_labels = self.get_data(archive,
                        "Database 1/"+gender+"_"+str(index)+".mat")
                else:
                    data1, labels1 = self.get_data(archive, "Database 2/male_day_1.mat")
                    data2, labels2 = self.get_data(archive, "Database 2/male_day_2.mat")
                    data3, labels3 = self.get_data(archive, "Database 2/male_day_3.mat")
                    current_data = np.vstack([data1, data2, data3]).astype(np.float32)
                    current_labels = np.hstack([labels1, labels2, labels3]).astype(np.float32)

                # Save
                data.append(current_data)
                labels.append(current_labels)

                domain_index = self.users.index(user)
                domains.append([self.domains[domain_index]]*len(current_labels))

        x = np.vstack(data).astype(np.float32)
        y = np.hstack(labels).astype(np.float32)
        domains = np.hstack(domains).astype(np.float32)

        # print("Selected data:", self.users)
        # print(x.shape, y.shape, domains.shape)

        return x, y, domains

    def load(self):
        dataset_fp = self.download()
        x, y, domain = self.load_file(dataset_fp)
        train_data, train_labels, test_data, test_labels, train_domain, test_domain = \
            self.train_test_split(x, y, domain)

        return train_data, train_labels, train_domain, \
            test_data, test_labels, test_domain


class WisdmBase(Dataset):
    """
    Loads Actitracker dataset
    http://www.cis.fordham.edu/wisdm/dataset.php#actitracker (note: click
    on Actitracker link on left menu)
    """
    invertible = False
    feature_names = [
        "acc_x", "acc_y", "acc_z",
    ]
    num_classes = 6
    class_labels = [
        "Walking", "Jogging", "Stairs", "Sitting", "Standing", "LyingDown",
    ]
    window_size = 128  # similar to HAR
    window_overlap = False

    def __init__(self, users, target, *args, **kwargs):
        self.users = users
        self.domains = calc_domains(users, target)
        super().__init__(WisdmBase.num_classes, WisdmBase.class_labels,
            WisdmBase.window_size, WisdmBase.window_overlap,
            WisdmBase.feature_names, *args, **kwargs)

    def download(self):
        (dataset_fp,) = self.download_dataset(["WISDM_at_latest.tar.gz"],
            "http://www.cis.fordham.edu/wisdm/includes/datasets/latest/")
        return dataset_fp

    def read_data(self, lines, user_list):
        """ Read the raw data CSV file """
        data_x = []
        data_label = []
        data_subject = []

        for line in lines:
            parts = line.strip().replace(";", "").split(",")

            # For some reason there's blank rows in the data, e.g.
            # a bunch of lines like "577,,;"
            if len(parts) != 6:
                continue

            user, activity, timestamp, x, y, z = parts
            user = int(user)

            # Skip users that may not have enough data
            if user in user_list:
                user = user_list.index(user)  # non-consecutive to consecutive

                # Skip users we don't care about
                if user in self.users:
                    x = float(x)
                    y = float(y)
                    z = float(z)
                    label = self.class_labels.index(activity)  # name --> number

                    data_x.append((x, y, z))
                    data_label.append(label)
                    data_subject.append(user)

        data_x = np.array(data_x, dtype=np.float32)
        data_label = np.array(data_label, dtype=np.float32)
        data_subject = np.array(data_subject, dtype=np.float32)

        return data_x, data_label, data_subject

    def read_user_list(self, lines):
        """ Read first column of the CSV file to get a unique list of uid's
        Also, skip users with too few samples """
        user_sample_count = {}

        for line in lines:
            parts = line.strip().split(",")
            uid = int(parts[0])

            # There are duplicates in the file for some reason (so, either the
            # same person or it's not truly unique)
            if uid not in user_sample_count:
                user_sample_count[uid] = 0
            else:
                user_sample_count[uid] += 1

        # Remove users with too few samples
        user_list = []

        # How many samples we need: to stratify the sklearn function says
        # The test_size = A should be greater or equal to the number of classes = B
        # x/128*.2 > 6
        # x > 6*128/.2
        test_percentage = 0.20  # default
        min_samples = int(len(self.class_labels)*self.window_size/test_percentage)

        for user, count in user_sample_count.items():
            if count > min_samples:
                user_list.append(user)

        # Data isn't sorted by user in the file
        user_list.sort()

        return user_list

    def get_lines(self, archive, name):
        """ Open and load file in tar file, get lines from file """
        f = archive.extractfile(
            "home/share/data/public_sets/WISDM_at_v2.0/WISDM_at_v2.0_"+name)

        if f is None:
            return None

        return f.read().decode("utf-8").strip().split("\n")

    def load_file(self, filename):
        """
        Load desired participants' data
        """
        # Get data
        with tarfile.open(filename, "r") as archive:
            raw_data = self.get_lines(archive, "raw.txt")

        # Some of the data doesn't have a uid in the demographics file? So,
        # instead just get the user list from the raw data. Also, one person
        # have very little data, so skip them (e.g. one person only has 25
        # samples, which is only 0.5 seconds of data -- not useful).
        user_list = self.read_user_list(raw_data)

        # For now just use phone data since the positions may differ too much
        all_data, all_labels, all_subjects = self.read_data(raw_data, user_list)

        # Otherwise, select based on the desired users
        data = []
        labels = []
        domains = []

        for user in self.users:
            # Load this user's data
            selection = all_subjects == user
            current_data = all_data[selection]
            current_labels = all_labels[selection]
            assert len(current_labels) > 0, "Error: no data for user "+str(user)

            # Split into windows
            current_data, current_labels = self.create_windows(current_data,
                current_labels, self.window_size, self.window_overlap)

            # Save
            data.append(current_data)
            labels.append(current_labels)

            domain_index = self.users.index(user)
            domains.append([self.domains[domain_index]]*len(current_labels))

        x = np.vstack(data).astype(np.float32)
        y = np.hstack(labels).astype(np.float32)
        domains = np.hstack(domains).astype(np.float32)

        # print("Selected data:", self.users)
        # print(x.shape, y.shape, domains.shape)

        return x, y, domains

    def load(self):
        dataset_fp = self.download()
        x, y, domain = self.load_file(dataset_fp)
        train_data, train_labels, test_data, test_labels, train_domain, test_domain = \
            self.train_test_split(x, y, domain)

        return train_data, train_labels, train_domain, \
            test_data, test_labels, test_domain


def make_uwave(days=None, users=None, target=False):
    """ Make uWave dataset split on either days or users """
    class uWaveGestures(uWaveBase):
        invertible = False
        num_classes = 8
        class_labels = list(range(num_classes))

        # If source domain, number is source domains + 1 for target
        # Set here to make it static
        num_domains = len(users or days)+1 if not target else None

        def __init__(self, *args, **kwargs):
            super().__init__(
                days, users,
                uWaveGestures.num_classes,
                uWaveGestures.class_labels,
                target, *args, **kwargs)

    return uWaveGestures


def make_sleep(days=None, users=None, target=False):
    """ Make RF sleep dataset split on either days or users """
    class SleepDataset(SleepBase):
        invertible = False
        num_classes = 6
        class_labels = ["Awake", "N1", "N2", "N3", "Light N2", "REM"]

        # If source domain, number is source domains + 1 for target
        # Set here to make it static
        num_domains = len(users or days)+1 if not target else None

        def __init__(self, *args, **kwargs):
            super().__init__(
                days, users,
                SleepDataset.num_classes,
                SleepDataset.class_labels,
                target, *args, **kwargs)

    return SleepDataset


def make_ucihar(users=None, target=False):
    """ Make UCI HAR dataset split on users """
    class UciHarDataset(UciHarBase):
        # If source domain, number is source domains + 1 for target
        # Set here to make it static
        num_domains = len(users)+1 if not target else None

        def __init__(self, *args, **kwargs):
            super().__init__(users, target, *args, **kwargs)

    return UciHarDataset


def make_ucihhar(users=None, target=False):
    """ Make UCI HHAR dataset split on users """
    class UciHHarDataset(UciHHarBase):
        # If source domain, number is source domains + 1 for target
        # Set here to make it static
        num_domains = len(users)+1 if not target else None

        def __init__(self, *args, **kwargs):
            super().__init__(users, target, *args, **kwargs)

    return UciHHarDataset


def make_ucihm(users=None, target=False):
    """ Make UCI HM dataset split on users """
    # More examples but each is only 1 second
    split = True

    class UciHmDataset(UciHmBase):
        # If source domain, number is source domains + 1 for target
        # Set here to make it static
        num_domains = len(users)+1 if not target else None

        def __init__(self, *args, **kwargs):
            super().__init__(users, target, split, *args, **kwargs)

    return UciHmDataset


def make_ucihm_full(users=None, target=False):
    """ Make UCI HM dataset split on users """
    # Fewer examples, but each is the full hand motion
    split = False

    class UciHmDataset(UciHmBase):
        # If source domain, number is source domains + 1 for target
        # Set here to make it static
        num_domains = len(users)+1 if not target else None

        def __init__(self, *args, **kwargs):
            super().__init__(users, target, split, *args, **kwargs)

    return UciHmDataset


def make_wisdm(users=None, target=False):
    """ Make WISDM Actitracker dataset split on users """
    class WisdmDataset(WisdmBase):
        # If source domain, number is source domains + 1 for target
        # Set here to make it static
        num_domains = len(users)+1 if not target else None

        def __init__(self, *args, **kwargs):
            super().__init__(users, target, *args, **kwargs)

    return WisdmDataset


# List of datasets, target separate from (multi-)source ones since the target
# always has domain=0 whereas the others have domain=1,2,... for each source
# See pick_multi_source.py
datasets = {
    "ucihar_n13_0": make_ucihar(users=[7,8,11,12,13,14,16,17,20,21,23,24,26]),
    "ucihar_n13_1": make_ucihar(users=[4,9,11,13,14,16,17,18,21,23,25,28,30]),
    "ucihar_n13_10": make_ucihar(users=[3,5,6,9,12,18,19,20,21,23,24,27,29]),
    "ucihar_n13_11": make_ucihar(users=[1,5,6,10,11,12,13,14,15,16,22,27,28]),
    "ucihar_n13_12": make_ucihar(users=[2,4,7,8,13,15,17,18,19,20,25,26,29]),
    "ucihar_n13_13": make_ucihar(users=[3,8,9,10,12,13,14,15,20,23,26,29,30]),
    "ucihar_n13_14": make_ucihar(users=[4,8,11,14,16,19,20,21,24,25,26,29,30]),
    "ucihar_n13_2": make_ucihar(users=[2,5,6,7,10,12,14,16,17,18,23,25,30]),
    "ucihar_n13_3": make_ucihar(users=[1,4,5,6,8,11,13,15,17,18,20,24,26]),
    "ucihar_n13_4": make_ucihar(users=[1,4,5,7,8,11,12,16,18,22,23,26,28]),
    "ucihar_n13_5": make_ucihar(users=[3,4,6,8,13,14,16,17,18,20,21,22,24]),
    "ucihar_n13_6": make_ucihar(users=[2,4,6,8,10,14,15,16,22,23,27,28,29]),
    "ucihar_n13_7": make_ucihar(users=[1,2,4,6,7,9,10,12,15,21,22,23,27]),
    "ucihar_n13_8": make_ucihar(users=[1,5,7,11,13,15,16,20,21,23,26,27,28]),
    "ucihar_n13_9": make_ucihar(users=[1,7,10,11,13,15,17,18,24,25,26,28,30]),
    "ucihar_n19_0": make_ucihar(users=[3,4,7,8,11,12,13,14,15,16,17,18,20,21,23,24,26,28,30]),
    "ucihar_n19_1": make_ucihar(users=[3,4,5,6,8,9,11,13,14,16,17,18,19,21,22,23,25,28,30]),
    "ucihar_n19_10": make_ucihar(users=[3,5,6,9,10,11,12,13,16,18,19,20,21,22,23,24,26,27,29]),
    "ucihar_n19_11": make_ucihar(users=[1,5,6,8,9,10,11,12,13,14,15,16,19,21,22,23,24,27,28]),
    "ucihar_n19_12": make_ucihar(users=[1,2,3,4,7,8,13,15,16,17,18,19,20,21,25,26,27,28,29]),
    "ucihar_n19_13": make_ucihar(users=[2,3,4,6,8,9,10,12,13,14,15,17,20,23,25,26,27,29,30]),
    "ucihar_n19_14": make_ucihar(users=[4,6,7,8,9,11,12,13,14,16,19,20,21,23,24,25,26,29,30]),
    "ucihar_n19_2": make_ucihar(users=[2,5,6,7,9,10,12,14,15,16,17,18,23,24,25,26,27,29,30]),
    "ucihar_n19_3": make_ucihar(users=[1,3,4,5,6,8,11,12,13,15,17,18,20,21,23,24,26,27,29]),
    "ucihar_n19_4": make_ucihar(users=[1,3,4,5,7,8,11,12,13,16,17,18,21,22,23,26,28,29,30]),
    "ucihar_n19_5": make_ucihar(users=[3,4,6,7,8,9,11,13,14,15,16,17,18,20,21,22,24,26,28]),
    "ucihar_n19_6": make_ucihar(users=[2,4,5,6,8,9,10,11,13,14,15,16,20,22,23,24,27,28,29]),
    "ucihar_n19_7": make_ucihar(users=[1,2,4,5,6,7,9,10,12,15,16,17,18,20,21,22,23,27,29]),
    "ucihar_n19_8": make_ucihar(users=[1,5,6,7,11,13,14,15,16,19,20,21,22,23,24,25,26,27,28]),
    "ucihar_n19_9": make_ucihar(users=[1,3,6,7,10,11,13,15,16,17,18,19,20,21,24,25,26,28,30]),
    "ucihar_n1_0": make_ucihar(users=[20]),
    "ucihar_n1_1": make_ucihar(users=[4]),
    "ucihar_n1_2": make_ucihar(users=[12]),
    "ucihar_n1_3": make_ucihar(users=[23]),
    "ucihar_n1_4": make_ucihar(users=[21]),
    "ucihar_n1_5": make_ucihar(users=[16]),
    "ucihar_n1_6": make_ucihar(users=[10]),
    "ucihar_n1_7": make_ucihar(users=[27]),
    "ucihar_n1_8": make_ucihar(users=[26]),
    "ucihar_n1_9": make_ucihar(users=[17]),
    "ucihar_n25_0": make_ucihar(users=[3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,26,27,28,29,30]),
    "ucihar_n25_1": make_ucihar(users=[3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,28,29,30]),
    "ucihar_n25_10": make_ucihar(users=[1,2,3,5,6,8,9,10,11,12,13,14,16,18,19,20,21,22,23,24,25,26,27,29,30]),
    "ucihar_n25_11": make_ucihar(users=[1,3,5,6,7,8,9,10,11,12,13,14,15,16,18,19,21,22,23,24,25,26,27,28,29]),
    "ucihar_n25_12": make_ucihar(users=[1,2,3,4,6,7,8,10,11,13,14,15,16,17,18,19,20,21,22,25,26,27,28,29,30]),
    "ucihar_n25_13": make_ucihar(users=[1,2,3,4,6,8,9,10,11,12,13,14,15,17,18,20,21,22,23,24,25,26,27,29,30]),
    "ucihar_n25_14": make_ucihar(users=[1,3,4,6,7,8,9,11,12,13,14,15,16,19,20,21,22,23,24,25,26,27,28,29,30]),
    "ucihar_n25_2": make_ucihar(users=[2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,20,21,23,24,25,26,27,29,30]),
    "ucihar_n25_3": make_ucihar(users=[1,3,4,5,6,8,10,11,12,13,14,15,16,17,18,20,21,22,23,24,26,27,28,29,30]),
    "ucihar_n25_4": make_ucihar(users=[1,3,4,5,7,8,10,11,12,13,14,15,16,17,18,20,21,22,23,24,26,27,28,29,30]),
    "ucihar_n25_5": make_ucihar(users=[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,24,25,26,27,28,30]),
    "ucihar_n25_6": make_ucihar(users=[1,2,4,5,6,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,27,28,29]),
    "ucihar_n25_7": make_ucihar(users=[1,2,4,5,6,7,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,27,28,29,30]),
    "ucihar_n25_8": make_ucihar(users=[1,5,6,7,8,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]),
    "ucihar_n25_9": make_ucihar(users=[1,2,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,24,25,26,27,28,30]),
    "ucihar_n7_0": make_ucihar(users=[7,8,12,16,20,23,24]),
    "ucihar_n7_1": make_ucihar(users=[4,9,14,17,18,21,23]),
    "ucihar_n7_10": make_ucihar(users=[6,18,20,21,23,24,27]),
    "ucihar_n7_11": make_ucihar(users=[6,10,13,16,22,27,28]),
    "ucihar_n7_12": make_ucihar(users=[7,8,17,18,19,20,25]),
    "ucihar_n7_13": make_ucihar(users=[8,9,14,20,23,26,30]),
    "ucihar_n7_14": make_ucihar(users=[4,11,19,20,24,26,29]),
    "ucihar_n7_2": make_ucihar(users=[2,5,10,12,14,16,25]),
    "ucihar_n7_3": make_ucihar(users=[1,4,13,18,20,24,26]),
    "ucihar_n7_4": make_ucihar(users=[5,7,11,18,23,26,28]),
    "ucihar_n7_5": make_ucihar(users=[8,16,17,18,20,21,24]),
    "ucihar_n7_6": make_ucihar(users=[6,8,10,14,15,16,29]),
    "ucihar_n7_7": make_ucihar(users=[4,6,9,12,21,22,27]),
    "ucihar_n7_8": make_ucihar(users=[1,7,15,20,23,26,27]),
    "ucihar_n7_9": make_ucihar(users=[1,7,10,15,18,24,25]),
    "ucihar_t1": make_ucihar(users=[1], target=True),
    "ucihar_t2": make_ucihar(users=[2], target=True),
    "ucihar_t3": make_ucihar(users=[3], target=True),
    "ucihar_t4": make_ucihar(users=[4], target=True),
    "ucihar_t5": make_ucihar(users=[5], target=True),
    "ucihhar_n1_0": make_ucihhar(users=[4]),
    "ucihhar_n1_1": make_ucihhar(users=[3]),
    "ucihhar_n1_2": make_ucihhar(users=[6]),
    "ucihhar_n1_3": make_ucihhar(users=[5]),
    "ucihhar_n1_4": make_ucihhar(users=[0]),
    "ucihhar_n1_5": make_ucihhar(users=[2]),
    "ucihhar_n1_6": make_ucihhar(users=[1]),
    "ucihhar_n2_0": make_ucihhar(users=[4,5]),
    "ucihhar_n2_1": make_ucihhar(users=[4,8]),
    "ucihhar_n2_10": make_ucihhar(users=[0,8]),
    "ucihhar_n2_11": make_ucihhar(users=[0,3]),
    "ucihhar_n2_2": make_ucihhar(users=[4,6]),
    "ucihhar_n2_3": make_ucihhar(users=[2,3]),
    "ucihhar_n2_4": make_ucihhar(users=[3,6]),
    "ucihhar_n2_5": make_ucihhar(users=[1,4]),
    "ucihhar_n2_6": make_ucihhar(users=[6,8]),
    "ucihhar_n2_7": make_ucihhar(users=[3,5]),
    "ucihhar_n2_8": make_ucihhar(users=[5,6]),
    "ucihhar_n2_9": make_ucihhar(users=[0,5]),
    "ucihhar_n3_0": make_ucihhar(users=[4,5,7]),
    "ucihhar_n3_1": make_ucihhar(users=[3,4,8]),
    "ucihhar_n3_10": make_ucihhar(users=[0,4,5]),
    "ucihhar_n3_11": make_ucihhar(users=[0,4,8]),
    "ucihhar_n3_12": make_ucihhar(users=[5,6,7]),
    "ucihhar_n3_13": make_ucihhar(users=[0,7,8]),
    "ucihhar_n3_2": make_ucihhar(users=[3,4,6]),
    "ucihhar_n3_3": make_ucihhar(users=[0,2,3]),
    "ucihhar_n3_4": make_ucihhar(users=[3,6,8]),
    "ucihhar_n3_5": make_ucihhar(users=[4,6,7]),
    "ucihhar_n3_6": make_ucihhar(users=[0,1,4]),
    "ucihhar_n3_7": make_ucihhar(users=[6,7,8]),
    "ucihhar_n3_8": make_ucihhar(users=[3,5,8]),
    "ucihhar_n3_9": make_ucihhar(users=[1,5,6]),
    "ucihhar_n4_0": make_ucihhar(users=[4,5,7,8]),
    "ucihhar_n4_1": make_ucihhar(users=[1,3,4,8]),
    "ucihhar_n4_10": make_ucihhar(users=[0,4,5,6]),
    "ucihhar_n4_11": make_ucihhar(users=[0,4,7,8]),
    "ucihhar_n4_12": make_ucihhar(users=[0,1,2,3]),
    "ucihhar_n4_13": make_ucihhar(users=[3,5,6,7]),
    "ucihhar_n4_14": make_ucihhar(users=[0,1,7,8]),
    "ucihhar_n4_2": make_ucihhar(users=[3,4,5,6]),
    "ucihhar_n4_3": make_ucihhar(users=[0,2,3,8]),
    "ucihhar_n4_4": make_ucihhar(users=[3,4,6,8]),
    "ucihhar_n4_5": make_ucihhar(users=[4,5,6,7]),
    "ucihhar_n4_6": make_ucihhar(users=[0,1,4,8]),
    "ucihhar_n4_7": make_ucihhar(users=[3,6,7,8]),
    "ucihhar_n4_8": make_ucihhar(users=[1,3,5,8]),
    "ucihhar_n4_9": make_ucihhar(users=[1,2,5,6]),
    "ucihhar_n5_0": make_ucihhar(users=[3,4,5,7,8]),
    "ucihhar_n5_1": make_ucihhar(users=[1,3,4,5,8]),
    "ucihhar_n5_10": make_ucihhar(users=[0,1,4,5,6]),
    "ucihhar_n5_11": make_ucihhar(users=[0,1,4,7,8]),
    "ucihhar_n5_12": make_ucihhar(users=[0,1,2,3,8]),
    "ucihhar_n5_13": make_ucihhar(users=[2,3,5,6,7]),
    "ucihhar_n5_14": make_ucihhar(users=[0,1,2,7,8]),
    "ucihhar_n5_2": make_ucihhar(users=[2,3,4,5,6]),
    "ucihhar_n5_3": make_ucihhar(users=[0,2,3,5,8]),
    "ucihhar_n5_4": make_ucihhar(users=[2,3,4,6,8]),
    "ucihhar_n5_5": make_ucihhar(users=[0,4,5,6,7]),
    "ucihhar_n5_6": make_ucihhar(users=[0,1,4,5,8]),
    "ucihhar_n5_7": make_ucihhar(users=[3,5,6,7,8]),
    "ucihhar_n5_8": make_ucihhar(users=[1,3,5,7,8]),
    "ucihhar_n5_9": make_ucihhar(users=[1,2,5,6,8]),
    "ucihhar_t0": make_ucihhar(users=[0], target=True),
    "ucihhar_t1": make_ucihhar(users=[1], target=True),
    "ucihhar_t2": make_ucihhar(users=[2], target=True),
    "ucihhar_t3": make_ucihhar(users=[3], target=True),
    "ucihhar_t4": make_ucihhar(users=[4], target=True),
    "uwave_n1_0": make_uwave(users=[3]),
    "uwave_n1_1": make_uwave(users=[5]),
    "uwave_n1_2": make_uwave(users=[4]),
    "uwave_n1_3": make_uwave(users=[8]),
    "uwave_n1_4": make_uwave(users=[7]),
    "uwave_n1_5": make_uwave(users=[6]),
    "uwave_n1_6": make_uwave(users=[2]),
    "uwave_n1_7": make_uwave(users=[1]),
    "uwave_n2_0": make_uwave(users=[3,5]),
    "uwave_n2_1": make_uwave(users=[5,8]),
    "uwave_n2_10": make_uwave(users=[5,6]),
    "uwave_n2_11": make_uwave(users=[1,8]),
    "uwave_n2_12": make_uwave(users=[4,8]),
    "uwave_n2_13": make_uwave(users=[6,8]),
    "uwave_n2_2": make_uwave(users=[4,5]),
    "uwave_n2_3": make_uwave(users=[4,7]),
    "uwave_n2_4": make_uwave(users=[1,7]),
    "uwave_n2_5": make_uwave(users=[6,7]),
    "uwave_n2_6": make_uwave(users=[4,6]),
    "uwave_n2_7": make_uwave(users=[1,2]),
    "uwave_n2_8": make_uwave(users=[3,6]),
    "uwave_n2_9": make_uwave(users=[1,5]),
    "uwave_n3_0": make_uwave(users=[3,5,6]),
    "uwave_n3_1": make_uwave(users=[4,5,8]),
    "uwave_n3_10": make_uwave(users=[1,5,6]),
    "uwave_n3_11": make_uwave(users=[4,6,8]),
    "uwave_n3_12": make_uwave(users=[1,6,8]),
    "uwave_n3_13": make_uwave(users=[6,7,8]),
    "uwave_n3_2": make_uwave(users=[4,5,6]),
    "uwave_n3_3": make_uwave(users=[1,4,7]),
    "uwave_n3_4": make_uwave(users=[5,7,8]),
    "uwave_n3_5": make_uwave(users=[1,5,7]),
    "uwave_n3_6": make_uwave(users=[2,6,7]),
    "uwave_n3_7": make_uwave(users=[2,4,6]),
    "uwave_n3_8": make_uwave(users=[1,2,5]),
    "uwave_n3_9": make_uwave(users=[3,6,7]),
    "uwave_n4_0": make_uwave(users=[3,4,5,6]),
    "uwave_n4_1": make_uwave(users=[4,5,6,8]),
    "uwave_n4_10": make_uwave(users=[1,5,6,8]),
    "uwave_n4_11": make_uwave(users=[2,5,6,8]),
    "uwave_n4_12": make_uwave(users=[1,3,6,8]),
    "uwave_n4_13": make_uwave(users=[2,4,6,8]),
    "uwave_n4_14": make_uwave(users=[2,6,7,8]),
    "uwave_n4_2": make_uwave(users=[2,4,5,7]),
    "uwave_n4_3": make_uwave(users=[3,5,7,8]),
    "uwave_n4_4": make_uwave(users=[1,5,6,7]),
    "uwave_n4_5": make_uwave(users=[3,4,6,7]),
    "uwave_n4_6": make_uwave(users=[2,4,6,7]),
    "uwave_n4_7": make_uwave(users=[1,2,5,7]),
    "uwave_n4_8": make_uwave(users=[2,4,5,6]),
    "uwave_n4_9": make_uwave(users=[3,5,6,7]),
    "uwave_n5_0": make_uwave(users=[3,4,5,6,8]),
    "uwave_n5_1": make_uwave(users=[2,4,5,6,8]),
    "uwave_n5_10": make_uwave(users=[1,2,5,6,8]),
    "uwave_n5_11": make_uwave(users=[1,2,4,6,8]),
    "uwave_n5_12": make_uwave(users=[1,3,6,7,8]),
    "uwave_n5_13": make_uwave(users=[1,2,6,7,8]),
    "uwave_n5_2": make_uwave(users=[3,4,5,6,7]),
    "uwave_n5_3": make_uwave(users=[1,4,5,7,8]),
    "uwave_n5_4": make_uwave(users=[1,3,5,7,8]),
    "uwave_n5_5": make_uwave(users=[1,3,5,6,7]),
    "uwave_n5_6": make_uwave(users=[2,4,6,7,8]),
    "uwave_n5_7": make_uwave(users=[2,4,5,6,7]),
    "uwave_n5_8": make_uwave(users=[1,2,5,7,8]),
    "uwave_n5_9": make_uwave(users=[2,3,5,6,8]),
    "uwave_t1": make_uwave(users=[1], target=True),
    "uwave_t2": make_uwave(users=[2], target=True),
    "uwave_t3": make_uwave(users=[3], target=True),
    "uwave_t4": make_uwave(users=[4], target=True),
    "uwave_t5": make_uwave(users=[5], target=True),
    "wisdm_n117_0": make_wisdm(users=[2,3,4,5,6,10,11,12,13,14,15,16,17,18,19,20,22,25,26,28,30,31,32,33,34,35,37,38,39,40,41,42,43,44,45,46,47,48,49,50,52,53,54,59,61,62,64,65,66,67,68,69,70,72,73,74,75,77,78,79,80,81,82,83,84,85,86,87,88,89,91,92,93,94,95,96,97,99,100,101,102,103,105,106,107,110,111,113,114,116,117,118,119,120,121,122,123,124,126,127,128,129,131,132,133,134,135,136,137,138,139,141,142,144,145,146,148]),
    "wisdm_n117_1": make_wisdm(users=[2,3,4,6,7,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,30,32,33,34,35,36,37,38,39,43,44,45,47,48,49,50,52,53,54,55,56,57,58,59,60,61,62,64,65,66,67,69,70,71,72,74,75,76,78,80,81,82,84,85,86,87,88,89,90,91,92,95,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,114,115,116,118,119,120,123,124,125,128,131,132,133,134,135,136,137,138,139,140,141,143,144,145,147,148]),
    "wisdm_n117_10": make_wisdm(users=[0,1,2,4,5,6,7,8,9,10,11,12,13,17,18,19,20,21,22,23,26,28,30,31,33,35,37,38,39,41,42,43,44,46,47,49,50,51,52,53,54,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,73,75,76,77,78,79,81,82,83,84,86,87,88,89,90,92,93,94,95,97,99,100,101,102,103,104,105,107,108,110,112,113,114,115,116,117,118,119,120,121,122,123,125,127,128,129,131,132,133,134,135,139,140,141,142,144,145,146,147,148]),
    "wisdm_n117_11": make_wisdm(users=[0,1,2,4,5,7,8,9,11,12,13,14,16,17,18,19,20,24,26,27,29,31,32,33,34,35,36,37,38,39,40,41,42,43,45,46,47,48,49,50,51,52,53,54,55,56,58,59,60,61,62,64,65,67,69,70,71,72,73,74,76,77,78,79,81,83,85,88,89,90,91,92,94,95,96,97,98,99,101,105,106,108,109,110,111,113,114,115,117,119,120,121,122,123,125,126,127,128,129,130,131,132,133,134,135,136,137,139,140,141,142,143,144,145,146,147,148]),
    "wisdm_n117_12": make_wisdm(users=[0,1,2,3,6,7,8,9,11,12,14,15,16,17,19,21,23,25,26,27,28,30,31,32,35,36,37,39,40,41,42,43,44,45,46,48,49,51,52,53,54,56,57,58,59,60,61,62,63,64,66,67,68,69,70,71,72,73,74,75,76,77,78,79,81,82,83,84,86,87,89,90,91,92,94,95,96,98,100,102,103,104,105,106,107,108,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,126,127,128,129,130,132,133,134,137,139,140,141,142,143,146,147]),
    "wisdm_n117_13": make_wisdm(users=[1,3,6,7,8,9,10,11,12,13,14,15,16,19,20,21,23,24,25,26,27,28,30,31,32,33,34,35,36,38,39,40,41,42,43,44,45,46,47,49,51,52,53,54,55,56,57,58,59,60,61,63,64,66,67,69,72,73,74,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,92,94,95,96,97,98,99,100,102,103,104,105,106,109,111,113,114,115,116,117,119,120,121,122,125,126,127,129,130,131,133,134,135,136,137,140,141,142,143,144,145,146,147]),
    "wisdm_n117_14": make_wisdm(users=[0,1,2,3,5,6,8,10,11,12,13,14,15,17,19,21,22,23,24,25,26,27,28,29,31,32,33,35,37,38,39,41,42,43,44,45,46,47,49,51,52,53,54,55,56,57,58,59,60,62,63,65,66,68,69,70,71,73,74,75,76,77,78,79,80,82,83,84,85,86,87,88,89,90,92,97,98,101,103,104,105,106,107,108,109,110,111,112,113,115,118,119,120,122,123,124,125,126,127,128,129,130,133,134,135,136,137,138,139,140,141,142,143,144,145,147,148]),
    "wisdm_n117_2": make_wisdm(users=[1,2,3,4,5,6,7,8,9,10,11,12,14,17,19,20,22,23,24,25,26,27,28,29,30,33,34,35,36,37,38,39,42,43,44,45,46,47,50,51,53,54,55,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,74,77,78,79,80,82,83,84,85,86,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,112,113,115,116,117,118,119,120,122,124,126,127,128,130,132,133,134,136,137,141,144,145,146,147,148]),
    "wisdm_n117_3": make_wisdm(users=[0,2,3,4,5,6,8,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,29,30,31,32,33,34,35,37,38,39,40,42,43,44,46,47,48,49,51,53,54,55,56,60,61,62,63,65,67,68,70,71,73,74,75,76,77,78,80,81,82,84,88,89,91,92,93,94,95,96,97,100,101,102,104,105,107,108,109,110,112,113,114,115,116,117,119,120,122,123,125,126,127,128,129,130,131,132,133,134,135,136,137,139,140,141,142,143,144,145,146,147,148]),
    "wisdm_n117_4": make_wisdm(users=[0,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,19,21,22,23,24,25,26,27,28,29,32,33,34,36,37,38,39,40,42,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,63,65,66,68,69,72,73,75,76,77,78,79,80,81,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,111,112,117,118,120,121,123,124,126,127,128,131,132,133,135,137,138,142,143,145,146,147,148]),
    "wisdm_n117_5": make_wisdm(users=[0,2,3,5,6,7,8,9,10,11,13,14,15,16,17,19,20,21,22,23,24,25,26,27,29,30,31,33,34,37,38,39,41,43,44,45,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,71,72,73,74,75,77,78,79,80,81,82,83,84,85,86,87,90,91,92,96,97,98,99,100,101,102,103,104,105,109,113,114,115,116,117,119,121,122,123,125,127,128,129,131,132,134,135,136,138,139,140,141,142,143,144,145,147,148]),
    "wisdm_n117_6": make_wisdm(users=[1,3,4,5,6,7,8,9,11,12,13,14,15,17,18,19,20,24,25,27,28,29,30,32,33,34,35,36,37,38,39,40,42,43,44,45,46,47,48,49,50,51,52,56,57,59,60,63,65,66,67,69,70,72,73,74,75,76,78,80,81,82,83,84,85,86,87,88,90,91,92,93,94,95,96,97,99,100,101,102,103,104,105,106,108,109,111,112,113,115,117,118,119,120,121,122,123,124,125,126,127,128,129,131,132,133,135,136,137,139,140,141,142,144,146,147,148]),
    "wisdm_n117_7": make_wisdm(users=[0,1,3,5,6,7,8,9,10,11,12,13,14,15,16,17,20,21,22,23,26,27,28,29,32,33,34,35,36,37,38,39,42,43,44,45,46,47,48,49,51,52,53,54,55,56,57,59,60,61,62,64,65,67,68,69,72,73,74,76,78,79,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,97,98,99,100,101,102,103,104,106,108,109,110,111,112,114,115,116,120,122,123,125,126,127,129,130,132,133,134,135,136,137,138,139,140,141,143,144,145,147,148]),
    "wisdm_n117_8": make_wisdm(users=[0,1,4,5,6,7,8,9,10,11,12,13,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,32,33,34,35,36,37,39,40,42,43,44,45,47,48,50,52,53,54,55,56,57,58,59,61,62,63,64,67,68,69,70,71,72,73,75,76,79,80,81,82,83,84,85,86,87,88,91,92,93,94,95,96,97,98,100,101,102,103,104,105,106,107,108,110,111,112,113,115,116,122,123,125,126,127,129,130,131,132,133,134,135,136,137,138,139,142,144,145,147,148]),
    "wisdm_n117_9": make_wisdm(users=[0,1,4,5,6,8,10,11,13,14,15,17,18,19,20,21,22,23,24,25,27,29,30,31,33,34,35,36,37,38,39,40,41,42,44,45,46,47,48,50,51,52,53,54,55,57,58,59,60,62,63,64,65,66,67,68,69,70,73,74,75,76,78,79,80,81,84,86,87,88,89,90,91,94,96,97,98,99,100,103,104,106,108,109,110,111,112,113,115,116,118,119,120,121,122,123,124,125,127,128,129,130,131,132,133,134,135,136,137,138,139,141,143,144,145,146,148]),
    "wisdm_n1_0": make_wisdm(users=[141]),
    "wisdm_n1_1": make_wisdm(users=[33]),
    "wisdm_n1_10": make_wisdm(users=[31]),
    "wisdm_n1_11": make_wisdm(users=[142]),
    "wisdm_n1_12": make_wisdm(users=[92]),
    "wisdm_n1_13": make_wisdm(users=[115]),
    "wisdm_n1_14": make_wisdm(users=[83]),
    "wisdm_n1_2": make_wisdm(users=[23]),
    "wisdm_n1_3": make_wisdm(users=[82]),
    "wisdm_n1_4": make_wisdm(users=[32]),
    "wisdm_n1_5": make_wisdm(users=[62]),
    "wisdm_n1_6": make_wisdm(users=[94]),
    "wisdm_n1_7": make_wisdm(users=[123]),
    "wisdm_n1_8": make_wisdm(users=[27]),
    "wisdm_n1_9": make_wisdm(users=[24]),
    "wisdm_n30_0": make_wisdm(users=[2,19,31,33,40,41,43,64,79,82,84,87,88,91,96,105,106,118,119,121,123,127,128,129,131,132,136,137,141,144]),
    "wisdm_n30_1": make_wisdm(users=[2,6,12,14,15,18,19,20,21,24,25,33,38,39,45,53,59,62,65,74,82,104,109,111,114,120,133,135,144,147]),
    "wisdm_n30_10": make_wisdm(users=[1,5,6,13,22,30,31,37,41,46,47,50,51,58,60,64,65,66,67,77,81,84,93,101,103,107,114,133,142,146]),
    "wisdm_n30_11": make_wisdm(users=[0,1,11,17,18,24,27,32,34,35,43,52,53,54,59,62,67,74,79,90,108,111,115,121,122,128,130,137,142,148]),
    "wisdm_n30_12": make_wisdm(users=[0,2,9,12,32,35,36,39,41,42,43,51,59,63,64,72,76,82,84,86,92,98,104,106,112,116,120,123,128,143]),
    "wisdm_n30_13": make_wisdm(users=[1,10,11,13,15,16,20,27,30,36,49,55,67,69,74,79,86,87,89,103,105,111,115,121,122,126,134,135,145,147]),
    "wisdm_n30_14": make_wisdm(users=[0,1,2,13,14,22,24,25,27,32,35,46,49,52,59,65,68,75,76,78,83,84,104,107,113,133,138,144,145,147]),
    "wisdm_n30_2": make_wisdm(users=[7,12,23,24,25,26,30,35,38,42,43,50,53,61,62,64,66,71,88,102,106,109,117,124,126,136,141,144,145,146]),
    "wisdm_n30_3": make_wisdm(users=[3,8,12,18,24,27,35,37,48,55,61,62,63,70,82,84,88,104,109,110,114,117,129,132,135,137,139,140,142,146]),
    "wisdm_n30_4": make_wisdm(users=[12,26,32,33,34,36,37,38,45,47,57,58,59,65,66,68,73,77,79,84,87,93,94,101,109,123,124,131,135,143]),
    "wisdm_n30_5": make_wisdm(users=[5,6,10,16,22,23,26,27,30,33,37,41,54,56,57,62,65,74,75,80,91,92,96,99,102,114,125,129,142,145]),
    "wisdm_n30_6": make_wisdm(users=[1,13,14,17,24,25,44,48,69,72,75,78,80,87,94,95,99,103,105,108,111,113,121,122,123,128,133,137,140,141]),
    "wisdm_n30_7": make_wisdm(users=[8,11,12,14,23,29,34,38,46,48,52,60,67,68,81,82,90,91,94,98,100,104,116,120,122,123,139,143,144,145]),
    "wisdm_n30_8": make_wisdm(users=[4,9,11,20,21,27,28,32,34,39,43,53,58,61,69,76,84,91,93,98,102,113,122,126,127,129,135,142,145,148]),
    "wisdm_n30_9": make_wisdm(users=[8,11,13,20,22,24,35,37,45,51,52,53,57,63,64,69,74,78,79,91,99,103,108,123,124,127,130,133,135,144]),
    "wisdm_n59_0": make_wisdm(users=[2,3,4,5,15,17,18,19,26,31,33,37,39,40,41,42,43,52,53,61,62,64,65,66,67,70,72,75,79,82,83,84,87,88,89,91,93,96,99,105,106,110,111,114,118,119,121,123,127,128,129,131,132,135,136,137,141,144,145]),
    "wisdm_n59_1": make_wisdm(users=[2,4,6,7,12,13,14,15,18,19,20,21,23,24,25,27,33,37,38,39,45,49,50,53,54,56,59,62,64,65,74,80,82,87,88,90,91,95,100,101,103,104,106,107,109,111,114,115,120,123,124,133,135,138,140,143,144,147,148]),
    "wisdm_n59_10": make_wisdm(users=[1,2,5,6,8,10,13,17,18,22,23,26,30,31,33,37,38,41,46,47,50,51,53,56,58,60,61,62,63,64,65,66,67,69,76,77,81,84,86,88,92,93,101,103,107,108,110,114,116,117,118,120,125,132,133,139,142,144,146]),
    "wisdm_n59_11": make_wisdm(users=[0,1,2,4,7,11,12,13,16,17,18,24,27,29,32,34,35,38,43,47,49,50,51,52,53,54,59,62,67,69,70,71,74,76,77,79,89,90,95,97,106,108,109,111,115,119,120,121,122,126,128,130,133,134,136,137,142,146,148]),
    "wisdm_n59_12": make_wisdm(users=[0,1,2,3,8,9,11,12,15,16,19,23,27,28,32,35,36,39,41,42,43,45,51,52,56,59,63,64,67,71,72,73,75,76,78,81,82,83,84,86,92,94,98,102,104,106,110,111,112,115,116,117,120,121,123,124,127,128,143]),
    "wisdm_n59_13": make_wisdm(users=[1,7,8,10,11,12,13,15,16,19,20,21,27,28,30,32,34,35,36,39,42,47,49,52,53,54,55,57,58,64,67,69,72,73,74,76,79,84,86,87,89,90,100,103,105,106,111,113,115,121,122,126,130,134,135,136,144,145,147]),
    "wisdm_n59_14": make_wisdm(users=[0,1,2,6,8,11,13,14,15,19,22,24,25,27,32,33,35,43,44,46,49,52,54,55,56,59,65,68,69,73,75,76,78,80,83,84,85,101,104,107,112,113,115,119,123,125,126,129,130,133,134,135,136,137,138,141,144,145,147]),
    "wisdm_n59_2": make_wisdm(users=[1,3,4,5,7,8,12,22,23,24,25,26,29,30,33,35,36,38,42,43,44,46,47,50,53,55,58,60,61,62,63,64,66,71,78,79,82,88,90,91,94,100,102,106,109,112,113,116,117,122,124,126,127,136,141,144,145,146,148]),
    "wisdm_n59_3": make_wisdm(users=[2,3,8,11,12,13,18,19,21,22,24,27,29,30,31,32,35,37,44,46,48,51,55,60,61,62,63,70,71,73,76,81,82,84,88,92,93,95,96,97,104,107,109,110,114,117,125,129,132,133,134,135,137,139,140,141,142,143,146]),
    "wisdm_n59_4": make_wisdm(users=[2,5,6,10,12,13,15,16,19,23,26,32,33,34,36,37,38,40,42,45,47,48,57,58,59,61,65,66,68,73,76,77,79,81,83,84,87,88,92,93,94,95,96,98,101,104,105,109,112,118,123,124,127,131,135,142,143,145,147]),
    "wisdm_n59_5": make_wisdm(users=[0,5,6,7,9,10,13,14,16,19,21,22,23,26,27,30,33,37,39,41,47,50,51,54,56,57,62,65,66,69,74,75,78,80,81,82,85,91,92,96,99,100,102,104,105,109,114,117,122,125,127,128,129,131,136,142,144,145,147]),
    "wisdm_n59_6": make_wisdm(users=[1,7,9,13,14,17,18,24,25,27,28,32,36,43,44,45,46,47,48,50,51,66,67,69,72,74,75,78,80,82,87,92,94,95,96,99,100,103,104,105,106,108,111,113,117,118,120,121,122,123,124,126,127,128,129,133,137,140,141]),
    "wisdm_n59_7": make_wisdm(users=[3,7,8,11,12,14,15,16,21,22,23,26,28,29,32,33,34,38,46,47,48,51,52,55,60,67,68,72,73,78,79,81,82,84,86,90,91,94,98,100,103,104,109,112,114,115,116,120,122,123,130,132,136,137,139,143,144,145,147]),
    "wisdm_n59_8": make_wisdm(users=[0,4,8,9,10,11,15,16,20,21,24,27,28,32,34,39,43,44,45,50,53,55,56,58,59,61,63,67,69,71,72,76,84,86,91,92,93,97,98,101,102,103,105,112,113,115,122,125,126,127,129,132,135,137,139,142,144,145,148]),
    "wisdm_n59_9": make_wisdm(users=[5,8,10,11,13,18,20,21,22,24,25,35,37,38,40,41,44,45,48,50,51,52,53,55,57,60,63,64,67,69,74,75,78,79,80,81,86,87,91,94,97,99,103,104,106,108,109,110,123,124,127,128,130,132,133,135,136,141,144]),
    "wisdm_n88_0": make_wisdm(users=[2,3,4,5,10,15,17,18,19,22,26,30,31,32,33,35,37,39,40,41,42,43,48,50,52,53,54,61,62,64,65,66,67,68,70,72,73,75,77,79,82,83,84,85,86,87,88,89,91,92,93,95,96,97,99,100,101,102,103,105,106,107,110,111,113,114,116,118,119,120,121,123,124,127,128,129,131,132,133,134,135,136,137,141,142,144,145,148]),
    "wisdm_n88_1": make_wisdm(users=[2,3,4,6,7,10,12,13,14,15,16,18,19,20,21,23,24,25,27,30,32,33,35,36,37,38,39,43,44,45,47,49,50,53,54,56,58,59,60,62,64,65,66,72,74,75,76,80,81,82,87,88,90,91,95,100,101,102,103,104,106,107,108,109,110,111,114,115,116,118,120,123,124,125,128,131,132,133,134,135,136,138,139,140,143,144,147,148]),
    "wisdm_n88_10": make_wisdm(users=[1,2,5,6,7,8,9,10,11,12,13,17,18,20,21,22,23,26,28,30,31,33,37,38,39,41,42,43,46,47,49,50,51,52,53,54,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,75,76,77,81,82,84,86,88,92,93,94,95,97,101,102,103,107,108,110,113,114,116,117,118,120,121,125,132,133,139,140,142,144,145,146,147]),
    "wisdm_n88_11": make_wisdm(users=[0,1,2,4,5,7,8,9,11,12,13,16,17,18,24,27,29,32,34,35,36,37,38,39,40,41,42,43,46,47,48,49,50,51,52,53,54,55,56,59,61,62,65,67,69,70,71,72,74,76,77,79,85,89,90,91,92,95,96,97,99,106,108,109,111,114,115,117,119,120,121,122,123,126,128,129,130,131,132,133,134,135,136,137,142,144,146,148]),
    "wisdm_n88_12": make_wisdm(users=[0,1,2,3,7,8,9,11,12,14,15,16,19,23,25,26,27,28,30,32,35,36,37,39,40,41,42,43,45,46,48,51,52,54,56,58,59,60,61,62,63,64,66,67,69,71,72,73,74,75,76,78,81,82,83,84,86,90,91,92,94,98,100,102,103,104,105,106,110,111,112,113,115,116,117,119,120,121,123,124,126,127,128,130,137,141,143,147]),
    "wisdm_n88_13": make_wisdm(users=[1,3,7,8,10,11,12,13,15,16,19,20,21,24,25,26,27,28,30,32,34,35,36,38,39,40,41,42,47,49,51,52,53,54,55,56,57,58,59,60,63,64,66,67,69,72,73,74,76,77,78,79,80,84,86,87,88,89,90,92,98,100,102,103,104,105,106,111,113,114,115,117,121,122,126,129,130,133,134,135,136,140,141,142,144,145,146,147]),
    "wisdm_n88_14": make_wisdm(users=[0,1,2,3,5,6,8,11,13,14,15,17,19,22,24,25,27,29,31,32,33,35,42,43,44,46,47,49,51,52,53,54,55,56,59,60,62,65,66,68,69,73,74,75,76,77,78,80,83,84,85,87,97,101,103,104,105,106,107,108,109,110,111,112,113,115,119,120,123,125,126,127,128,129,130,133,134,135,136,137,138,139,140,141,142,144,145,147]),
    "wisdm_n88_2": make_wisdm(users=[1,3,4,5,7,8,9,12,14,19,20,22,23,24,25,26,28,29,30,33,35,36,37,38,42,43,44,45,46,47,50,53,55,57,58,60,61,62,63,64,65,66,68,69,70,71,72,74,78,79,82,84,88,89,90,91,93,94,96,97,98,99,100,101,102,104,106,109,112,113,116,117,119,122,124,126,127,128,130,132,133,136,137,141,144,145,146,148]),
    "wisdm_n88_3": make_wisdm(users=[2,3,8,11,12,13,16,17,18,19,20,21,22,24,26,27,29,30,31,32,35,37,38,39,40,42,44,46,47,48,49,51,53,54,55,60,61,62,63,65,67,68,70,71,73,76,80,81,82,84,88,89,92,93,95,96,97,102,104,107,109,110,114,115,116,117,119,123,125,126,127,128,129,130,132,133,134,135,136,137,139,140,141,142,143,146,147,148]),
    "wisdm_n88_4": make_wisdm(users=[2,4,5,6,7,10,12,13,14,15,16,17,19,21,22,23,24,25,26,29,32,33,34,36,37,38,40,42,44,45,47,48,51,52,55,56,57,58,59,61,63,65,66,68,72,73,76,77,78,79,80,81,83,84,85,86,87,88,92,93,94,95,96,97,98,100,101,104,105,107,108,109,112,118,120,121,123,124,126,127,131,135,137,142,143,145,146,147]),
    "wisdm_n88_5": make_wisdm(users=[0,3,5,6,7,8,9,10,13,14,15,16,17,19,20,21,22,23,26,27,29,30,33,37,38,39,41,43,45,47,48,50,51,52,53,54,55,56,57,59,61,62,65,66,68,69,71,74,75,78,80,81,82,83,84,85,87,91,92,96,99,100,101,102,103,104,105,109,114,115,116,117,119,122,125,127,128,129,131,134,136,139,142,143,144,145,147,148]),
    "wisdm_n88_6": make_wisdm(users=[1,3,4,6,7,9,11,12,13,14,17,18,24,25,27,28,32,33,36,39,40,42,43,44,45,46,47,48,49,50,51,52,56,57,60,63,65,66,67,69,72,73,74,75,78,80,82,84,86,87,92,93,94,95,96,97,99,100,103,104,105,106,108,111,112,113,117,118,119,120,121,122,123,124,126,127,128,129,131,133,135,136,137,140,141,144,146,147]),
    "wisdm_n88_7": make_wisdm(users=[1,3,5,6,7,8,10,11,12,13,14,15,16,17,21,22,23,26,27,28,29,32,33,34,38,42,46,47,48,51,52,53,55,56,57,59,60,61,64,67,68,72,73,74,78,79,81,82,83,84,86,87,88,89,90,91,92,94,95,97,98,100,103,104,108,109,112,114,115,116,120,122,123,125,126,130,132,134,136,137,138,139,140,143,144,145,147,148]),
    "wisdm_n88_8": make_wisdm(users=[0,4,5,6,8,9,10,11,12,13,15,16,17,20,21,22,23,24,26,27,28,32,34,35,39,40,42,43,44,45,47,48,50,52,53,54,55,56,57,58,59,61,62,63,67,68,69,70,71,72,75,76,79,83,84,86,91,92,93,94,95,97,98,101,102,103,104,105,112,113,115,122,125,126,127,129,132,133,135,136,137,138,139,142,144,145,147,148]),
    "wisdm_n88_9": make_wisdm(users=[1,5,6,8,10,11,13,14,18,20,21,22,24,25,29,31,34,35,37,38,39,40,41,42,44,45,46,47,48,50,51,52,53,54,55,57,58,59,60,62,63,64,65,66,67,68,69,74,75,78,79,80,81,84,86,87,88,89,90,91,94,97,99,103,104,106,108,109,110,111,112,120,121,123,124,127,128,130,132,133,134,135,136,137,138,139,141,144]),
    "wisdm_t0": make_wisdm(users=[0], target=True),
    "wisdm_t1": make_wisdm(users=[1], target=True),
    "wisdm_t2": make_wisdm(users=[2], target=True),
    "wisdm_t3": make_wisdm(users=[3], target=True),
    "wisdm_t4": make_wisdm(users=[4], target=True),
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
    sd, td = load_da("ucihar_1", "ucihar_t2")

    print("Source")
    print(sd.train_data, sd.train_labels, sd.train_domain)
    print(sd.train_data.shape, sd.train_labels.shape, sd.train_domain.shape)
    print(sd.test_data, sd.test_labels, sd.test_domain)
    print(sd.test_data.shape, sd.test_labels.shape, sd.test_domain.shape)
    print("Target")
    print(td.train_data, td.train_labels, td.train_domain)
    print(td.train_data.shape, td.train_labels.shape, td.train_domain.shape)
    print(td.test_data, td.test_labels, td.test_domain)
    print(td.test_data.shape, td.test_labels.shape, td.test_domain.shape)


if __name__ == "__main__":
    app.run(main)
