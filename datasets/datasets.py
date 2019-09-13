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

    def pad_to(self, data, desired_length):
        """ Pad to the desired length """
        current_length = data.shape[0]
        assert current_length <= desired_length, "Cannot shrink size by padding"
        return np.pad(data, [(0, desired_length - current_length), (0, 0)],
                mode="constant", constant_values=0)

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


# List of datasets
datasets = {
    "ucihar_t1": make_ucihar(users=[1], target=True),
    "ucihar_t2": make_ucihar(users=[2], target=True),
    "ucihar_t3": make_ucihar(users=[3], target=True),
    "ucihar_t4": make_ucihar(users=[4], target=True),
    "ucihar_t5": make_ucihar(users=[5], target=True),
    "ucihar_t6": make_ucihar(users=[6], target=True),
    "ucihar_t7": make_ucihar(users=[7], target=True),
    "ucihar_t8": make_ucihar(users=[8], target=True),
    "ucihar_t9": make_ucihar(users=[9], target=True),
    "ucihar_t10": make_ucihar(users=[10], target=True),
    "ucihar_t11": make_ucihar(users=[11], target=True),
    "ucihar_t12": make_ucihar(users=[12], target=True),
    "ucihar_t13": make_ucihar(users=[13], target=True),
    "ucihar_t14": make_ucihar(users=[14], target=True),
    "ucihar_t15": make_ucihar(users=[15], target=True),
    "ucihar_t16": make_ucihar(users=[16], target=True),
    "ucihar_t17": make_ucihar(users=[17], target=True),
    "ucihar_t18": make_ucihar(users=[18], target=True),
    "ucihar_t19": make_ucihar(users=[19], target=True),
    "ucihar_t20": make_ucihar(users=[20], target=True),
    "ucihar_t21": make_ucihar(users=[21], target=True),
    "ucihar_t22": make_ucihar(users=[22], target=True),
    "ucihar_t23": make_ucihar(users=[23], target=True),
    "ucihar_t24": make_ucihar(users=[24], target=True),
    "ucihar_t25": make_ucihar(users=[25], target=True),
    "ucihar_t26": make_ucihar(users=[26], target=True),
    "ucihar_t27": make_ucihar(users=[27], target=True),
    "ucihar_t28": make_ucihar(users=[28], target=True),
    "ucihar_t29": make_ucihar(users=[29], target=True),
    "ucihar_t30": make_ucihar(users=[30], target=True),

    "ucihar_1": make_ucihar(users=[1]),
    "ucihar_1,2": make_ucihar(users=[1, 2]),
    "ucihar_1,2,3": make_ucihar(users=[1, 2, 3]),

    "uwave_t1": make_uwave(users=[1], target=True),
    "uwave_t2": make_uwave(users=[2], target=True),
    "uwave_t3": make_uwave(users=[3], target=True),
    "uwave_t4": make_uwave(users=[4], target=True),
    "uwave_t5": make_uwave(users=[5], target=True),
    "uwave_t6": make_uwave(users=[6], target=True),
    "uwave_t7": make_uwave(users=[7], target=True),
    "uwave_t8": make_uwave(users=[8], target=True),

    "uwave_1": make_uwave(users=[1]),
    "uwave_1,2": make_uwave(users=[1, 2]),
    "uwave_1,2,3": make_uwave(users=[1, 2, 3]),

    "sleep_t0": make_sleep(users=[0], target=True),
    "sleep_t1": make_sleep(users=[1], target=True),
    "sleep_t2": make_sleep(users=[2], target=True),
    "sleep_t3": make_sleep(users=[3], target=True),
    "sleep_t4": make_sleep(users=[4], target=True),
    "sleep_t5": make_sleep(users=[5], target=True),
    "sleep_t6": make_sleep(users=[6], target=True),
    "sleep_t7": make_sleep(users=[7], target=True),
    "sleep_t8": make_sleep(users=[8], target=True),
    "sleep_t9": make_sleep(users=[9], target=True),
    "sleep_t10": make_sleep(users=[10], target=True),
    "sleep_t11": make_sleep(users=[11], target=True),
    "sleep_t12": make_sleep(users=[12], target=True),
    "sleep_t13": make_sleep(users=[13], target=True),
    "sleep_t14": make_sleep(users=[14], target=True),
    "sleep_t15": make_sleep(users=[15], target=True),
    "sleep_t16": make_sleep(users=[16], target=True),
    "sleep_t17": make_sleep(users=[17], target=True),
    "sleep_t18": make_sleep(users=[18], target=True),
    "sleep_t19": make_sleep(users=[19], target=True),
    "sleep_t20": make_sleep(users=[20], target=True),
    "sleep_t21": make_sleep(users=[21], target=True),
    "sleep_t22": make_sleep(users=[22], target=True),
    "sleep_t23": make_sleep(users=[23], target=True),
    "sleep_t24": make_sleep(users=[24], target=True),
    "sleep_t25": make_sleep(users=[25], target=True),

    "sleep_0": make_sleep(users=[0]),
    "sleep_0,1": make_sleep(users=[0, 1]),
    "sleep_0,1,2": make_sleep(users=[0, 1, 2]),
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
