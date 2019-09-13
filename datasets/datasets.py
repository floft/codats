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


# List of datasets, target separate from (multi-)source ones since the target
# always has domain=0 whereas the others have domain=1,2,... for each source
# See pick_multi_source.py
datasets = {
    "sleep_n11_0,14,10,12,25,6,15,4,7,13,20": make_sleep(users=[0,14,10,12,25,6,15,4,7,13,20]),
    "sleep_n11_1,10,13,3,24,0,25,6,16,15,12": make_sleep(users=[1,10,13,3,24,0,25,6,16,15,12]),
    "sleep_n11_10,19,12,20,16,6,0,25,9,13,1": make_sleep(users=[10,19,12,20,16,6,0,25,9,13,1]),
    "sleep_n11_10,8,0,14,6,2,18,22,21,17,24": make_sleep(users=[10,8,0,14,6,2,18,22,21,17,24]),
    "sleep_n11_13,12,14,10,20,15,2,0,16,4,24": make_sleep(users=[13,12,14,10,20,15,2,0,16,4,24]),
    "sleep_n11_16,5,1,12,8,19,18,14,23,11,22": make_sleep(users=[16,5,1,12,8,19,18,14,23,11,22]),
    "sleep_n11_17,13,10,20,19,7,6,11,16,22,12": make_sleep(users=[17,13,10,20,19,7,6,11,16,22,12]),
    "sleep_n11_18,15,13,8,25,5,0,7,3,12,10": make_sleep(users=[18,15,13,8,25,5,0,7,3,12,10]),
    "sleep_n11_2,17,4,23,0,13,9,21,10,12,6": make_sleep(users=[2,17,4,23,0,13,9,21,10,12,6]),
    "sleep_n11_23,1,3,13,14,11,6,0,16,10,15": make_sleep(users=[23,1,3,13,14,11,6,0,16,10,15]),
    "sleep_n11_23,1,9,24,11,25,6,16,14,22,15": make_sleep(users=[23,1,9,24,11,25,6,16,14,22,15]),
    "sleep_n11_25,8,10,12,19,22,7,23,17,2,11": make_sleep(users=[25,8,10,12,19,22,7,23,17,2,11]),
    "sleep_n11_3,13,12,8,25,4,10,23,18,16,22": make_sleep(users=[3,13,12,8,25,4,10,23,18,16,22]),
    "sleep_n11_4,11,22,18,2,10,19,12,9,15,24": make_sleep(users=[4,11,22,18,2,10,19,12,9,15,24]),
    "sleep_n11_5,6,9,16,7,19,23,12,14,13,18": make_sleep(users=[5,6,9,16,7,19,23,12,14,13,18]),
    "sleep_n16_0,14,10,12,25,6,15,4,7,13,20,23,9,24,11,17": make_sleep(users=[0,14,10,12,25,6,15,4,7,13,20,23,9,24,11,17]),
    "sleep_n16_1,10,13,3,24,0,25,6,16,15,12,21,2,18,5,8": make_sleep(users=[1,10,13,3,24,0,25,6,16,15,12,21,2,18,5,8]),
    "sleep_n16_10,19,12,20,16,6,0,25,9,13,1,23,22,17,2,14": make_sleep(users=[10,19,12,20,16,6,0,25,9,13,1,23,22,17,2,14]),
    "sleep_n16_10,8,0,14,6,2,18,22,21,17,24,20,12,4,3,9": make_sleep(users=[10,8,0,14,6,2,18,22,21,17,24,20,12,4,3,9]),
    "sleep_n16_13,12,14,10,20,15,2,0,16,4,24,11,5,7,17,19": make_sleep(users=[13,12,14,10,20,15,2,0,16,4,24,11,5,7,17,19]),
    "sleep_n16_16,5,1,12,8,19,18,14,23,11,22,24,13,20,3,7": make_sleep(users=[16,5,1,12,8,19,18,14,23,11,22,24,13,20,3,7]),
    "sleep_n16_17,13,10,20,19,7,6,11,16,22,12,18,2,15,23,14": make_sleep(users=[17,13,10,20,19,7,6,11,16,22,12,18,2,15,23,14]),
    "sleep_n16_18,15,13,8,25,5,0,7,3,12,10,14,2,20,9,22": make_sleep(users=[18,15,13,8,25,5,0,7,3,12,10,14,2,20,9,22]),
    "sleep_n16_2,17,4,23,0,13,9,21,10,12,6,24,11,15,1,25": make_sleep(users=[2,17,4,23,0,13,9,21,10,12,6,24,11,15,1,25]),
    "sleep_n16_23,1,3,13,14,11,6,0,16,10,15,21,22,24,20,19": make_sleep(users=[23,1,3,13,14,11,6,0,16,10,15,21,22,24,20,19]),
    "sleep_n16_23,1,9,24,11,25,6,16,14,22,15,4,17,5,13,8": make_sleep(users=[23,1,9,24,11,25,6,16,14,22,15,4,17,5,13,8]),
    "sleep_n16_25,8,10,12,19,22,7,23,17,2,11,0,16,24,9,15": make_sleep(users=[25,8,10,12,19,22,7,23,17,2,11,0,16,24,9,15]),
    "sleep_n16_3,13,12,8,25,4,10,23,18,16,22,21,17,15,2,20": make_sleep(users=[3,13,12,8,25,4,10,23,18,16,22,21,17,15,2,20]),
    "sleep_n16_4,11,22,18,2,10,19,12,9,15,24,8,17,5,21,0": make_sleep(users=[4,11,22,18,2,10,19,12,9,15,24,8,17,5,21,0]),
    "sleep_n16_5,6,9,16,7,19,23,12,14,13,18,1,24,22,10,20": make_sleep(users=[5,6,9,16,7,19,23,12,14,13,18,1,24,22,10,20]),
    "sleep_n1_0": make_sleep(users=[0]),
    "sleep_n1_1": make_sleep(users=[1]),
    "sleep_n1_10": make_sleep(users=[10]),
    "sleep_n1_13": make_sleep(users=[13]),
    "sleep_n1_16": make_sleep(users=[16]),
    "sleep_n1_17": make_sleep(users=[17]),
    "sleep_n1_18": make_sleep(users=[18]),
    "sleep_n1_2": make_sleep(users=[2]),
    "sleep_n1_23": make_sleep(users=[23]),
    "sleep_n1_25": make_sleep(users=[25]),
    "sleep_n1_3": make_sleep(users=[3]),
    "sleep_n1_4": make_sleep(users=[4]),
    "sleep_n1_5": make_sleep(users=[5]),
    "sleep_n21_0,14,10,12,25,6,15,4,7,13,20,23,9,24,11,17,5,18,21,16,22": make_sleep(users=[0,14,10,12,25,6,15,4,7,13,20,23,9,24,11,17,5,18,21,16,22]),
    "sleep_n21_1,10,13,3,24,0,25,6,16,15,12,21,2,18,5,8,20,7,9,11,17": make_sleep(users=[1,10,13,3,24,0,25,6,16,15,12,21,2,18,5,8,20,7,9,11,17]),
    "sleep_n21_10,19,12,20,16,6,0,25,9,13,1,23,22,17,2,14,5,15,18,21,7": make_sleep(users=[10,19,12,20,16,6,0,25,9,13,1,23,22,17,2,14,5,15,18,21,7]),
    "sleep_n21_10,8,0,14,6,2,18,22,21,17,24,20,12,4,3,9,5,15,13,16,25": make_sleep(users=[10,8,0,14,6,2,18,22,21,17,24,20,12,4,3,9,5,15,13,16,25]),
    "sleep_n21_13,12,14,10,20,15,2,0,16,4,24,11,5,7,17,19,8,18,6,3,21": make_sleep(users=[13,12,14,10,20,15,2,0,16,4,24,11,5,7,17,19,8,18,6,3,21]),
    "sleep_n21_16,5,1,12,8,19,18,14,23,11,22,24,13,20,3,7,10,4,21,25,17": make_sleep(users=[16,5,1,12,8,19,18,14,23,11,22,24,13,20,3,7,10,4,21,25,17]),
    "sleep_n21_17,13,10,20,19,7,6,11,16,22,12,18,2,15,23,14,3,24,5,25,8": make_sleep(users=[17,13,10,20,19,7,6,11,16,22,12,18,2,15,23,14,3,24,5,25,8]),
    "sleep_n21_18,15,13,8,25,5,0,7,3,12,10,14,2,20,9,22,6,21,17,16,19": make_sleep(users=[18,15,13,8,25,5,0,7,3,12,10,14,2,20,9,22,6,21,17,16,19]),
    "sleep_n21_2,17,4,23,0,13,9,21,10,12,6,24,11,15,1,25,14,20,8,16,22": make_sleep(users=[2,17,4,23,0,13,9,21,10,12,6,24,11,15,1,25,14,20,8,16,22]),
    "sleep_n21_23,1,3,13,14,11,6,0,16,10,15,21,22,24,20,19,8,25,12,5,9": make_sleep(users=[23,1,3,13,14,11,6,0,16,10,15,21,22,24,20,19,8,25,12,5,9]),
    "sleep_n21_23,1,9,24,11,25,6,16,14,22,15,4,17,5,13,8,2,3,7,19,12": make_sleep(users=[23,1,9,24,11,25,6,16,14,22,15,4,17,5,13,8,2,3,7,19,12]),
    "sleep_n21_25,8,10,12,19,22,7,23,17,2,11,0,16,24,9,15,18,13,3,20,14": make_sleep(users=[25,8,10,12,19,22,7,23,17,2,11,0,16,24,9,15,18,13,3,20,14]),
    "sleep_n21_3,13,12,8,25,4,10,23,18,16,22,21,17,15,2,20,7,5,24,11,14": make_sleep(users=[3,13,12,8,25,4,10,23,18,16,22,21,17,15,2,20,7,5,24,11,14]),
    "sleep_n21_4,11,22,18,2,10,19,12,9,15,24,8,17,5,21,0,20,25,1,13,7": make_sleep(users=[4,11,22,18,2,10,19,12,9,15,24,8,17,5,21,0,20,25,1,13,7]),
    "sleep_n21_5,6,9,16,7,19,23,12,14,13,18,1,24,22,10,20,0,11,3,21,8": make_sleep(users=[5,6,9,16,7,19,23,12,14,13,18,1,24,22,10,20,0,11,3,21,8]),
    "sleep_n6_0,14,10,12,25,6": make_sleep(users=[0,14,10,12,25,6]),
    "sleep_n6_1,10,13,3,24,0": make_sleep(users=[1,10,13,3,24,0]),
    "sleep_n6_10,19,12,20,16,6": make_sleep(users=[10,19,12,20,16,6]),
    "sleep_n6_10,8,0,14,6,2": make_sleep(users=[10,8,0,14,6,2]),
    "sleep_n6_13,12,14,10,20,15": make_sleep(users=[13,12,14,10,20,15]),
    "sleep_n6_16,5,1,12,8,19": make_sleep(users=[16,5,1,12,8,19]),
    "sleep_n6_17,13,10,20,19,7": make_sleep(users=[17,13,10,20,19,7]),
    "sleep_n6_18,15,13,8,25,5": make_sleep(users=[18,15,13,8,25,5]),
    "sleep_n6_2,17,4,23,0,13": make_sleep(users=[2,17,4,23,0,13]),
    "sleep_n6_23,1,3,13,14,11": make_sleep(users=[23,1,3,13,14,11]),
    "sleep_n6_23,1,9,24,11,25": make_sleep(users=[23,1,9,24,11,25]),
    "sleep_n6_25,8,10,12,19,22": make_sleep(users=[25,8,10,12,19,22]),
    "sleep_n6_3,13,12,8,25,4": make_sleep(users=[3,13,12,8,25,4]),
    "sleep_n6_4,11,22,18,2,10": make_sleep(users=[4,11,22,18,2,10]),
    "sleep_n6_5,6,9,16,7,19": make_sleep(users=[5,6,9,16,7,19]),
    "sleep_t0": make_sleep(users=[0], target=True),
    "sleep_t1": make_sleep(users=[1], target=True),
    "sleep_t2": make_sleep(users=[2], target=True),
    "sleep_t3": make_sleep(users=[3], target=True),
    "sleep_t4": make_sleep(users=[4], target=True),
    "ucihar_n13_10,15,24,18,25,7,1,11,13,26,30,28,17": make_ucihar(users=[10,15,24,18,25,7,1,11,13,26,30,28,17]),
    "ucihar_n13_12,5,25,2,14,16,10,30,18,7,17,6,23": make_ucihar(users=[12,5,25,2,14,16,10,30,18,7,17,6,23]),
    "ucihar_n13_16,10,27,22,6,28,13,15,1,12,14,5,11": make_ucihar(users=[16,10,27,22,6,28,13,15,1,12,14,5,11]),
    "ucihar_n13_16,14,6,8,15,29,10,4,23,22,28,2,27": make_ucihar(users=[16,14,6,8,15,29,10,4,23,22,28,2,27]),
    "ucihar_n13_20,16,12,24,23,8,7,14,13,17,11,26,21": make_ucihar(users=[20,16,12,24,23,8,7,14,13,17,11,26,21]),
    "ucihar_n13_20,7,25,8,17,18,19,29,4,26,13,2,15": make_ucihar(users=[20,7,25,8,17,18,19,29,4,26,13,2,15]),
    "ucihar_n13_21,20,18,16,24,17,8,3,4,22,6,14,13": make_ucihar(users=[21,20,18,16,24,17,8,3,4,22,6,14,13]),
    "ucihar_n13_21,9,4,27,12,22,6,15,10,23,1,7,2": make_ucihar(users=[21,9,4,27,12,22,6,15,10,23,1,7,2]),
    "ucihar_n13_23,1,15,20,7,27,26,13,11,28,5,21,16": make_ucihar(users=[23,1,15,20,7,27,26,13,11,28,5,21,16]),
    "ucihar_n13_23,26,11,7,18,28,5,22,8,12,4,16,1": make_ucihar(users=[23,26,11,7,18,28,5,22,8,12,4,16,1]),
    "ucihar_n13_26,23,8,30,20,14,9,12,29,3,10,13,15": make_ucihar(users=[26,23,8,30,20,14,9,12,29,3,10,13,15]),
    "ucihar_n13_26,4,19,20,24,29,11,25,21,14,8,16,30": make_ucihar(users=[26,4,19,20,24,29,11,25,21,14,8,16,30]),
    "ucihar_n13_27,23,18,6,21,24,20,5,3,9,12,19,29": make_ucihar(users=[27,23,18,6,21,24,20,5,3,9,12,19,29]),
    "ucihar_n13_4,20,1,13,26,18,24,11,15,5,17,6,8": make_ucihar(users=[4,20,1,13,26,18,24,11,15,5,17,6,8]),
    "ucihar_n13_4,23,14,9,21,17,18,28,30,11,13,25,16": make_ucihar(users=[4,23,14,9,21,17,18,28,30,11,13,25,16]),
    "ucihar_n19_10,15,24,18,25,7,1,11,13,26,30,28,17,3,20,6,16,19,21": make_ucihar(users=[10,15,24,18,25,7,1,11,13,26,30,28,17,3,20,6,16,19,21]),
    "ucihar_n19_12,5,25,2,14,16,10,30,18,7,17,6,23,26,15,24,27,29,9": make_ucihar(users=[12,5,25,2,14,16,10,30,18,7,17,6,23,26,15,24,27,29,9]),
    "ucihar_n19_16,10,27,22,6,28,13,15,1,12,14,5,11,23,8,9,19,21,24": make_ucihar(users=[16,10,27,22,6,28,13,15,1,12,14,5,11,23,8,9,19,21,24]),
    "ucihar_n19_16,14,6,8,15,29,10,4,23,22,28,2,27,20,24,9,11,13,5": make_ucihar(users=[16,14,6,8,15,29,10,4,23,22,28,2,27,20,24,9,11,13,5]),
    "ucihar_n19_20,16,12,24,23,8,7,14,13,17,11,26,21,30,18,28,3,15,4": make_ucihar(users=[20,16,12,24,23,8,7,14,13,17,11,26,21,30,18,28,3,15,4]),
    "ucihar_n19_20,7,25,8,17,18,19,29,4,26,13,2,15,21,27,1,3,28,16": make_ucihar(users=[20,7,25,8,17,18,19,29,4,26,13,2,15,21,27,1,3,28,16]),
    "ucihar_n19_21,20,18,16,24,17,8,3,4,22,6,14,13,28,9,7,15,11,26": make_ucihar(users=[21,20,18,16,24,17,8,3,4,22,6,14,13,28,9,7,15,11,26]),
    "ucihar_n19_21,9,4,27,12,22,6,15,10,23,1,7,2,29,17,5,20,18,16": make_ucihar(users=[21,9,4,27,12,22,6,15,10,23,1,7,2,29,17,5,20,18,16]),
    "ucihar_n19_23,1,15,20,7,27,26,13,11,28,5,21,16,25,14,24,22,6,19": make_ucihar(users=[23,1,15,20,7,27,26,13,11,28,5,21,16,25,14,24,22,6,19]),
    "ucihar_n19_23,26,11,7,18,28,5,22,8,12,4,16,1,21,3,17,30,29,13": make_ucihar(users=[23,26,11,7,18,28,5,22,8,12,4,16,1,21,3,17,30,29,13]),
    "ucihar_n19_26,23,8,30,20,14,9,12,29,3,10,13,15,2,17,6,4,25,27": make_ucihar(users=[26,23,8,30,20,14,9,12,29,3,10,13,15,2,17,6,4,25,27]),
    "ucihar_n19_26,4,19,20,24,29,11,25,21,14,8,16,30,9,13,7,23,6,12": make_ucihar(users=[26,4,19,20,24,29,11,25,21,14,8,16,30,9,13,7,23,6,12]),
    "ucihar_n19_27,23,18,6,21,24,20,5,3,9,12,19,29,13,26,11,16,10,22": make_ucihar(users=[27,23,18,6,21,24,20,5,3,9,12,19,29,13,26,11,16,10,22]),
    "ucihar_n19_4,20,1,13,26,18,24,11,15,5,17,6,8,27,29,23,21,3,12": make_ucihar(users=[4,20,1,13,26,18,24,11,15,5,17,6,8,27,29,23,21,3,12]),
    "ucihar_n19_4,23,14,9,21,17,18,28,30,11,13,25,16,19,3,5,22,8,6": make_ucihar(users=[4,23,14,9,21,17,18,28,30,11,13,25,16,19,3,5,22,8,6]),
    "ucihar_n1_10": make_ucihar(users=[10]),
    "ucihar_n1_12": make_ucihar(users=[12]),
    "ucihar_n1_16": make_ucihar(users=[16]),
    "ucihar_n1_17": make_ucihar(users=[17]),
    "ucihar_n1_20": make_ucihar(users=[20]),
    "ucihar_n1_21": make_ucihar(users=[21]),
    "ucihar_n1_23": make_ucihar(users=[23]),
    "ucihar_n1_26": make_ucihar(users=[26]),
    "ucihar_n1_27": make_ucihar(users=[27]),
    "ucihar_n1_4": make_ucihar(users=[4]),
    "ucihar_n25_10,15,24,18,25,7,1,11,13,26,30,28,17,3,20,6,16,19,21,8,9,27,12,14,2": make_ucihar(users=[10,15,24,18,25,7,1,11,13,26,30,28,17,3,20,6,16,19,21,8,9,27,12,14,2]),
    "ucihar_n25_12,5,25,2,14,16,10,30,18,7,17,6,23,26,15,24,27,29,9,3,4,8,20,13,21": make_ucihar(users=[12,5,25,2,14,16,10,30,18,7,17,6,23,26,15,24,27,29,9,3,4,8,20,13,21]),
    "ucihar_n25_16,10,27,22,6,28,13,15,1,12,14,5,11,23,8,9,19,21,24,3,25,29,7,26,18": make_ucihar(users=[16,10,27,22,6,28,13,15,1,12,14,5,11,23,8,9,19,21,24,3,25,29,7,26,18]),
    "ucihar_n25_16,14,6,8,15,29,10,4,23,22,28,2,27,20,24,9,11,13,5,25,17,12,21,1,18": make_ucihar(users=[16,14,6,8,15,29,10,4,23,22,28,2,27,20,24,9,11,13,5,25,17,12,21,1,18]),
    "ucihar_n25_20,16,12,24,23,8,7,14,13,17,11,26,21,30,18,28,3,15,4,19,29,6,27,9,10": make_ucihar(users=[20,16,12,24,23,8,7,14,13,17,11,26,21,30,18,28,3,15,4,19,29,6,27,9,10]),
    "ucihar_n25_20,7,25,8,17,18,19,29,4,26,13,2,15,21,27,1,3,28,16,11,22,30,6,14,10": make_ucihar(users=[20,7,25,8,17,18,19,29,4,26,13,2,15,21,27,1,3,28,16,11,22,30,6,14,10]),
    "ucihar_n25_21,20,18,16,24,17,8,3,4,22,6,14,13,28,9,7,15,11,26,12,10,30,27,5,25": make_ucihar(users=[21,20,18,16,24,17,8,3,4,22,6,14,13,28,9,7,15,11,26,12,10,30,27,5,25]),
    "ucihar_n25_21,9,4,27,12,22,6,15,10,23,1,7,2,29,17,5,20,18,16,13,14,11,28,30,24": make_ucihar(users=[21,9,4,27,12,22,6,15,10,23,1,7,2,29,17,5,20,18,16,13,14,11,28,30,24]),
    "ucihar_n25_23,1,15,20,7,27,26,13,11,28,5,21,16,25,14,24,22,6,19,8,17,10,30,18,29": make_ucihar(users=[23,1,15,20,7,27,26,13,11,28,5,21,16,25,14,24,22,6,19,8,17,10,30,18,29]),
    "ucihar_n25_23,26,11,7,18,28,5,22,8,12,4,16,1,21,3,17,30,29,13,14,24,15,20,10,27": make_ucihar(users=[23,26,11,7,18,28,5,22,8,12,4,16,1,21,3,17,30,29,13,14,24,15,20,10,27]),
    "ucihar_n25_26,23,8,30,20,14,9,12,29,3,10,13,15,2,17,6,4,25,27,1,18,22,21,11,24": make_ucihar(users=[26,23,8,30,20,14,9,12,29,3,10,13,15,2,17,6,4,25,27,1,18,22,21,11,24]),
    "ucihar_n25_26,4,19,20,24,29,11,25,21,14,8,16,30,9,13,7,23,6,12,1,27,28,15,22,3": make_ucihar(users=[26,4,19,20,24,29,11,25,21,14,8,16,30,9,13,7,23,6,12,1,27,28,15,22,3]),
    "ucihar_n25_27,23,18,6,21,24,20,5,3,9,12,19,29,13,26,11,16,10,22,1,25,30,2,14,8": make_ucihar(users=[27,23,18,6,21,24,20,5,3,9,12,19,29,13,26,11,16,10,22,1,25,30,2,14,8]),
    "ucihar_n25_4,20,1,13,26,18,24,11,15,5,17,6,8,27,29,23,21,3,12,28,30,22,10,14,16": make_ucihar(users=[4,20,1,13,26,18,24,11,15,5,17,6,8,27,29,23,21,3,12,28,30,22,10,14,16]),
    "ucihar_n25_4,23,14,9,21,17,18,28,30,11,13,25,16,19,3,5,22,8,6,29,12,15,24,7,26": make_ucihar(users=[4,23,14,9,21,17,18,28,30,11,13,25,16,19,3,5,22,8,6,29,12,15,24,7,26]),
    "ucihar_n7_10,15,24,18,25,7,1": make_ucihar(users=[10,15,24,18,25,7,1]),
    "ucihar_n7_12,5,25,2,14,16,10": make_ucihar(users=[12,5,25,2,14,16,10]),
    "ucihar_n7_16,10,27,22,6,28,13": make_ucihar(users=[16,10,27,22,6,28,13]),
    "ucihar_n7_16,14,6,8,15,29,10": make_ucihar(users=[16,14,6,8,15,29,10]),
    "ucihar_n7_20,16,12,24,23,8,7": make_ucihar(users=[20,16,12,24,23,8,7]),
    "ucihar_n7_20,7,25,8,17,18,19": make_ucihar(users=[20,7,25,8,17,18,19]),
    "ucihar_n7_21,20,18,16,24,17,8": make_ucihar(users=[21,20,18,16,24,17,8]),
    "ucihar_n7_21,9,4,27,12,22,6": make_ucihar(users=[21,9,4,27,12,22,6]),
    "ucihar_n7_23,1,15,20,7,27,26": make_ucihar(users=[23,1,15,20,7,27,26]),
    "ucihar_n7_23,26,11,7,18,28,5": make_ucihar(users=[23,26,11,7,18,28,5]),
    "ucihar_n7_26,23,8,30,20,14,9": make_ucihar(users=[26,23,8,30,20,14,9]),
    "ucihar_n7_26,4,19,20,24,29,11": make_ucihar(users=[26,4,19,20,24,29,11]),
    "ucihar_n7_27,23,18,6,21,24,20": make_ucihar(users=[27,23,18,6,21,24,20]),
    "ucihar_n7_4,20,1,13,26,18,24": make_ucihar(users=[4,20,1,13,26,18,24]),
    "ucihar_n7_4,23,14,9,21,17,18": make_ucihar(users=[4,23,14,9,21,17,18]),
    "ucihar_t1": make_ucihar(users=[1], target=True),
    "ucihar_t2": make_ucihar(users=[2], target=True),
    "ucihar_t3": make_ucihar(users=[3], target=True),
    "ucihar_t4": make_ucihar(users=[4], target=True),
    "ucihar_t5": make_ucihar(users=[5], target=True),
    "uwave_n1_1": make_uwave(users=[1]),
    "uwave_n1_2": make_uwave(users=[2]),
    "uwave_n1_3": make_uwave(users=[3]),
    "uwave_n1_4": make_uwave(users=[4]),
    "uwave_n1_5": make_uwave(users=[5]),
    "uwave_n1_6": make_uwave(users=[6]),
    "uwave_n1_7": make_uwave(users=[7]),
    "uwave_n1_8": make_uwave(users=[8]),
    "uwave_n2_2,1": make_uwave(users=[2,1]),
    "uwave_n2_3,5": make_uwave(users=[3,5]),
    "uwave_n2_3,6": make_uwave(users=[3,6]),
    "uwave_n2_4,6": make_uwave(users=[4,6]),
    "uwave_n2_4,7": make_uwave(users=[4,7]),
    "uwave_n2_5,1": make_uwave(users=[5,1]),
    "uwave_n2_5,4": make_uwave(users=[5,4]),
    "uwave_n2_5,8": make_uwave(users=[5,8]),
    "uwave_n2_6,3": make_uwave(users=[6,3]),
    "uwave_n2_6,7": make_uwave(users=[6,7]),
    "uwave_n2_7,1": make_uwave(users=[7,1]),
    "uwave_n2_8,1": make_uwave(users=[8,1]),
    "uwave_n2_8,4": make_uwave(users=[8,4]),
    "uwave_n2_8,5": make_uwave(users=[8,5]),
    "uwave_n3_2,1,5": make_uwave(users=[2,1,5]),
    "uwave_n3_3,5,6": make_uwave(users=[3,5,6]),
    "uwave_n3_3,6,5": make_uwave(users=[3,6,5]),
    "uwave_n3_4,6,2": make_uwave(users=[4,6,2]),
    "uwave_n3_4,6,8": make_uwave(users=[4,6,8]),
    "uwave_n3_4,7,1": make_uwave(users=[4,7,1]),
    "uwave_n3_5,1,6": make_uwave(users=[5,1,6]),
    "uwave_n3_5,4,6": make_uwave(users=[5,4,6]),
    "uwave_n3_5,8,4": make_uwave(users=[5,8,4]),
    "uwave_n3_6,3,7": make_uwave(users=[6,3,7]),
    "uwave_n3_6,7,2": make_uwave(users=[6,7,2]),
    "uwave_n3_7,1,5": make_uwave(users=[7,1,5]),
    "uwave_n3_8,1,6": make_uwave(users=[8,1,6]),
    "uwave_n3_8,4,6": make_uwave(users=[8,4,6]),
    "uwave_n3_8,5,7": make_uwave(users=[8,5,7]),
    "uwave_n4_2,1,5,7": make_uwave(users=[2,1,5,7]),
    "uwave_n4_3,5,6,4": make_uwave(users=[3,5,6,4]),
    "uwave_n4_3,6,5,2": make_uwave(users=[3,6,5,2]),
    "uwave_n4_4,6,2,7": make_uwave(users=[4,6,2,7]),
    "uwave_n4_4,6,8,2": make_uwave(users=[4,6,8,2]),
    "uwave_n4_4,7,1,5": make_uwave(users=[4,7,1,5]),
    "uwave_n4_5,1,6,8": make_uwave(users=[5,1,6,8]),
    "uwave_n4_5,4,6,3": make_uwave(users=[5,4,6,3]),
    "uwave_n4_5,8,4,6": make_uwave(users=[5,8,4,6]),
    "uwave_n4_6,3,7,5": make_uwave(users=[6,3,7,5]),
    "uwave_n4_6,7,2,4": make_uwave(users=[6,7,2,4]),
    "uwave_n4_7,1,5,6": make_uwave(users=[7,1,5,6]),
    "uwave_n4_8,1,6,3": make_uwave(users=[8,1,6,3]),
    "uwave_n4_8,4,6,2": make_uwave(users=[8,4,6,2]),
    "uwave_n4_8,5,7,3": make_uwave(users=[8,5,7,3]),
    "uwave_n5_2,1,5,7,8": make_uwave(users=[2,1,5,7,8]),
    "uwave_n5_3,5,6,4,8": make_uwave(users=[3,5,6,4,8]),
    "uwave_n5_3,6,5,2,8": make_uwave(users=[3,6,5,2,8]),
    "uwave_n5_4,6,2,7,5": make_uwave(users=[4,6,2,7,5]),
    "uwave_n5_4,6,8,2,1": make_uwave(users=[4,6,8,2,1]),
    "uwave_n5_4,7,1,5,8": make_uwave(users=[4,7,1,5,8]),
    "uwave_n5_5,1,6,8,2": make_uwave(users=[5,1,6,8,2]),
    "uwave_n5_5,4,6,3,7": make_uwave(users=[5,4,6,3,7]),
    "uwave_n5_5,8,4,6,2": make_uwave(users=[5,8,4,6,2]),
    "uwave_n5_6,3,7,5,1": make_uwave(users=[6,3,7,5,1]),
    "uwave_n5_6,7,2,4,8": make_uwave(users=[6,7,2,4,8]),
    "uwave_n5_7,1,5,6,3": make_uwave(users=[7,1,5,6,3]),
    "uwave_n5_8,1,6,3,7": make_uwave(users=[8,1,6,3,7]),
    "uwave_n5_8,4,6,2,1": make_uwave(users=[8,4,6,2,1]),
    "uwave_n5_8,5,7,3,1": make_uwave(users=[8,5,7,3,1]),
    "uwave_t1": make_uwave(users=[1], target=True),
    "uwave_t2": make_uwave(users=[2], target=True),
    "uwave_t3": make_uwave(users=[3], target=True),
    "uwave_t4": make_uwave(users=[4], target=True),
    "uwave_t5": make_uwave(users=[5], target=True),
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
