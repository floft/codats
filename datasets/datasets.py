"""
Datasets

Load the desired datasets into memory so we can write them to tfrecord files
in generate_tfrecords.py
"""
import os
import rarfile  # pip install rarfile
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split


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
            test_percent=0.2):
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
    num_classes = 13
    class_labels = ["walk", "stand", "jog", "sit", "bike", "upstairs",
        "downstairs", "type", "write", "coffee", "talk", "smoke", "eat"]
    window_size = 250  # 5s of data sampled at 50Hz
    window_overlap = False

    def __init__(self, utdata_domain, *args, **kwargs):
        self.utdata_domain = utdata_domain
        super().__init__(UTDataBase.num_classes, UTDataBase.class_labels,
            UTDataBase.window_size, UTDataBase.window_overlap,
            *args, **kwargs)

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


# List of datasets
datasets = {
    "utdata_wrist": UTDataWrist,
    "utdata_pocket": UTDataPocket,
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


if __name__ == "__main__":
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
