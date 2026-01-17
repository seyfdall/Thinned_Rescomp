import numpy as np
import os
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [20, 5]
# Set seed for reproducibility
np.random.seed(1)
from math import comb
import h5py
from glob import glob


"""
String Helpers
"""

def append_keywords_to_string(base_string, ending="", **kwargs):
    """
    Appends keyword arguments as key-value pairs to the base string.
    Decimal values are rounded to 2 decimal places.

    Parameters:
        base_string (str): The initial string.
        **kwargs: Arbitrary keyword arguments to append.

    Returns:
        str: The resulting string with appended key-value pairs.
    """
    for key, value in kwargs.items():
        if isinstance(value, (float, int)):  # Round if value is a decimal
            value = round(value, 2)
        base_string += f"_{key}={value}"
    return base_string + ending



"""
Classes for writing to and reading from HDF5 Files
"""

# TODO: Look into virtual dataset slicing
class HDF5FileHandler:
    def __init__(self, file_path, **kwargs):
        # Ensure filepath exists
        os.makedirs(file_path, exist_ok=True)

        self.file_path = append_keywords_to_string(file_path, ending=".h5", **kwargs)
        self.file = None      # HDF5 file handle

        self.attrs = {}  # Global attributes
        for key, value in kwargs.items():
            self.attrs[key] = value
    
    def open_file(self, mode='w'):
        """ Opens the HDF5 file in the specified mode. """
        self.file = h5py.File(self.file_path, mode)
    
    def close_file(self):
        """ Closes the HDF5 file. """
        if self.file is not None:
            self.file.close()
            self.file = None

    def add_attr(self, key, value):
        """ Adds a global attribute to the HDF5 file. """
        self.attrs[key] = value

    def save_attrs(self):
        """ Saves global attributes to the HDF5 file. """
        if self.file is not None:
            for key, value in self.attrs.items():
                self.file.attrs[key] = value

    def get_group_handler(self, group_name, **kwargs):
        """ Returns a GroupIOHandler for a specific group. """
        if self.file is None:
            raise RuntimeError("File must be opened before accessing groups.")
        return GroupIOHandler(self.file, group_name, **kwargs)

    def load_attrs(self):
        """ Loads global attributes from the HDF5 file. """
        if self.file is not None:
            self.attrs = {key: self.file.attrs[key] for key in self.file.attrs.keys()}

    # Context Manager Methods
    def __enter__(self):
        self.open_file()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close_file()



class GroupIOHandler:
    def __init__(self, h5file, group_name, **kwargs):
        self.attrs = {}
        self.datasets = {}

        group_name = append_keywords_to_string(group_name, **kwargs)
        # print(group_name, h5file.keys(), group_name in h5file, '\n')
        if group_name in h5file:
            self.group = h5file[group_name]
        else:
            self.group = h5file.require_group(group_name)
        self.add_attrs(**kwargs)

    def add_datasets(self, **kwargs):
        """ Adds datasets to the group. """
        for name, dataset in kwargs.items():
            self.datasets[name] = dataset

    def add_attrs(self, **kwargs):
        """ Adds attributes to the group. """
        for name, attr in kwargs.items():
            self.attrs[name] = attr

    def save_data(self):
        """ Saves all data to the group. Clears them from memory after saving. """
        for name, data in self.datasets.items():
            if name in self.group:
                del self.group[name]  # Ensure overwriting without duplicates
            self.group.create_dataset(name, data=data)

        for name, data in self.attrs.items():
            if name in self.group.attrs.keys():
                del self.group.attrs[name]  # Ensure overwriting without duplicates
            self.group.attrs[name] = data

        self.clear_data()

    def load_data(self):
        """ Loads all data from the group into memory. """
        self.datasets = {name: self.group[name][()] for name in self.group}
        self.attrs = {name: value for name, value in self.group.attrs.items()}

    def clear_data(self):
        """ Clears in-memory datasets to free up memory. """
        self.datasets.clear()
        self.attrs.clear()

    def get_dataset(self, name):
        """ Retrieves a specific dataset from in-memory storage. """
        return self.datasets.get(name)
    

def create_rescomp_datasets_template(
        div_der = [], 
        div_pos = [], 
        div_spect = [],
        div_rank = [],
        vpt = [],
        pred = [],
        err = [],
        consistency_correlation = [],
        giant_diam = [],
        largest_diam = [],
        average_diam = []
    ):
    """ Template to create attributes/datasets object for hdf5 file groups """

    return {
        "div_der": div_der, 
        "div_pos": div_pos,
        "div_spect": div_spect,
        "div_rank": div_rank,
        "vpt": vpt,
        "pred": pred,
        "err": err,
        "consistency_correlation": consistency_correlation,
        "giant_diam": giant_diam,
        "largest_diam": largest_diam,
        "average_diam": average_diam
    }


def generate_rescomp_means(datasets):
    """ Takes in datasets and returns means of each of the datasets """

    attributes = {}
    for key, value in datasets.items():
        mean_key = f"mean_{key}"
        attributes[mean_key] = np.mean(value)
    
    return attributes


"""
Example Usage:
"""

"""
    # Initialize and open the HDF5 file
    file_handler = HDF5FileHandler('example_data_with_groups.h5')
    file_handler.open_file()

    # Add and save a global attribute
    file_handler.add_attribute('experiment_date', '2024-11-09')
    file_handler.save_attributes()

    # Work with a specific group using GroupIOHandler
    group1 = file_handler.get_group_handler('experiment1')
    group1.add_dataset('sensor_readings', np.random.random((100, 3)))
    group1.add_dataset('metadata', np.array([1, 2, 3]))
    group1.save_datasets()   # Saves and clears data from memory

    # Load group data back into memory if needed
    group1.load_datasets()
    print(group1.get_dataset('sensor_readings'))

    # Close the HDF5 file
    file_handler.close_file()
"""




"""
Reading from the files
"""

metric_attrs = ['mean_average_diam', 'mean_consistency_correlation', 'mean_div_der', 'mean_div_pos', 
                'mean_div_rank', 'mean_div_spect', 'mean_giant_diam', 'mean_vpt']

def get_file_metrics(hdf5_file):
    """
    
    """
    with h5py.File(hdf5_file, 'r') as file:
        metrics = {}

        for group_name in file.keys():
            group = file[group_name]

            for attr in group.attrs:
                # Skip run_attributes for now
                if attr not in metric_attrs:
                    continue

                if attr not in metrics.keys():
                    metrics[attr] = [group.attrs[attr]]
                else:
                    metrics[attr].append(group.attrs[attr])

        print(f"Number of draws successfully made for {hdf5_file}: {len(metrics['mean_vpt'])}")
        print(f"Mean position diversity: {np.mean(metrics['mean_div_pos'])}")

        return metrics
    

def get_average_file_metrics(hdf5_file):
    """

    """
    metrics = get_file_metrics(hdf5_file)
    mean_metrics = {}

    for attr in metrics.keys():
        mean_metrics[attr] = np.mean(metrics[attr])

    return mean_metrics
    

def get_average_system_metrics(p_thins, rhos, results_path):
    """
    
    """
    comp_metrics = {attr: np.zeros((len(rhos), len(p_thins))) for attr in metric_attrs}

    for i, rho in enumerate(rhos):
        for j, p_thin in enumerate(p_thins):
            hdf5_file = results_path + f"_rho={round(rho,2)}_p_thin={round(p_thin,2)}.h5"
            mean_metrics = get_average_file_metrics(hdf5_file=hdf5_file)

            for attr in mean_metrics:
                comp_metrics[attr][i,j] = mean_metrics[attr]

            print("VPT", comp_metrics['mean_vpt'][i,j])

    print(f"Overall: {np.max(comp_metrics['mean_consistency_correlation']), np.min(comp_metrics['mean_consistency_correlation'])}")
    return comp_metrics


def remove_system_data(results_path):
    """
    
    """
    for file_path in glob(f'{results_path}*.h5'):
        os.remove(file_path)