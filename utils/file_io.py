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
    

#to add a new dataset template you must change this function's declaration
def create_rescomp_datasets_template(
        attributes = None,
        **kwargs,
    ):
    """ Template to create attributes/datasets object for hdf5 file groups """


    #safe default mutable argument
    if attributes is None:
        attributes = ["div_der", "div_pos", "div_spect", "div_rank", "vpt", "pred", "err", "consistency_correlation"]
        
    if not kwargs.keys():
        #no keyword arguments, then set all attributes to empty
        template = {attribute: [] for attribute in attributes}
        return template

    #we have attributes passed and keyword arguments overriding their defaults, make sure they contain the same amount
    assert set(kwargs.keys()) == set(attributes), f"Keyword arguments and attributes must match. Attributes - {attributes}, Keywords - {list(kwargs.keys())}"
    template = {attribute: kwargs[attribute] for attribute in attributes}
    return template


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

def get_file_data(
        hdf5_file='results/erdos_results_0.h5', 
        wanted_attributes=['mean_vpt','mean_div_pos','mean_div_der','mean_div_spect','mean_div_rank','mean_consistency_correlation','mean_component_size','mean_fraction_driving']
    ):

    with h5py.File(hdf5_file, 'r') as file:

        lists = {k: [] for k in wanted_attributes}

        for group_name in file.keys():
            group = file[group_name]
            if 'mean_vpt' not in list(group.attrs):
                continue

            for k in wanted_attributes:
                # wanted attributes must match the exact spelling in the group.attrs otherwise an error will be thrown
                lists[k].append(group.attrs[k])
            
            # print('{}, c: {}, vpt_connected: {}, p_thin: {}, vpt_thinned: {}'.format(group_name, c, vpt_connected, p_thin, vpt_thinned))
        # print('vpt_connected_average: {}, vpt_thinned_average: {}'.format(np.mean(vpt_connected_list), np.mean(vpt_thinned_list)))
        
        means = {key: np.mean(value) for key, value in lists.items()}
        print(f"Number of draws successfully made for {hdf5_file}: {len(lists['mean_vpt'])}")

        #TODO remove this comment so it prints the diversities
        #print(f"Mean diversity: {lists['mean_div_pov'], lists['mean_div_der'], lists['mean_div_spect'], lists['mean_div_rank']}")
        
        return means
    

def get_system_data(
        p_thins, 
        rhos, 
        results_path, 
        wanted_attributes=['mean_vpt','mean_div_pos','mean_div_der','mean_div_spect','mean_div_rank','mean_consistency_correlation','mean_component_size','mean_fraction_driving']
        ):


    mean_arrays = {attribute: np.zeros((len(rhos), len(p_thins))) for attribute in wanted_attributes}

    for i, rho in enumerate(rhos):
        for j, p_thin in enumerate(p_thins):
            hdf5_file = results_path + f"_rho={round(rho,2)}_p_thin={round(p_thin,2)}.h5"
            means = get_file_data(hdf5_file=hdf5_file, wanted_attributes=wanted_attributes)
            #check to make sure we don't have extra attributes and we are not silently ignoring not assigning attributes
            assert set(means) == set(wanted_attributes), f"Expected {wanted_attributes}, got {list(means)}"
            
            for attribute in wanted_attributes:
                mean_arrays[attribute][i, j] = means[attribute]
            print("VPT", mean_arrays['mean_vpt'][i,j])

    #print(f"Overall: {np.max(mean_arrays['mean_consistency_correlation']), np.min(mean_arrays['mean_consistency_correlation'])}")
    return tuple(mean_arrays[attribute] for attribute in wanted_attributes)

def get_system_data_dict(
        p_thins, 
        rhos, 
        results_path, 
        wanted_attributes=['mean_vpt','mean_div_pos','mean_div_der','mean_div_spect','mean_div_rank','mean_consistency_correlation','mean_component_size','mean_fraction_driving']
        ):

    mean_arrays = {attribute: np.zeros((len(rhos), len(p_thins))) for attribute in wanted_attributes}

    for i, rho in enumerate(rhos):
        for j, p_thin in enumerate(p_thins):
            hdf5_file = results_path + f"_rho={round(rho,2)}_p_thin={round(p_thin,2)}.h5"
            means = get_file_data(hdf5_file=hdf5_file, wanted_attributes=wanted_attributes)
            #check to make sure we don't have extra attributes and we are not silently ignoring not assigning attributes
            assert set(means) == set(wanted_attributes), f"Expected {wanted_attributes}, got {list(means)}"
            
            for attribute in wanted_attributes:
                mean_arrays[attribute][i, j] = means[attribute]
            print("VPT", mean_arrays['mean_vpt'][i,j])

    #print(f"Overall: {np.max(mean_arrays['mean_consistency_correlation']), np.min(mean_arrays['mean_consistency_correlation'])}")
    return mean_arrays


def remove_system_data(results_path):
    """
    
    """
    for file_path in glob(f'{results_path}*.h5'):
        os.remove(file_path)



