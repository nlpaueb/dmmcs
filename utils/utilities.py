
import os
import logging
import pandas as pd
import numpy as np

def split_(dict_to_split_train:dict, dict_to_split_val:dict, dict_to_split_test:dict):

        train, dev, test = list(), list(), list()
        for k in dict_to_split_train.keys():
            train.append((dict_to_split_train[k].split(';')))
        
        for k in dict_to_split_val.keys():
            dev.append((dict_to_split_val[k].split(';')))

        for k in dict_to_split_test.keys():
            test.append((dict_to_split_test[k].split(';')))
                
        return train, dev, test


def split_data(captions_train:dict, captions_valid:dict, captions_test:dict):

        train_ids = list(captions_train.keys())
        dev_ids = list(captions_valid.keys())
        test_ids = list(captions_test.keys())

        return train_ids, dev_ids, test_ids


def load_tags_data(dataset_concepts_path_test) -> dict:
        """ Loads ImageCLEF dataset from directory

        Returns:
            tuple[dict, dict]: Image vectors, captions in dictionary format, with keys to be the Image IDs.
        """

        clef_concepts_df_test = pd.read_csv(dataset_concepts_path_test, sep='\t', header=None, names=['ID', 'cuis'])
        
        return dict( zip( clef_concepts_df_test.ID.to_list(), clef_concepts_df_test.cuis.to_list() ) )

    
def load_imageclef_data(captions_df_train_path, captions_df_valid_path, captions_df_test_path, dataset_concepts_mapper) -> dict:
        """ Loads ImageCLEF dataset from directory

        Returns:
            tuple[dict, dict]: Image vectors, captions in dictionary format, with keys to be the Image IDs.
        """

        # Load the three subsets into pandas dataframes
        clef_captions_df_train = pd.read_csv(captions_df_train_path)
        clef_captions_df_valid = pd.read_csv(captions_df_valid_path)
        clef_captions_df_test = pd.read_csv(captions_df_test_path)

        # and now zip them into a dict!
        captions_train = dict( zip( clef_captions_df_train.ID.to_list(), clef_captions_df_train.caption.to_list() ) )
        captions_valid = dict( zip( clef_captions_df_valid.ID.to_list(), clef_captions_df_valid.caption.to_list() ) )
        captions_test = dict( zip( clef_captions_df_test.ID.to_list(), clef_captions_df_test.caption.to_list() ) )

        concepts_mapper = pd.read_csv(dataset_concepts_mapper, sep="\t", header=None, names=['cui', 'concept'])

        # Build a mapper
        _concepts_dict = {row['cui']: row['concept'] for _, row in concepts_mapper.iterrows()}

        
        return captions_train, captions_valid, captions_test, _concepts_dict

    
def set_logger(log_path, file_name, print_on_screen):
    """
    Write logs to checkpoint and console
    """

    log_file = os.path.join(log_path, file_name)

    logging.basicConfig(
        format="[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_file,
        filemode="w",
    )
    if print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
        )
        console.setFormatter(formatter)
        logging.getLogger("").addHandler(console)


def normalize_dmm(value):
    xmin = 0
    xmax = 1.5

    norm = ((value - xmin) / (xmax-xmin))

    return norm

def normalize_lm(tensor_):
    xmin = 2
    xmax = 22

    norm = ((tensor_ - xmin) / (xmax-xmin))

    return norm
