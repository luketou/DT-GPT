import logging
import concurrent.futures
import os
from joblib import Parallel, delayed
from tqdm import tqdm


def resolve_df_conversion_n_jobs(n_jobs=None):
    try:
        if n_jobs is not None:
            resolved_n_jobs = int(n_jobs)
        else:
            resolved_n_jobs = int(os.environ.get("DTGPT_DF_CONVERSION_N_JOBS", "1"))
    except ValueError as error:
        raise ValueError("DTGPT_DF_CONVERSION_N_JOBS must be a positive integer.") from error

    if resolved_n_jobs < 1:
        raise ValueError("DTGPT_DF_CONVERSION_N_JOBS must be a positive integer.")

    return resolved_n_jobs


def process_all_tuples(list_of_data_tuples, conversion_function):

    # Setup
    list_input_strings = []
    list_target_strings = []
    list_meta_data = []
    
    for idx, data in enumerate(list_of_data_tuples):

        # log
        if idx % 10 == 0:
            logging.info("Converting DFs to Strings: " + str(idx) + " / " + str(len(list_of_data_tuples)))

        # Do actual conversion
        string_input, string_output, meta_data = conversion_function(*data)

        # Save output
        list_input_strings.append(string_input)
        list_target_strings.append(string_output)
        list_meta_data.append(meta_data)

    return list_input_strings, list_target_strings, list_meta_data



def process_all_tuples_multiprocessing(list_of_data_tuples, conversion_function, n_jobs=None):

    # Setup
    resolved_n_jobs = resolve_df_conversion_n_jobs(n_jobs)
    logging.info(
        "Converting DFs to Strings with joblib workers: %s for %s tuples",
        resolved_n_jobs,
        len(list_of_data_tuples),
    )
    results = Parallel(n_jobs=resolved_n_jobs)(delayed(conversion_function)(*i) for i in list_of_data_tuples)
    
    # unpack for returning
    list_input_strings, list_target_strings, list_meta_data = zip(*results)

    return list_input_strings, list_target_strings, list_meta_data


