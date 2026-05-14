import logging
from abc import ABC, abstractmethod
import pandas as pd
import random
from datetime import datetime
import json
import os


######################################################## Make LoT splits here #################################################


def _add_missing_columns(dataframe, required_columns, context):
    missing_columns = []
    for col in required_columns:
        if col not in dataframe.columns and col not in missing_columns:
            missing_columns.append(col)
    if missing_columns:
        logging.warning(
            "%s: adding %s missing declared columns as empty columns: %s",
            context,
            len(missing_columns),
            missing_columns,
        )
        dataframe = dataframe.reindex(columns=dataframe.columns.tolist() + missing_columns)
    return dataframe


def _is_empty_measurement_frame(dataframe, columns, count_zeros_as_nans=False):
    measurement_frame = dataframe[columns]
    if count_zeros_as_nans:
        measurement_frame = measurement_frame.mask(measurement_frame == 0)
    return measurement_frame.isnull().all().all()


def _summarize_empty_split_skip(skip_summary, patientid, input_empty, output_empty):
    skip_summary["total"] += 1
    if input_empty:
        skip_summary["empty_input"] += 1
    if output_empty:
        skip_summary["empty_output"] += 1
    if len(skip_summary["examples"]) < 10:
        skip_summary["examples"].append(
            {
                "patientid": str(patientid),
                "empty_input": bool(input_empty),
                "empty_output": bool(output_empty),
            }
        )


def _log_empty_split_summary(splitter_name, skip_summary, total_patients):
    if skip_summary["total"] == 0:
        return
    logging.info(
        "%s skipped %s / %s patients due to empty input or output. Empty input: %s Empty output: %s Examples: %s",
        splitter_name,
        skip_summary["total"],
        total_patients,
        skip_summary["empty_input"],
        skip_summary["empty_output"],
        skip_summary["examples"],
    )

        


class LoTSplitNDays:
    """
    Does LoT level splitting, with all observations until at most nr_days_to_forecast returned.
    """


    def _get_index_first_date_n_days_after_input_date_else_retun_last_date(self, index_date, list_all_dates, max_days):

        assert sorted(list_all_dates) == list_all_dates, "LoTSplitNDaysRandomBackup - Date list not sorted!!!"

        # Assuming list is sorted

        ret_date = len(list_all_dates) - 1  # Default to last date

        for i in range(len(list_all_dates)):

            diff = list_all_dates[i] - index_date

            if list_all_dates[i] > index_date and diff.days <= max_days:
                ret_date = i
                break

        return ret_date
    
    def _get_index_first_date_n_days_from_index_date_else_return_none(self, index_date, list_all_dates, max_days):

        assert sorted(list_all_dates) == list_all_dates, "LoTSplitNDaysRandomBackup - Date list not sorted!!!"

        # Assuming list is sorted
        ret_date = None  # Default to last date

        for i in reversed(range(len(list_all_dates))):

            diff = list_all_dates[i] - index_date

            if diff.days <= max_days:
                ret_date = i
                break

        return ret_date
    

    def _get_first_date_with_non_na_values(self, dataframe, columns_to_extract, min_date, max_days_after_min_date):

        ret_val = None  # Default to none return val

        patient_dates = dataframe["date"].tolist()

        for i in range(dataframe.shape[0] - 1):
            
            if patient_dates[i] >= min_date and (patient_dates[i] - min_date).days < max_days_after_min_date:

                before_dates = patient_dates[0:i+1]

                extracted_df = dataframe[dataframe["date"].isin(before_dates)]
                extracted_df = extracted_df[columns_to_extract]

                is_null = extracted_df.isnull().all().all()

                if not is_null:
                    ret_val = patient_dates[i]
                    break
        
        return ret_val
        

    def _return_subset_dates_with_values(self, original_date_list, dataframe, cols_to_extract):
        
        #: go through every date, and only keep if there is at least one value for that date
        ret_list = []
        
        for idx in range(len(original_date_list)):
            
            curr_date = original_date_list[idx]

            test_df = dataframe.loc[dataframe["date"].isin([curr_date]), cols_to_extract]

            is_null = test_df.isnull().all().all()

            if not is_null:
                ret_list.append(curr_date)
        
        return ret_list



    def setup_split_indices(self, list_of_dfs, eval_manager, nr_days_to_forecast, therapies_to_ignore=("Clinical Study Drug",)):
        
        # Set seed
        random.seed(42)

        #: load LoT file
        base_path = os.path.dirname(__file__).split("/uc2_nsclc")[0] + "/uc2_nsclc/"  # Hacky way to get the base path
        df_lot = pd.read_csv(base_path + "2_experiments/2023_11_07_neutrophils/1_data/line_of_therapy.csv")
        df_lot['startdate'] = pd.to_datetime(df_lot['startdate'])

        #: setup columns to extract
        cols_to_extract = eval_manager.get_column_usage()[2].copy()
        skip_cols = ["patientid", "patient_sample_index", "disease_death.death", "disease_progression.progression", "months_to_death"]
        cols_to_extract = [x for x in cols_to_extract if x not in skip_cols]
        inputs_cols, future_known_inputs_cols, target_cols = eval_manager.get_column_usage()
        target_cols = target_cols.copy()
        target_cols.extend(["date", "patientid", "patient_sample_index"])
        inputs_cols = inputs_cols.copy()
        inputs_cols.extend(["patient_sample_index"])
        future_known_inputs_cols = future_known_inputs_cols.copy()
        future_known_inputs_cols.extend(["patient_sample_index"])
        drug_cols = [x for x in inputs_cols if "drug_" in x]

        # Setup return
        ret_list = []
        meta_data = []
        splitting_stats = {
            "nr_patients_with_no_useful_lot" : 0,
            "nr_patients_with_lots" : 0,
            "nr_useful_lots" : 0,
            "nr_lots_with_no_target_measurements" : 0,
            "nr_lots_with_no_baseline_measurements_for_target" : 0,
            "nr_lots_with_no_drug_measurements" : 0,
            "nr_lots_with_no_target_visit_dates" : 0,
            "nr_raw_lots_potentially_available" : 0,
            "nr_raw_lots_potentially_available_excluding_line0_and_bad_lots" : 0,
            "nr_line_zero" : 0,
            "nr_total_patients" : 0,
            "target_col_meta_data" : {x : {"num_lots_in_input": 0, "num_total_obs_in_input": 0, "num_lots_in_output": 0, "num_total_obs_in_output": 0} for x in target_cols},
        }

        #: go through all
        for idx, patient_events_table in enumerate(list_of_dfs):

            if idx % 100 == 0:
                logging.info("LoT set building - at patient nr: " + str(idx+1) + " / " + str(len(list_of_dfs)))

            #: load events file
            patient_events_table = patient_events_table.drop(['Unnamed: 0', 'X.2', 'X.1', 'X'], axis=1, errors="ignore")
            patient_events_table['date'] = pd.to_datetime(patient_events_table['date'])
            patient_events_table = patient_events_table.sort_values(by=["date"])

            #: check how many lots and add to statistics
            patientid = patient_events_table["patientid"].tolist()[0]
            patient_lots = df_lot[df_lot["patientid"] == patientid]
            splitting_stats["nr_raw_lots_potentially_available"] += patient_lots.shape[0]
            patient_lots = patient_lots.sort_values(by=["startdate"])
            test_lot_0 = patient_lots[patient_lots["linenumber"] == 0]
            splitting_stats["nr_line_zero"] += test_lot_0.shape[0]
            patient_lots = patient_lots[patient_lots["linenumber"] > 0]   # Skip LoT 0

            #: filter out bad LoTs that we do not want (e.g. "clinical study drug")
            therapies_to_keep = [x for x in patient_lots["linename"].tolist() if not any([bad_therapy in x for bad_therapy in therapies_to_ignore])]
            patient_lots = patient_lots[patient_lots["linename"].isin(therapies_to_keep)]
            patient_lot_dates = patient_lots[["startdate", "timetonexttreatment", "linename", "linenumber", "linesetting", "ismaintenancetherapy"]].values.tolist()

            #: extract correct columns
            patient_df_dates = patient_events_table['date'].tolist()

            # Setup
            curr_dic = {}
            num_lots_used = 0
            splitting_stats["nr_total_patients"] += 1
            splitting_stats["nr_raw_lots_potentially_available_excluding_line0_and_bad_lots"] += patient_lots.shape[0]


            #: go through every lot of this patient
            for idx, (curr_lot_start_date, curr_time_to_next_treatment, curr_line_name, curr_line_number, curr_linesetting, curr_is_maintenance) in enumerate(patient_lot_dates):
                
                # Make so split_date includes next possible patient visit --> or else might exclude therapy informaiton (if curr_lot_start_date is not on a patient event)
                # This also checks that we have the baseline measurements definetly in our dataframe 
                split_date = self._get_first_date_with_non_na_values(dataframe=patient_events_table, columns_to_extract=cols_to_extract, 
                                                                     min_date=curr_lot_start_date, max_days_after_min_date=nr_days_to_forecast)

                if split_date is None:
                    splitting_stats["nr_lots_with_no_baseline_measurements_for_target"] += 1
                    continue

                #: use the lower value for max nr days to forecast: timetonexttreatment, nr_days_to_forecast, adjusted by split date
                forecasting_days_decision = min(nr_days_to_forecast, curr_time_to_next_treatment - (split_date - curr_lot_start_date).days)

                #: get corresponding LoT data
                before_dates = [x for x in patient_df_dates if x <= split_date]
                after_dates = [x for x in patient_df_dates if x > split_date]
                
                #: get last date index to predict
                last_date_index_to_use = self._get_index_first_date_n_days_from_index_date_else_return_none(curr_lot_start_date, after_dates, forecasting_days_decision)

                #: in edge case, check that split_date is actually before the last date, since we move the split date to first observed value
                if last_date_index_to_use is None or after_dates[last_date_index_to_use] <= split_date:
                    splitting_stats["nr_lots_with_no_target_visit_dates"] += 1
                    continue

                # Extract correct dates
                after_dates = after_dates[0:last_date_index_to_use + 1]

                #: post process after_dates so that it only contains dates which have values (i.e. skip NA rows)
                after_dates = self._return_subset_dates_with_values(original_date_list=after_dates, dataframe=patient_events_table, 
                                                                    cols_to_extract=cols_to_extract)
                #: check that there are after dates
                if len(after_dates) == 0:
                    splitting_stats["nr_lots_with_no_target_measurements"] += 1
                    continue
                
                #: check if only nas before for this LoT -> skip this LoT
                extracted_df = patient_events_table[patient_events_table["date"].isin(before_dates)]
                extracted_df = extracted_df[cols_to_extract]
                is_null = extracted_df.isnull().all().all()
                if is_null:
                    splitting_stats["nr_lots_with_no_baseline_measurements_for_target"] += 1
                    continue

                # check if DF has therapy information -> skip this lot
                extracted_df_drugs = patient_events_table[patient_events_table["date"].isin(before_dates)]
                extracted_df_drugs = extracted_df_drugs[drug_cols]
                is_null_drugs = extracted_df_drugs.isnull().all().all()
                if is_null_drugs:
                    splitting_stats["nr_lots_with_no_drug_measurements"] += 1
                    continue

                #: extract current DF and append to list
                num_lots_used += 1
                split_index_name = "lot_" + str(idx)
                curr_patient_events_table = patient_events_table.copy()
                curr_patient_events_table["patient_sample_index"] = split_index_name 

                true_events_input = curr_patient_events_table.loc[curr_patient_events_table["date"].isin(before_dates), inputs_cols]
                true_future_events_input = curr_patient_events_table.loc[curr_patient_events_table["date"].isin(after_dates), future_known_inputs_cols]
                target_dataframe = curr_patient_events_table.loc[curr_patient_events_table["date"].isin(after_dates), target_cols]
                curr_const_row = eval_manager._current_master_constants_table.loc[eval_manager._current_master_constants_table["patientid"] == patientid]

                ret_list.append((curr_const_row, true_events_input, true_future_events_input, target_dataframe))

                #: create last_input_values_of_targets
                last_input_dates = self._return_subset_dates_with_values(original_date_list=before_dates, dataframe=patient_events_table,  cols_to_extract=cols_to_extract)
                last_input_values_of_targets = curr_patient_events_table.loc[curr_patient_events_table["date"].isin(last_input_dates), cols_to_extract]
                last_input_values_of_targets = [(split_date, value, col_name) for col_name, value in zip(last_input_values_of_targets.columns, last_input_values_of_targets.iloc[-1, :])]

                #: extract splitting_stats for every target_variable in both input and output
                for target_col in target_cols:
                    num_obs_in_input = len(true_events_input[target_col].dropna())
                    num_obs_in_output = len(target_dataframe[target_col].dropna())
                    splitting_stats["target_col_meta_data"][target_col]["num_lots_in_input"] += 1 if num_obs_in_input > 0 else 0
                    splitting_stats["target_col_meta_data"][target_col]["num_total_obs_in_input"] += num_obs_in_input
                    splitting_stats["target_col_meta_data"][target_col]["num_lots_in_output"] += 1 if num_obs_in_output > 0 else 0
                    splitting_stats["target_col_meta_data"][target_col]["num_total_obs_in_output"] += num_obs_in_output


                #: add to meta_data all line of therapy information and length of processing
                meta_data.append(({
                    "patientid": patientid,
                    "patient_sample_index" : split_index_name,
                    "line_name" : curr_line_name,
                    "time_to_next_treatment" : curr_time_to_next_treatment,
                    "nr_visits_to_predict" : len(after_dates),
                    "line_setting" : curr_linesetting, 
                    "line_number" : curr_line_number,
                    "is_maintenance" : curr_is_maintenance,
                    "lot_start_date" : curr_lot_start_date,
                    "last_visit_delta_days" : (after_dates[-1] - split_date).days,
                    "split_date" : split_date,
                    "vist_days_distribution_after_split_date" : list([(x - split_date).days for x in after_dates]),
                    "last_input_values_of_targets": last_input_values_of_targets,
                }))


            #: if all LoTs empty
            if num_lots_used == 0:
                splitting_stats["nr_patients_with_no_useful_lot"] += 1
            else: 
                splitting_stats["nr_patients_with_lots"] += 1
            splitting_stats["nr_useful_lots"] += num_lots_used
        
        #: log at the end the number of non useful lot patients
        logging.info("Splitting statistics: " + json.dumps(splitting_stats, indent = 4) )
               
        return ret_list, meta_data
    










class After24HSplitter:



    def _return_subset_dates_with_values(self, original_date_list, dataframe, cols_to_extract):
        
        #: go through every date, and only keep if there is at least one value for that date
        ret_list = []
        
        for idx in range(len(original_date_list)):
            
            curr_date = original_date_list[idx]

            test_df = dataframe.loc[dataframe["date"].isin([curr_date]), cols_to_extract]

            is_null = test_df.isnull().all().all()

            if not is_null:
                ret_list.append(curr_date)
        
        return ret_list





    def setup_split_indices(self, list_of_dfs, eval_manager, count_zeros_as_nans_in_target=True):
        
        # Set seed
        random.seed(42)


        #: setup columns to extract
        cols_to_extract = eval_manager.get_column_usage()[2].copy()
        skip_cols = ["patientid", "patient_sample_index", "disease_death.death", "disease_progression.progression", "months_to_death"]
        cols_to_extract = [x for x in cols_to_extract if x not in skip_cols]
        inputs_cols, future_known_inputs_cols, target_cols = eval_manager.get_column_usage()
        target_cols = target_cols.copy()
        target_cols.extend(["date", "patientid", "patient_sample_index"])
        inputs_cols = inputs_cols.copy()
        inputs_cols.extend(["patient_sample_index"])
        future_known_inputs_cols = future_known_inputs_cols.copy()
        future_known_inputs_cols.extend(["patient_sample_index"])

        if count_zeros_as_nans_in_target:
            logging.info("24H Splitter: Counting zeros as nans in target!")

        # Setup return
        ret_list = []
        meta_data = []
        skip_summary = {
            "total": 0,
            "empty_input": 0,
            "empty_output": 0,
            "examples": [],
        }

        #: go through all
        for idx, patient_events_table in enumerate(list_of_dfs):

            if idx % 100 == 0:
                logging.info("LoT set building - at patient nr: " + str(idx+1) + " / " + str(len(list_of_dfs)))

            #: load events file
            patient_events_table = patient_events_table.drop(['Unnamed: 0', 'X.2', 'X.1', 'X'], axis=1, errors="ignore")
            patient_events_table['date'] = pd.to_datetime(patient_events_table['date'])
            patient_events_table = patient_events_table.sort_values(by=["date"])

            #: check how many lots and add to statistics
            patientid = patient_events_table["patientid"].tolist()[0]

            #: split on 24 hours after first visit into before_dates and after_dates
            
            #: get split_date (last input date) - using hacky method since we use own dating system anyway
            split_date = pd.to_datetime("2024-01-01 23:00:00")
            before_dates = [x for x in patient_events_table["date"] if x <= split_date]
            after_dates = [x for x in patient_events_table["date"] if x > split_date]

            # Finalize final DFs
            curr_patient_events_table = patient_events_table.copy()
            split_index_name = "split_0"
            curr_patient_events_table["patient_sample_index"] = split_index_name
            curr_patient_events_table = _add_missing_columns(
                curr_patient_events_table,
                inputs_cols + future_known_inputs_cols + target_cols + cols_to_extract,
                "24H Splitter patientid " + str(patientid),
            )

            true_events_input = curr_patient_events_table.loc[curr_patient_events_table["date"].isin(before_dates), inputs_cols]
            true_future_events_input = curr_patient_events_table.loc[curr_patient_events_table["date"].isin(after_dates), future_known_inputs_cols]
            target_dataframe = curr_patient_events_table.loc[curr_patient_events_table["date"].isin(after_dates), target_cols]
            curr_const_row = eval_manager._current_master_constants_table.loc[eval_manager._current_master_constants_table["patientid"] == patientid]

            #: Check if empty input or output and then skip
            input_empty_check = _is_empty_measurement_frame(
                true_events_input,
                cols_to_extract,
                count_zeros_as_nans=count_zeros_as_nans_in_target,
            )
            output_empty_check = _is_empty_measurement_frame(
                target_dataframe,
                cols_to_extract,
                count_zeros_as_nans=count_zeros_as_nans_in_target,
            )

            if input_empty_check or output_empty_check:
                _summarize_empty_split_skip(skip_summary, patientid, input_empty_check, output_empty_check)
                continue
            
            # Add to return list
            ret_list.append((curr_const_row, true_events_input, true_future_events_input, target_dataframe))

            #: create last_input_values_of_targets
            last_input_dates = self._return_subset_dates_with_values(original_date_list=before_dates, dataframe=curr_patient_events_table, cols_to_extract=cols_to_extract)
            last_input_values_of_targets = curr_patient_events_table.loc[curr_patient_events_table["date"].isin(last_input_dates), cols_to_extract]
            last_input_values_of_targets = [(split_date, value, col_name) for col_name, value in zip(last_input_values_of_targets.columns, last_input_values_of_targets.iloc[-1, :])]

            #: add meta data
            meta_data.append(({
                "patientid": str(patientid),
                "patient_sample_index" : split_index_name,
                "nr_visits_to_predict" : len(after_dates),
                "last_visit_delta_hours" : (after_dates[-1] - split_date).seconds / 3600,
                "split_date" : split_date,
                "last_input_values_of_targets": last_input_values_of_targets,
            }))




        _log_empty_split_summary("24H Splitter", skip_summary, len(list_of_dfs))

        return ret_list, meta_data
    






class After1VisitSplitter:



    def _return_subset_dates_with_values(self, original_date_list, dataframe, cols_to_extract):
        
        #: go through every date, and only keep if there is at least one value for that date
        ret_list = []
        
        for idx in range(len(original_date_list)):
            
            curr_date = original_date_list[idx]

            test_df = dataframe.loc[dataframe["date"].isin([curr_date]), cols_to_extract]

            is_null = test_df.isnull().all().all()

            if not is_null:
                ret_list.append(curr_date)
        
        return ret_list





    def setup_split_indices(self, list_of_dfs, eval_manager, count_zeros_as_nans_in_target=True):
        
        # Set seed
        random.seed(42)


        #: setup columns to extract
        cols_to_extract = eval_manager.get_column_usage()[2].copy()
        skip_cols = ["patientid", "patient_sample_index", "disease_death.death", "disease_progression.progression", "months_to_death"]
        cols_to_extract = [x for x in cols_to_extract if x not in skip_cols]
        inputs_cols, future_known_inputs_cols, target_cols = eval_manager.get_column_usage()
        target_cols = target_cols.copy()
        target_cols.extend(["date", "patientid", "patient_sample_index"])
        inputs_cols = inputs_cols.copy()
        inputs_cols.extend(["patient_sample_index"])
        future_known_inputs_cols = future_known_inputs_cols.copy()
        future_known_inputs_cols.extend(["patient_sample_index"])

        if count_zeros_as_nans_in_target:
            logging.info("1 visit Splitter: Counting zeros as nans in target!")

        # Setup return
        ret_list = []
        meta_data = []
        skip_summary = {
            "total": 0,
            "empty_input": 0,
            "empty_output": 0,
            "examples": [],
        }

        #: go through all
        for idx, patient_events_table in enumerate(list_of_dfs):

            if idx % 100 == 0:
                logging.info("LoT set building - at patient nr: " + str(idx+1) + " / " + str(len(list_of_dfs)))

            #: load events file
            patient_events_table = patient_events_table.drop(['Unnamed: 0', 'X.2', 'X.1', 'X'], axis=1, errors="ignore")
            patient_events_table['date'] = pd.to_datetime(patient_events_table['date'])
            patient_events_table = patient_events_table.sort_values(by=["date"])

            #: check how many lots and add to statistics
            patientid = patient_events_table["patientid"].tolist()[0]

            #: split after first visit into before_dates and after_dates
            
            #: get split_date (last input date)
            split_date = patient_events_table["date"].tolist()[0]
            before_dates = [x for x in patient_events_table["date"] if x <= split_date]
            after_dates = [x for x in patient_events_table["date"] if x > split_date]

            # Finalize final DFs
            curr_patient_events_table = patient_events_table.copy()
            split_index_name = "split_0"
            curr_patient_events_table["patient_sample_index"] = split_index_name

            true_events_input = curr_patient_events_table.loc[curr_patient_events_table["date"].isin(before_dates), inputs_cols]
            true_future_events_input = curr_patient_events_table.loc[curr_patient_events_table["date"].isin(after_dates), future_known_inputs_cols]
            target_dataframe = curr_patient_events_table.loc[curr_patient_events_table["date"].isin(after_dates), target_cols]
            curr_const_row = eval_manager._current_master_constants_table.loc[eval_manager._current_master_constants_table["patientid"] == patientid]

            #: Check if empty input or output and then skip
            input_empty_check = _is_empty_measurement_frame(
                true_events_input,
                cols_to_extract,
                count_zeros_as_nans=count_zeros_as_nans_in_target,
            )
            output_empty_check = _is_empty_measurement_frame(
                target_dataframe,
                cols_to_extract,
                count_zeros_as_nans=count_zeros_as_nans_in_target,
            )

            if input_empty_check or output_empty_check:
                _summarize_empty_split_skip(skip_summary, patientid, input_empty_check, output_empty_check)
                continue
            
            # Add to return list
            ret_list.append((curr_const_row, true_events_input, true_future_events_input, target_dataframe))

            #: create last_input_values_of_targets
            last_input_dates = self._return_subset_dates_with_values(original_date_list=before_dates, dataframe=patient_events_table, cols_to_extract=cols_to_extract)
            last_input_values_of_targets = curr_patient_events_table.loc[curr_patient_events_table["date"].isin(last_input_dates), cols_to_extract]
            last_input_values_of_targets = [(split_date, value, col_name) for col_name, value in zip(last_input_values_of_targets.columns, last_input_values_of_targets.iloc[-1, :])]

            #: add meta data
            meta_data.append(({
                "patientid": str(patientid),
                "patient_sample_index" : split_index_name,
                "nr_visits_to_predict" : len(after_dates),
                "last_visit_delta_hours" : (after_dates[-1] - split_date).seconds / 3600,
                "split_date" : split_date,
                "last_input_values_of_targets": last_input_values_of_targets,
            }))




        _log_empty_split_summary("1 visit Splitter", skip_summary, len(list_of_dfs))

        return ret_list, meta_data
    





