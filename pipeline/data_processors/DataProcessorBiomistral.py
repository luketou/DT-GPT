import __init__  # Do all imports
import logging
import pandas as pd
import json
import numpy as np
import torch
from transformers import AutoTokenizer, LongT5Model, DataCollatorForSeq2Seq, T5Tokenizer, DataCollatorForLanguageModeling
from datasets import Dataset
import re


class CompletionOnlyDataCollator:
    """Pad causal-LM batches and compute loss only after a response template.

    This replaces TRL's removed completion-only collator for the repository's
    prompt format:

        <prompt text> <patient_prediction><target JSON>

    Labels before and including ``response_template`` are set to ``-100``.
    Padding labels are also set to ``-100``.
    """

    def __init__(self, response_template, tokenizer, label_pad_token_id=-100):
        self.response_template = response_template
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id
        self.response_token_ids = tokenizer(
            response_template,
            add_special_tokens=False,
        )["input_ids"]

        if not self.response_token_ids:
            raise ValueError("response_template tokenized to an empty token list")

    def __call__(self, features):
        input_ids = [list(feature["input_ids"]) for feature in features]
        attention_masks = [
            list(feature.get("attention_mask", [1] * len(feature["input_ids"])))
            for feature in features
        ]
        completion_masks = [self._build_completion_mask(ids) for ids in input_ids]

        max_length = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        padded_attention_masks = []
        padded_completion_masks = []

        for ids, attention_mask, completion_mask in zip(input_ids, attention_masks, completion_masks):
            pad_length = max_length - len(ids)
            if getattr(self.tokenizer, "padding_side", "right") == "left":
                padded_input_ids.append([self.tokenizer.pad_token_id] * pad_length + ids)
                padded_attention_masks.append([0] * pad_length + attention_mask)
                padded_completion_masks.append([0] * pad_length + completion_mask)
            else:
                padded_input_ids.append(ids + [self.tokenizer.pad_token_id] * pad_length)
                padded_attention_masks.append(attention_mask + [0] * pad_length)
                padded_completion_masks.append(completion_mask + [0] * pad_length)

        batch = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
        }
        completion_mask_tensor = torch.tensor(padded_completion_masks, dtype=torch.bool)
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = self.label_pad_token_id
        labels[~completion_mask_tensor] = self.label_pad_token_id
        batch["labels"] = labels
        return batch

    def _build_completion_mask(self, input_ids):
        response_start = self._find_subsequence(input_ids, self.response_token_ids)
        if response_start is None:
            logging.warning(
                "Response template %r was not found in tokenized sample; masking all labels.",
                self.response_template,
            )
            return [0] * len(input_ids)

        response_end = response_start + len(self.response_token_ids)
        return [0] * response_end + [1] * (len(input_ids) - response_end)

    @staticmethod
    def _find_subsequence(sequence, subsequence):
        max_start = len(sequence) - len(subsequence)
        for start in range(max_start + 1):
            if sequence[start:start + len(subsequence)] == subsequence:
                return start
        return None


class DataProcessorBiomistral():

    def __init__(self, experiment, path_to_statistics_file, column_name_mapping, model_to_use,
                 max_total_length=4000, collator_setting="completion"):


        #: make tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_to_use, 
                                                        force_download=False,
                                                        truncation_side="left",
                                                        padding_side="right",
                                                        add_eos_token=True,
                                                        add_bos_token=True)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.tokenizer.chat_template = ""

        # Set constants
        self.max_total_length = max_total_length
        self.column_name_mapping = column_name_mapping

        self.all_columns = None
        self.all_unk_columns = None

        # Processed data
        self.processed_dataset = {}

        self.collator_setting = collator_setting
        self.response_template = "<patient_prediction>"
        self.data_collator = None

        # Load statistics file
        with open(path_to_statistics_file) as f:
            self.statistics = json.load(f)

    def setup_cols(self, target_cols):
        #: set up columns
        self.all_columns = target_cols.copy()

        self.all_columns.insert(0, "patientid")
        self.all_columns.insert(1, "patient_sample_index")
        self.all_columns.insert(2, "date")
        self.all_unk_columns = [col for col in self.all_columns if col.startswith("diagnosis") or col.startswith("discontinuationreason")] #: get all <UNK> columns

    ##################################### MODES ##################################################
    
    def set_for_training(self):
        self.tokenizer.add_eos_token = True
    
    def set_for_inference(self):
        self.tokenizer.add_eos_token = False

    def set_converter(self, converter):
        self.current_converter = converter


    ##################################### DATASET PREPROCESSING ##################################################

    def preprocess_dataset(self, list_of_input_strings, list_of_target_strings, tokenize=True):
        
        #: call _preprocess_dic
        l = self._preprocess_dic(list_of_input_strings, list_of_target_strings, tokenize=tokenize)
        return l
    
    def preprocess_converted_records(self, converted_records, tokenize=True, keep_text_columns=False):
        logging.info("DataProcessor: building dataset from converted records")
        curr_dataset = Dataset.from_list(list(converted_records))

        if "meta_data" in curr_dataset.column_names:
            curr_dataset = curr_dataset.remove_columns(["meta_data"])

        def preprocess_function(samples):
            inputs = self.preprocess_inputs(samples["input_text"])
            targets = self.preprocess_outputs(samples["target_text"])
            concat_text = [str(input_text) + " " + str(target_text) for input_text, target_text in zip(inputs, targets)]

            if tokenize:
                model_inputs = self.tokenizer(
                    text=concat_text,
                    max_length=self.max_total_length,
                    truncation=True,
                )
            else:
                model_inputs = {}

            if keep_text_columns:
                model_inputs["input_text"] = inputs
                model_inputs["target_text"] = targets

            model_inputs["concatenated_text"] = concat_text
            return model_inputs

        remove_columns = [] if keep_text_columns else [
            column_name for column_name in ["input_text", "target_text"] if column_name in curr_dataset.column_names
        ]
        tokenized_dataset = curr_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=remove_columns,
        )
        logging.info("DataProcessor: finished building dataset from converted records")
        return tokenized_dataset

    def _preprocess_dic(self, list_of_input_strings, list_of_target_strings, tokenize):
        
        # : see https://discuss.huggingface.co/t/longt5-fine-tunning/22650
        logging.info("DataProcessor: tokenizing dataset!")

        #: gather inputs and outputs
        all_inputs = list_of_input_strings
        all_outputs = list_of_target_strings

        # Preprocess inputs
        logging.info("Started preprocessing inputs for basic stuff")
        all_inputs = self.preprocess_inputs(all_inputs)
        logging.info("Finished preprocessing inputs for basic stuff")

        #: preprocess outputs
        logging.info("Started preprocessing outputs for basic stuff")
        all_outputs = self.preprocess_outputs(all_outputs)
        logging.info("Finished preprocessing outputs for basic stuff")

        # Setup dataset
        curr_dataset = Dataset.from_dict({"input_text": all_inputs, "target_text": all_outputs})

        #: preprocess function
        def preprocess_function(samples):

            inputs = samples['input_text']
            targets = samples['target_text']

            #: concatenate inputs & outputs, since we're doing causal LM
            concat_text = [str(inputs[x]) + " " + str(targets[x]) for x in range(len(inputs))]
            
            #: preprocess output using tokenizer
            if tokenize:
                model_inputs = self.tokenizer(text=concat_text, max_length=self.max_total_length, truncation=True)
            else:
                model_inputs = {}

            model_inputs["concatenated_text"] = concat_text

            #: return lists
            return model_inputs

        # Perform actual tokenization
        tokenized_dataset = curr_dataset.map(preprocess_function, batched=True)

        logging.info("DataProcessor: finished tokenizing dataset!")

        return tokenized_dataset
    
    def preprocess_inputs(self, input_string_list):
        
        # In completion loss setting we need the response template in the input
        if self.collator_setting == "completion":
            input_string_list = [x + " " + self.response_template for x in input_string_list]

        #: remove " since it creates unnecessary labels in the input
        ret_string_list = [x.replace('"', '') for x in input_string_list]

        return ret_string_list

    def preprocess_outputs(self, output_string_list):

        return output_string_list
    


    ##################################### DF -> STRING ##################################################

    def convert_to_string_single_patient(self, constants_row, true_events_input, true_future_events_input, target_dataframe, input_filtering_function, prompt, decimal_precision=2,
                                         index_version=False, generated_trajectories_df_path=None, **kwargs):
        
        assert generated_trajectories_df_path is None
        assert index_version is False
        assert self.current_converter is not None, "DataProcessor: Converter not set up correctly"

        str_input, str_output, meta_data = self.current_converter.convert_df_to_strings(self.column_name_mapping, 
                                                                                        constants_row, true_events_input, true_future_events_input, target_dataframe,
                                                                                        input_filtering_function=input_filtering_function,
                                                                                        max_token_full_length=self.max_total_length,
                                                                                        decimal_precision=decimal_precision,
                                                                                        prompt=prompt,
                                                                                        **kwargs)


        #: return strings and meta data
        return str_input, str_output, meta_data

    
    ##################################### STRING -> DF ##################################################

    
    def convert_to_df_single_patient(self, prediction_string, all_prediction_days, patientid, patient_sample_index, prediction_days_column_wise=None):

        # NOTE: all_prediction_days can be gotten from the meta_data["all_days_output"] output from preprocess_single_patient

        assert self.all_columns is not None and self.all_unk_columns is not None, "DataProcessor: Columns not set up correctly"

        #: call converter to convert to DF
        predicted_dataframe = self.current_converter.convert_from_strings_to_df(self.column_name_mapping, 
                                                                                prediction_string, 
                                                                                all_prediction_days, 
                                                                                patientid, patient_sample_index, 
                                                                                self.all_columns, self.all_unk_columns,
                                                                                prediction_days_column_wise=prediction_days_column_wise)


        #: return DF
        return predicted_dataframe
    

    ##################################### TOKENIZATION ##################################################
    
    def encode_input_string(self, str_input):
        str_input_encoded = self.tokenizer(text=str_input, return_tensors="pt", max_length=self.max_total_length, truncation=True, padding="longest")
        return str_input_encoded
    

    def decode_generated_string(self, model_output):
        output_text = self.tokenizer.decode(model_output[0], skip_special_tokens=True)
        return output_text
    
    ##################################### POST PROCESSING GENERATED STRING ##################################################


    def post_process_string(self, output_text):

        # Extract second dic (first dic is input dic)        

        def extract_second_top_level_curly_braces(text):
            braces_count = 0
            first_brace_index = None
            second_brace_content = ''
            second_brace_found = False

            for i, char in enumerate(text):
                if char == '{':
                    braces_count += 1
                    if braces_count == 1:
                        first_brace_index = i
                elif char == '}':
                    if braces_count == 1:
                        if second_brace_found:
                            second_brace_content = text[first_brace_index:i+1]
                            break
                        second_brace_found = True
                    braces_count -= 1

            return second_brace_content

        # Example usage:
        result = extract_second_top_level_curly_braces(output_text)

        return result

    ##################################### STANDARDIZING DFS ##################################################
    
    def standardize_numeric_columns(self, df):
        #: standardize all numeric columns
        for col in df.columns:
            if col in self.statistics.keys():                
                if self.statistics[col]["type"] == "numeric":
                    #: apply standardization
                    df[col] = (df[col] - self.statistics[col]["mean_3_sigma_filtered"]) / self.statistics[col]["std_3_sigma_filtered"]

        return df
    

    def destandardize_df(self, df):
        #: destandardize all numeric columns
        for col in df.columns:
            if col in self.statistics.keys():                
                if self.statistics[col]["type"] == "numeric":
                    
                    #: first convert to numeric
                    df[col] = pd.to_numeric(df[col])
                    
                    #: apply standardization
                    df[col] = (df[col] * self.statistics[col]["std_3_sigma_filtered"]) + self.statistics[col]["mean_3_sigma_filtered"]

        return df

    ##################################### DATASET LOADER ##################################################

    def get_collator(self, model):

        if self.collator_setting == "causal_lm":
            self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

        elif self.collator_setting == "completion":
            self.data_collator = CompletionOnlyDataCollator(self.response_template, tokenizer=self.tokenizer)
            

        return self.data_collator
    
    
