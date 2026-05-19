from pipeline.data_processors.DataProcessorBiomistral import DataProcessorBiomistral


class FakeTokenizer:
    eos_token = "<eos>"
    add_eos_token = True
    chat_template = ""

    def __call__(self, text, max_length, truncation):
        return {
            "input_ids": [[min(len(item), max_length)] for item in text],
            "attention_mask": [[1] for _ in text],
        }


def make_processor(tmp_path):
    stats_path = tmp_path / "stats.json"
    stats_path.write_text("{}")
    processor = DataProcessorBiomistral.__new__(DataProcessorBiomistral)
    processor.tokenizer = FakeTokenizer()
    processor.max_total_length = 16
    processor.collator_setting = "completion"
    processor.response_template = "<patient_prediction>"
    return processor


def test_preprocess_converted_records_removes_redundant_text_columns(tmp_path):
    processor = make_processor(tmp_path)
    records = [
        {"input_text": "history A", "target_text": "target A"},
        {"input_text": "history B", "target_text": "target B"},
    ]

    dataset = processor.preprocess_converted_records(records, tokenize=True, keep_text_columns=False)

    assert len(dataset) == 2
    assert dataset.column_names == ["input_ids", "attention_mask", "concatenated_text"]
    assert dataset[0]["concatenated_text"] == "history A <patient_prediction> target A"
    assert dataset[0]["input_ids"] == [16]


def test_preprocess_converted_records_can_keep_text_columns(tmp_path):
    processor = make_processor(tmp_path)
    records = [
        {"input_text": "history A", "target_text": "target A"},
    ]

    dataset = processor.preprocess_converted_records(records, tokenize=False, keep_text_columns=True)

    assert dataset.column_names == ["input_text", "target_text", "concatenated_text"]
    assert dataset[0]["input_text"] == "history A <patient_prediction>"
    assert dataset[0]["target_text"] == "target A"
    assert dataset[0]["concatenated_text"] == "history A <patient_prediction> target A"
