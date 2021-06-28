#!/usr/bin/env bash

model=$1

python -m table.experiments test \
	--cuda \
	--model $model \
	--add_home \
	--test-file data/wikitable/wtq_preprocess_0805_no_anonymize_ent/test_split.jsonl \
	--extra-config='{
	  "table_file": "data/wikitable/wtq_preprocess_0805_no_anonymize_ent/tables.jsonl",
	  "train_shard_dir": "data/wikitable/wtq_preprocess_0805_no_anonymize_ent/data_split_1",
	  "saved_program_file": "data/wikitable/wtq_preprocess_0805_no_anonymize_ent/saved_programs.json",
	  "dev_file": "data/wikitable/wtq_preprocess_0805_no_anonymize_ent/data_split_1/dev_split.jsonl"
	}'
