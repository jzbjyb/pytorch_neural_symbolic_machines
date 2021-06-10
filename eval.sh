#!/usr/bin/env bash

model=$1

python -m table.experiments test \
	--cuda \
	--model $model \
	--test-file data/wikitable/wtq_preprocess_0805_no_anonymize_ent/test_split.jsonl
