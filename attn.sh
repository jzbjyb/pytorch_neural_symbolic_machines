#!/usr/bin/env bash

model=$1

export USE_TRANSFORMER=1; python -m table.attn \
	--model $model \
	--test-file data/wikitable/wtq_preprocess_0805_no_anonymize_ent/test_split.jsonl \
	--table-file data/wikitable/wtq_preprocess_0805_no_anonymize_ent/tables.jsonl
