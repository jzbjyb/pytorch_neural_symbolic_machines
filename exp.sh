#!/usr/bin/env bash

NAMES=("${1}")  # ("tabert_electra_1m_ep1")
MPS=("${2}")  # ("../TaBERT/data/runs/vanilla_tabert_electra_1m_ep3/pytorch_model_epoch00.bin")
root_dir=$3

# activate env if needed
if [[ "$PATH" == *"tabert"* ]]; then
  echo "tabert env activated"
else
  echo "tabert env not activated"
  conda_base=$(conda info --base)
  source ${conda_base}/etc/profile.d/conda.sh
  conda activate tabert
fi

for i in "${!NAMES[@]}"; do
	name="${NAMES[i]}"
	mp="${MPS[i]}"
	echo $name $mp

	extra_config='{
    "table_bert_model_or_config": "'${mp}'",
    "table_file": "'${root_dir}'/data/wikitable/wtq_preprocess_0805_no_anonymize_ent/tables.jsonl",
    "train_shard_dir" : "'${root_dir}'/data/wikitable/wtq_preprocess_0805_no_anonymize_ent/data_split_1",
    "saved_program_file": "'${root_dir}'/data/wikitable/wtq_preprocess_0805_no_anonymize_ent/saved_programs.json",
    "dev_file": "'${root_dir}'/data/wikitable/wtq_preprocess_0805_no_anonymize_ent/data_split_1/dev_split.jsonl"
  }'
  echo $extra_config

	OMP_NUM_THREADS=1 python -m table.experiments train \
		--seed 0 \
		--cuda \
		--work-dir=${root_dir}/runs/${name} \
		--config=data/config/config.vanilla_bert.json \
		--extra-config="${extra_config}"
done
