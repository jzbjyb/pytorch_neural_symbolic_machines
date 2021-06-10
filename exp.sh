#!/usr/bin/env bash

NAMES=("bert" "tabert_ori" "tabert_ep1", "tabert_ep2")
MPS=("bert-base-uncased" "../TaBERT/pretrained_models/tabert_base_k1/model.bin" "../TaBERT/data/runs/vanilla_tabert_ep1/pytorch_model_epoch00.bin", "../TaBERT/data/runs/vanilla_tabert_ep2/pytorch_model_epoch01.bin")

for i in "${!NAMES[@]}"; do
	name="${NAMES[i]}"
	mp="${MPS[i]}"
	echo $name $mp
	OMP_NUM_THREADS=1 python -m table.experiments train \
		--seed 0 \
		--cuda \
		--work-dir=runs/demo_run_${name} \
		--config=data/config/config.vanilla_bert.json \
		--extra-config='{"table_bert_model_or_config": "'${mp}'"}' &> runs/demo_run_${name}.out
done
