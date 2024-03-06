#!/bin/bash

set -x

source env.sh

echo "args: $@"

# set the dataset dir via `DATADIR_LLP`
DATADIR=$3
[[ -z $DATADIR ]] && DATADIR='./datasets/LLP'
# set a comment via `COMMENT`
suffix=${COMMENT}

# PN, PFN, PCNN, ParT
model=$1

# "kin" "full" "vtx"
FEATURE_TYPE=$2
[[ -z ${FEATURE_TYPE} ]] && FEATURE_TYPE="full"
if ! [[ "${FEATURE_TYPE}" =~ ^(kin|full|vtx)$ ]]; then
    echo "Invalid feature type ${FEATURE_TYPE}!"
    exit 1
fi

extraopts=""
if [[ "$model" == "ParT" ]]; then
    if [[ "$FEATURE_TYPE" == "vtx" ]]; then
      modelopts="networks/ParT_vtx.py --use-amp --optimizer-option weight_decay 0.01"
      lr="1e-3"
    else
      modelopts="networks/example_ParticleTransformer.py --use-amp --optimizer-option weight_decay 0.01"
      lr="1e-3"
    fi
elif [[ "$model" == "ParT-FineTune" ]]; then
    modelopts="networks/example_ParticleTransformer_finetune.py --use-amp --optimizer-option weight_decay 0.01"
    lr="1e-4"
    extraopts="--optimizer-option lr_mult (\"fc.*\",50) --lr-scheduler none --load-model-weights models/ParT_full.pt -o fc_params [(7,0)]"
elif [[ "$model" == "PN" ]]; then
    modelopts="networks/example_ParticleNet.py"
    lr="1e-2"
elif [[ "$model" == "PN-FineTune" ]]; then
    modelopts="networks/example_ParticleNet_finetune.py"
    lr="1e-3"
    extraopts="--optimizer-option lr_mult (\"fc_out.*\",50) --lr-scheduler none --load-model-weights models/ParticleNet_kin.pt"
elif [[ "$model" == "PFN" ]]; then
    modelopts="networks/example_PFN.py"
    lr="2e-2"
    extraopts="--batch-size 4096"
elif [[ "$model" == "PCNN" ]]; then
    modelopts="networks/example_PCNN.py"
    lr="2e-2"
    extraopts="--batch-size 4096"
else
    echo "Invalid model $model!"
    exit 1
fi

#weaver \
#    --data-train "${DATADIR}/train.root" \
#    --data-val "${DATADIR}/val.root" \
#    --data-test "${DATADIR}/test.root" \
#    --data-config data/LLP/LLP_${FEATURE_TYPE}.yaml --network-config $modelopts \
#    --model-prefix training/LLP/${model}/{auto}${suffix}/net \
#    --num-workers 1 --fetch-step 1 --in-memory \
#    --batch-size 512 --samples-per-epoch $((2400 * 512)) --samples-per-epoch-val $((800 * 512)) --num-epochs 3 --gpus "" \
#    --start-lr $lr --optimizer ranger --log logs/LLP_${model}_{auto}${suffix}.log --predict-output pred.root \
#    --tensorboard LLP_${FEATURE_TYPE}_${model}${suffix} \
#    ${extraopts} "${@:3}"

weaver \
    --data-train "${DATADIR}/stop_M1000_*_train.parquet" \
    --data-test "${DATADIR}/stop_M1000_*_test.parquet" \
    --data-config data/LLP/LLP_${FEATURE_TYPE}.yaml --network-config $modelopts \
    --model-prefix training/LLP/${model}/{auto}${suffix}/net \
    --num-workers 0 --fetch-step 1 --in-memory --train-val-split 0.8889 \
    --batch-size 512 --samples-per-epoch $((500 * 512)) --samples-per-epoch-val $((200 * 512)) --num-epochs 20 --gpus "" \
    --start-lr $lr --optimizer ranger --log logs/LLP_${model}_{auto}${suffix}.log --predict-output pred.root \
    --tensorboard LLP_${FEATURE_TYPE}_${model}${suffix} \
    ${extraopts} "${@:3}"
