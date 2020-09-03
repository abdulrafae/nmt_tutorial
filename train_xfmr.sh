#!/usr/bin/env bash
SCRIPTS=mosesdecoder/scripts
GPU=2

SRC=fr
TRG=en
TEXT=iwslt17.tokenized.fr-en

DATA_BIN=data-bin/iwslt17_${SRC}_${TRG}
mkdir -p $DATA_BIN

#(5) Word to Integer Sequence
python fairseq/preprocess.py --source-lang ${SRC} --target-lang ${TRG} \
    --trainpref $TEXT/train --validpref $TEXT/valid \
    --destdir $DATA_BIN \
    --workers 20
	
CPKT=checkpoint/iwslt17_${SRC}_${TRG}
LOG=log/iwslt17_${SRC}_${TRG}
mkdir -p  $CPKT $LOG

#(6) Train NMT Model (Transformer)
CUDA_VISIBLE_DEVICES=$GPU python fairseq/train.py $DATA_BIN \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.1 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir $CPKT | tee $LOG/train_transformer.out
