#!/usr/bin/env bash
SCRIPTS=mosesdecoder/scripts
GPU=2

SRC=fr
TRG=en
TEXT=iwslt17.tokenized.fr-en

# Translate and score the test set with sacrebleu
RESULT=result/iwslt17_${SRC}_${TRG}
mkdir -p $RESULT
CUDA_VISIBLE_DEVICES=$GPU fairseq-interactive $DATA_BIN \
    --path $CPKT/checkpoint_best.pt \
    --buffer-size 2000 --batch-size 128 \
    --beam 5 --remove-bpe \
    --input $TEXT/test.${SRC} | tee $RESULT/iwslt17.test.${SRC}-${TRG}.${TRG}.sys

# Extract translations from output file 
grep ^H $RESULT/iwslt17.test.${SRC}-${TRG}.${TRG}.sys | cut -f3- > $RESULT/hypo.tok.sys

# Remove tokenization and truecasing
cat $RESULT/hypo.tok.sys | $SCRIPTS/recaser/detruecase.perl | $SCRIPTS/tokenizer/detokenizer.perl -l ${TRG} > $RESULT/hypo.sys

# Score the test set with sacrebleu
cat $RESULT/hypo.sys | sacrebleu --test-set iwslt17 --language-pair ${SRC}-en --tokenize=13a 