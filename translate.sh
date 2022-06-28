#!/usr/bin/env bash
SCRIPTS=mosesdecoder/scripts
GPU=0

SRC=fr
TRG=en
TEXT=iwslt17.tokenized.fr-en


#(7) Decoding NMT Model
DATA_BIN=data-bin/iwslt17_${SRC}_${TRG}
CPKT=checkpoint/iwslt17_${SRC}_${TRG}
RESULT=result/iwslt17_${SRC}_${TRG}
mkdir -p $RESULT
CUDA_VISIBLE_DEVICES=$GPU python fairseq/interactive.py $DATA_BIN \
    --path $CPKT/checkpoint_best.pt \
    --buffer-size 2000 --batch-size 128 \
    --beam 5 --remove-bpe \
    --input $TEXT/test.${SRC} | tee $RESULT/iwslt17.test.${SRC}-${TRG}.${TRG}.sys

#(8) Post-processing	
# Extract translations from output file 
grep ^H $RESULT/iwslt17.test.${SRC}-${TRG}.${TRG}.sys | cut -f3- > $RESULT/hypo.tok.sys

# Remove tokenization and truecasing
cat $RESULT/hypo.tok.sys | $SCRIPTS/recaser/detruecase.perl | $SCRIPTS/tokenizer/detokenizer.perl -l ${TRG} > $RESULT/hypo.sys

#(9) Automatic Evaluation
# Score the test set with sacrebleu
cat $RESULT/hypo.sys | sacrebleu --test-set iwslt17 --language-pair ${SRC}-en --tokenize=13a 
