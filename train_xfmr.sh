#!/usr/bin/env bash
SCRIPTS=mosesdecoder/scripts
GPU=2

SRC=fr
TRG=en
TEXT=iwslt17.tokenized.fr-en

DATA_BIN=data-bin/iwslt17_${SRC}_${TRG}
mkdir -p $DATA_BIN

#(5) Word to Integer Sequence
fairseq-preprocess --source-lang ${SRC} --target-lang ${TRG} \
    --trainpref $TEXT/train --validpref $TEXT/valid \
    --destdir $DATA_BIN \
    --workers 20
	
CPKT=checkpoint/iwslt17_${SRC}_${TRG}
LOG=log/iwslt17_${SRC}_${TRG}
mkdir -p  $CPKT $LOG

#(6) Train NMT Model (Transformer)
CUDA_VISIBLE_DEVICES=$GPU python $FAIRSEQ/train.py $DATA_BIN \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
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

#(7) Decoding NMT Model
RESULT=result/iwslt17_${SRC}_${TRG}
mkdir -p $RESULT
CUDA_VISIBLE_DEVICES=$GPU fairseq-interactive $DATA_BIN \
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