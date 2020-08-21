#!/usr/bin/env bash

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
TCROOT=$SCRIPTS/recaser/
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=16000

URL="https://wit3.fbk.eu/archive/2017-01-trnted/texts/fr/en/fr-en.tgz"
GZ=fr-en.tgz

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

SRC=fr
TRG=en
LANG=fr-en
PREP=iwslt17.tokenized.fr-en
TMP=$PREP/tmp
ORIG=orig

mkdir -p $ORIG $TMP $PREP

echo "Downloading data from ${URL}..."
cd $ORIG
wget "$URL"

if [ -f $GZ ]; then
    echo "Data successfully downloaded."
else
    echo "Data not successfully downloaded."
    exit
fi

tar zxvf $GZ
cd ..

echo "pre-processing train data..."
for L in $SRC $TRG
 do
    F=train.tags.$LANG.$L
    TOK=train.tags.$LANG.tok.$L

    cat $ORIG/$LANG/$F | \
    grep -v '<url>' | \
    grep -v '<talkid>' | \
    grep -v '<keywords>' | \
    sed -e 's/<title>//g' | \
    sed -e 's/<\/title>//g' | \
    sed -e 's/<description>//g' | \
    sed -e 's/<\/description>//g' | \
    $TOKENIZER -threads 8 -a -l $L > $TMP/$TOK
    echo ""
done

echo "Pre-processing valid data"
for L in $SRC $TRG
 do
    for O in `ls $ORIG/$LANG/IWSLT17.TED*.$L.xml`; do
    fname=${O##*/}
    F=$TMP/${fname%.*}
    echo $O $F
    grep '<seg id' $O | \
		grep -v '<url>' | \
		grep -v '<talkid>' | \
		grep -v '<keywords>' | \
		grep -v '<speaker>' | \
		grep -v '<reviewer' | \
		grep -v '<translator' | \
		grep -v '<doc' | \
		grep -v '</doc>' | \
		sed -e 's/<title>//g' | \
		sed -e 's/<\/title>//g' | \
		sed -e 's/<description>//g' | \
		sed -e 's/<\/description>//g' | \
		sed 's/^\s*//g' | \
		sed 's/\s*$//g' | \
    $TOKENIZER -threads 8 -a -l $L > $F
    echo ""
    done
done

echo "Creating train, valid, test"
for L in $SRC $TRG
 do

	cat $TMP/IWSLT17.TED.dev2010.fr-en.$L \
		$TMP/IWSLT17.TED.tst2010.fr-en.$L \
		$TMP/IWSLT17.TED.tst2011.fr-en.$L \
		$TMP/IWSLT17.TED.tst2012.fr-en.$L \
		$TMP/IWSLT17.TED.tst2013.fr-en.$L \
		$TMP/IWSLT17.TED.tst2014.fr-en.$L \
		$TMP/IWSLT17.TED.tst2015.fr-en.$L \
		> $TMP/valid.tok.$L
	
done

echo "cleaning train data..."
perl $CLEAN -ratio 1.5 $TMP/train.tags.$LANG.tok $SRC $TRG $TMP/train.tok.clean 1 175

perl $TCROOT/train-truecaser.perl -corpus $TMP/train.tok.clean.$SRC -model $PREP/truecase-model.$SRC
perl $TCROOT/train-truecaser.perl -corpus $TMP/train.tok.clean.$TRG -model $PREP/truecase-model.$TRG

echo "Truecasing train data"
for L in $SRC $TRG
 do
	$TCROOT/truecase.perl -model $PREP/truecase-model.$L < $TMP/train.tok.clean.$L > $TMP/train.tc.$L
done

echo "Truecasing valid data"
for L in $SRC $TRG 
 do
	$TCROOT/truecase.perl -model $PREP/truecase-model.$L < $TMP/valid.tok.$L > $TMP/valid.tc.$L
done

TRAIN=$TMP/train.en-fr
BPE_CODE=$PREP/code
rm -f $TRAIN
for L in $SRC $TRG
 do
    cat $TMP/train.tc.$L >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}"
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $SRC $TRG
 do
    for F in train valid
     do
        echo "apply_bpe.py on ${F}"
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $TMP/$F.tc.$L > $PREP/$F.$L
    done
done

echo "Download test.$SRC"
sacrebleu --test-set iwslt17 --language-pair ${SRC}-${TRG} --echo src > $TMP/test.$SRC

echo "Tokenize test.$SRC"
cat $TMP/test.$SRC | $TOKENIZER -threads 8 -a -l ${SRC} > $TMP/test.tok.$SRC

echo "Truecase test.$SRC"
$TCROOT/truecase.perl -model $PREP/truecase-model.$SRC < $TMP/test.tok.$SRC > $TMP/test.tc.$SRC

echo "Apply bpe on test.$SRC"
python $BPEROOT/apply_bpe.py -c $BPE_CODE < $TMP/test.tc.$SRC > $PREP/test.$SRC

echo "Removing temporary directory"
rm -rf $TMP
