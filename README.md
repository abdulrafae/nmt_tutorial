# NMT Tutorial

## Install dependencies
Setup fairseq toolkit
```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./
```

Install evaluation packages
```
pip install sacrebleu
```
Install tokenization packages
```
git clone https://github.com/moses-smt/mosesdecoder.git
```

Install Byte-pair Encoding pacakges
```
git clone https://github.com/rsennrich/subword-nmt.git
```

## Download and Pre-process IWSLT'17 French-English data
Clone the repository and change the current directory
```
git clone https://github.com/abdulrafae/nmt_tutorial.git
cd nmt_tutorial
```
Download the dataset inside the current directory from the following link:
```
https://wit3.fbk.eu/2017-01-c
```
Extract the data and preprocess
```
tar -xvzf 2017-01-trnted.tgz
mkdir orig
mv 2017-01-trnted/texts/fr/en/fr-en.tgz orig/
bash prepare_data.sh
```

## Train NMT System
Train Convolutional Sequence-to-Sequence Model
```
bash train_fconv.sh
```
or

Train Transformer Sequence-to-Sequence Model
```
bash train_xfmr.sh
```

## Decode and Evaluate NMT System
Translate and Evaluate IWSLT'17 French Test data
```
bash translate.sh
```
