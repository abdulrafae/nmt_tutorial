# NMT Tutorial

Installing dependencies
```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./

pip install sacrebleu

git clone https://github.com/moses-smt/mosesdecoder.git
git clone https://github.com/rsennrich/subword-nmt.git
```

Download and Pre-process IWSLT'17 French-English data
```
bash prepare_data.sh
```

Train Convolutional Sequence-to-Sequence Model
```
bash train_fconv.sh
```
or

Train Transformer Sequence-to-Sequence Model
```
bash train_xfmr.sh
```

Translate and Evaluate IWSLT'17 French Test data
```
bash translate.sh
```
