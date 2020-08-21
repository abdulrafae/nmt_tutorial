# NMT Tutorial

Installing dependencies
```
pip install fairseq
git clone https://github.com/moses-smt/mosesdecoder.git
git clone https://github.com/rsennrich/subword-nmt.git
```

Downloading and Pre-processing IWSLT'17 ENglish-French data
```
bash prepare.sh
```

Train Convolution Sequence-to-Sequence Model
```
bash train_fconv.sh
```

Train Transformer Sequence-to-Sequence Model
```
bash train_xfmr.sh
```

Translate and Evaluate IWSLT'17 English Test data
```
bash translate.sh
```
