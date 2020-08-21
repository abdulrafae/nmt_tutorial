# NMT Tutorial

Installing dependencies
```
pip install fairseq
git clone https://github.com/moses-smt/mosesdecoder.git
git clone https://github.com/rsennrich/subword-nmt.git
```

Download and Pre-process IWSLT'17 French-English data
```
bash prepare.sh
```

Train Convolutional Sequence-to-Sequence Model
```
bash train_fconv.sh
```

Train Transformer Sequence-to-Sequence Model
```
bash train_xfmr.sh
```

Translate and Evaluate IWSLT'17 French Test data
```
bash translate.sh
```
