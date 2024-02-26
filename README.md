# cs6216_group1
Seq2Seq Approach to End to End Relation Extraction

Set-up environment
```bash
conda install --file requirements.txt
```

Preprocess raw data for seq2seq approach
```bash
python prepro.py
# output: data/processed/docred/dev_features.pkl test_features.pkl train_features.pkl
```

Training from Scratch:
```bash
python main.py
```
Checkpoint: To be updated

Evaluation
```bash
# Note to uncomment 1st 3 lines in main() to generate dev_processed.pkl
python eval.py
```
