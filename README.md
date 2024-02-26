# CS6216 Group 1
## Seq2Seq Approach to End to End Relation Extraction

### Set-up environment
```bash
conda install --file requirements.txt
```

### Preprocess raw data for seq2seq approach
```bash
python prepro.py
# output: data/processed/docred/dev_features.pkl test_features.pkl train_features.pkl
```

### Training from Scratch:
```bash
python main.py
```
Checkpoint: https://nusu-my.sharepoint.com/:u:/g/personal/dissta_nus_edu_sg/Ecra4WcjNbtJrq3vfPJGffgB6BN2bnkkkOiKpvGU1eLw1Q

### Evaluation
```bash
# Note to uncomment 1st 3 lines in main() to generate dev_processed.pkl
python eval.py
```
### Experiment Result for seq2seq approach
|Task|F1|Precision|Recall|
|----|--|---------|------|
|Mentions Extraction|0.8296|0.8401|0.8193|
|Coreferences|0.7237|0.7481|0.7007|
|Relations Extraction|0.06386|0.1492|0.0406|

Comments: Quite okay for both Mentions Extractions and Coreferences but a very steep fall in performance for Relations Extraction. 
