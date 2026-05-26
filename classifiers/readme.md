1. For the LLM-classifier training using the "google/gemma-2-2b" model QLoRA fine-tuning, run:

```python LLM-cllasifier_prosocialdialog.py --train_file_path "prosocial_dialog_train_appraised.csv" --dev_file_path "prosocial_dialog_validation_appraised.csv" --test_file_path "prosocial_dialog_test_appraised.csv" --prediction_file_path "prosocialdialog_gemma_scores.csv.gz" --model "google/gemma-2-2b" --subtask B```

2. For evaluation of the trained LLM-classifier, just append ```--test_only```. Analogously train separate classifiers for the other datasets.

3. For training of ML classifier with hyperparameter tuning and selection from LR, RF, and SVM, along with export of evaluation scores and results using the test sets, run the corresponding python script ```pyhon clf_<dataset>.py```.
