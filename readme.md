ImagNet-Vid-Robust playground
=============================

Download the dataset first:
```
./download_dataset.sh
```

Visualize failure cases by scoring prediction (example predictions provided):

```
python score_predictions.py  example_predictions/resnet152_finetune.json
```

Failure examples stored in local directory:
```
ls resnet152_finetune_predictions/
```



