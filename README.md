## Envs
- Pytorch 1.0
- Python 3+
- cuda 9.0+

install
```
$ pip install -r  requirements.txt
```

`apex` :  Tools for easy mixed precision and distributed training in Pytorch
```
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Dataset
├── data/
│   ├── tissue-train-neg/     
│   ├── tissue-train-pos-v1/

## Preprocessing
```
$ cd code/
$ python preprocessing.py
```

## Training
```
$ cd code/
$ python train.py --config_file='config/cac-unet-r50.yaml'
```

