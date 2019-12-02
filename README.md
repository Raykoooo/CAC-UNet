## 测试的环境
- Linux (CentOS)
- Pytorch 1.1
- cuda 9.0
- cudnn 7.6
- GCC 4.9


python 依赖库：

主要的依赖包在`requirement.txt`中
```
$ pip install -r  requirements.txt
```

`apex`安装，用来混合精度训练：
```
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

`mmcv`安装
```
$ pip install mmcv
```

## 模型训练
```
$ cd code/
$ python train.py --config_file='config/miccai/rx101.yaml' --gpu_id=0
```
- `--config_file`:        模型配置文件`config/uda_d5/f0.yfaml`
- `--gpu_id`: 单卡gpu id 

