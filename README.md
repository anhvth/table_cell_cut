# TABLE CELL CUT
### 1.Combine photoshop 
File to create origin training set, a psd file is expected to have three layers, [dot, line, input]
```sh
for f in data/*
do
python data_utils.py  \
    --mode combine \
    --input_dir $f/psd \
    --output_dir $f/origin_combine;
done
```
### 2. Random crop 
big combined images to create training set

```sh
python data_utils.py \
    --mode crop \
    --input_dir data/dic/origin_combine \
    --output_dir data/dic/crop \
    --crop_size 512
```
---> The output now is at data/dict/crop/512    

### 3 Train coarse network

```sh 
python shalowunet.py \
    --mode train \
    --input_dir data/crop/512 \
    --output_dir output/shalowunet/dic \
    --ngf 16 \
    --save_freq 1000 \
    --crop_size 512 \
    --scale_size 572 \
    --checkpoint output/shalowunet/dic \
    --batch_size 4
```
### 4 Test coarse network
#### 4.1 Export coarse network
```sh 
CUDA_VISIBLE_DEVICES=-1 python shalowunet.py \
    --mode export \
    --output_dir output/frozen \
    --checkpoint output/ \
    --crop_size 512
```
#### 4.2 Run course network
Frozen model:
https://drive.google.com/open?id=1-9r7mqMxoFSUDMw2RpKJEZN0gbTnpYFZ

```sh
python run_frozen_model.py  \
    --input_dir data/dic/raw/  \
    --output_dir test \
    --checkpoint output/frozen/ \
    --ext tif \
    --mode join
```


### 5 Train fine network
```sh
python refine/deepunet_refine.py \
    --mode train \
    --ngf 16 \
    --input_dir data/augment \
    --output_dir output/refine/dic/1 \
    --save_freq 1000 \
    --crop_size 1024 \
    --scale_size 1144
```


#### 5.1 Run frozen fine network
```sh
python refine/run_refine.py --input_dir test/ --checkpoint output/refine/dic/1/
```

### RM GIT
-- Remove the history from 
    rm -rf .git

-- recreate the repos from the current content only
```sh
git init
git add .
git commit -m "Initial commit"

-- push to the github remote repos ensuring you overwrite history

git remote add origin git@github.com:<YOUR ACCOUNT>/<YOUR REPOS>.git
git push -u --force origin master
```