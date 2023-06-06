# VidChapters-7M: Video Chapters at Scale

[Webpage](https://antoyang.github.io/vidchapters-7m.html)

![Teaser](https://antoyang.github.io/img/vidchapters-7m.png)

In this work, we present VidChapters-7M, a large-scale dataset of user-chaptered videos. 
We study three tasks on top of this dataset and show that video chapter generation models trained on VidChapters-7M transfer well to dense video captioning.

This repository provides the code for our paper, including:
- Environment setup
- Data collection pipeline for VidChapters-7M
- Data downloading instructions
- Data processing scripts
- Training and evaluation scripts for the tasks of video chapter generation without or with ground-truth boundaries and video chapter grounding on VidChapters-7M, and dense video captioning on YouCook2 and ViTT

Preprocessed data and model checkpoints will be released soon.

## Paths and Requirements
Fill the empty paths in the file `args.py`.

To install requirements, run:
```
pip install -r requirements.txt
```

Note: all [PDVC](https://github.com/ttengwang/PDVC) experiments are run with a separate conda environment as suggested in the PDVC codebase, so to compile the deformable attention layer.

## Data collection pipeline
To start, you may get a bunch of YouTube video IDs (that do not necessarily contain video chapters) 
and use [yt-dlp](https://github.com/yt-dlp/yt-dlp) to download descriptions from YouTube, e.g., ``yt-dlp https://www.youtube.com/watch?v=<VIDEO_ID> --write-description --skip-download``.

Then, assuming the descriptions are downloaded as `.txt` files in `SSD_DIR/chapters_descriptions`, you can run ``python collection/desc2chapters.py`` to extract chapters from descriptions.
The output file maps video IDs of user-chaptered videos to the chapter titles and timestamps.
You can then download the YouTube video content of videos with chapters with [yt-dlp](https://github.com/yt-dlp/yt-dlp), e.g., ``yt-dlp https://www.youtube.com/watch?v=<VIDEO_ID>``.

## Data downloading
***VidChapters-7M:*** We provide the dataset at [this link](https://antoyang.github.io/vidchapters7m.html).
You should download the annotations in `DATA_DIR/AllChapters`.
The license is in `LICENSE`.

***HowTo100M:*** We use a [sentencified version](https://www.robots.ox.ac.uk/~vgg/research/tan/) of the dataset.
You should download it in `DATA_DIR/howto100m`.
The license is [here](https://github.com/antoine77340/howto100m/blob/master/LICENSE).

***ViTT:*** Download it from [the data providers](https://github.com/google-research-datasets/Video-Timeline-Tags-ViTT). 
You will also need to download the mapping between 4-character IDs from YouTube-8M to YouTube video IDs. 
You should download these in `DATA_DIR/ViTT`.
The license is [here](https://github.com/google-research-datasets/Video-Timeline-Tags-ViTT/blob/main/LICENSE).

***YouCook2:*** Download it from [the data providers](http://youcook2.eecs.umich.edu/).
You should download these in `YouCook2`.
The license is [here](https://github.com/LuoweiZhou/ProcNets-YouCook2/blob/master/LICENSE).

## Data processing

### Visual Feature Extraction
We follow [FrozenBiLM](https://github.com/antoyang/FrozenBiLM) to extract CLIP ViT-L/14 @ 224 pixels features at 1 FPS for all videos. 
We store them in `SSD_DIR/chapters_clipvitl14_features`, one file per video, for VidChapters-7M/HowTo100M, and gather them in a single `.pth` file for all videos in YouCook2/ViTT.

### ASR Extraction
To extract ASR, given a `csv` file prepared like for the visual feature extraction and an `output_path` where to store the extracted ASR, we run on a single GPU:
```
python asr_extract/whisper_inference.py --csv=$csv --output_path=$output_path --faster
```
You may parallelize this over many jobs.
Note that this requires having downloaded the [Whisper Large-V2](https://github.com/openai/whisper) model weights in `<MODEL_DIR>`.

We then gather the extracted ASR into a single file `asr` by running:
```
python asr_extract/merge_asr_whisper.py $output_path DATA_DIR/AllChapters/whisper.pkl
```

To extract word-level timestamps and segment the ASR into sentences, we run on a single GPU:
```
python asr_extract/whisper_align.py --csv=$csv --asr=DATA_DIR/AllChapters/whisper.pkl --output_path=$align_output_path
```
You may parallelize this over many jobs.
Note that this requires having downloaded the alignment model weights for all languages from [WhisperX](https://github.com/m-bain/whisperX) in `<MODEL_DIR>`.

Finally, we merge the aligned ASR into a single file by running:
```
python asr_extract/merge_asr_whisper_align.py $align_output_path DATA_DIR/AllChapters/asr.pkl DATA_DIR/AllChapters/whisper.pkl
```

### Annotation files
To preprocess annotation files, use:
```
python preproc/chapters_to_dvc.py
python preproc/chapters_to_vmr.py
python preproc/vitt.py
python preproc/youcook.py
```

### Ethical considerations
To detect videos with NSFW frames or toxic chapter titles or ASR, we run on single GPUs:
```
python nsfw.py 
```
You may parallelize this over many jobs.
Note that this requires having downloaded [this NSFW classifier](https://github.com/LAION-AI/CLIP-based-NSFW-Detector) and the [Detoxify language model](https://github.com/unitaryai/detoxify).

## Training and evaluation

### Vid2Seq Pretraining on HowTo100M
Run:
```
python -m torch.distributed.launch --nproc_per_node 8 --use_env dvc.py --epochs=5 --fraction_warmup_steps=0.01 --lr=3e-4 --print_freq=1000 --save_dir=howto100m --combine_datasets htm --batch_size=8 --clip_max_norm=0.1
```
The pretrained checkpoint on HowTo100M can then be loaded with the argument `--load`. 

### Video Chapter Generation
For Vid2Seq, run:
```
python -m torch.distributed.launch --nproc_per_node 8 --use_env dvc.py --epochs=10 --lr=3e-4 --save_dir=chapters --combine_datasets chapters --combine_datasets_val chapters --batch_size=8 --batch_size_val=8 --clip_max_norm=0.1 --schedule="cosine_with_warmup"
```
Multiple baselines reported in the paper can also be found in `args.py`, e.g. using only visual or speech input with `--no_speech` or `--no_video`, or training only using ASR with `--gen_asr`.

For PDVC, run:
```
python PDVC/train.py --cfg_path PDVC/cfgs/chapter_clip_pdvc.yml --gpu_id 0 --epoch 5 --no_self_iou --lr 1e-4
```

For the text tiling + LLaMA zero-shot baseline, run:
```
python -m torch.distributed.launch --nproc_per_node 8 --use_env zs_speechvcg.py --combine_datasets=chapters --combine_datasets_val=chapters --save_dir=chapters_texttilingllama
```
Pass `--random` to the previous command to run the random baseline.

For the shot detection + BLIP-2 zero-shot baseline, run:
```
python -m torch.distributed.launch --nproc_per_node 8 --use_env zs_visualvcg.py --combine_datasets=chapters --combine_datasets_val=chapters --save_dir=chapters_shotdetectblip2
```

### Video Chapter Generation with Ground-Truth Boundaries
For Vid2Seq, run:
```
python -m torch.distributed.launch --nproc_per_node 8 --use_env vc.py --epochs=20 --lr=3e-4 --save_dir=chapters_vcggt --combine_datasets chapters --combine_datasets_val chapters --batch_size=64 --batch_size_val=64 --schedule="cosine_with_warmup" --max_input_tokens=256 --max_output_tokens=32
```

For the LLaMA zero-shot baseline, run:
```
python -m torch.distributed.launch --nproc_per_node 8 --use_env vc.py --model_name=<MODEL_DIR>/7BHF --save_dir=chapters_vcggt_zeroshotllama --combine_datasets chapters --combine_datasets_val chapters --batch_size_val=1 --max_input_tokens=256 --max_output_tokens=32 --eval
```
Pass `--random` to the previous command to run the random baseline.

For the BLIP-2 zero-shot baseline, run:
```
python -m torch.distributed.launch --nproc_per_node 8 --use_env vc.py --model_name=Salesforce/blip2-flan-t5-xl --save_dir=chapters_vcggt_zeroshotblip2 --combine_datasets chapters --combine_datasets_val chapters --batch_size_val=1 --max_input_tokens=256 --max_output_tokens=32 --eval
```

### Video Chapter Generation Grounding
For Moment-DETR, run:
```
bash moment_detr/moment_detr/scripts/chapters.sh --use_speech --max_v_l=1200 --downsample --clip_length=3 --lr=3e-4 --n_epoch=50 --max_es_cnt=50 --exp_id=chapters --bsz=256 --eval_bsz=256 --num_workers=16
```

For the CLIP zero-shot baseline, run:
```
python -m torch.distributed.launch --nproc_per_node 8 --use_env vcgr.py --save_dir=chapters_vcgr_clip --combine_datasets chapters --combine_datasets_val chapters
```

For the BERT zero-shot baseline, run:
```
python -m torch.distributed.launch --nproc_per_node 8 --use_env vcgr.py --save_dir=chapters_vcgr_bert --combine_datasets chapters --combine_datasets_val chapters --no_video
```

For the random zero-shot baseline, run:
```
python -m torch.distributed.launch --nproc_per_node 8 --use_env vcgr.py --save_dir=chapters_vcgr_random --combine_datasets chapters --combine_datasets_val chapters --random
```

### Dense Video Captioning
For Vid2Seq on YouCook2/ViTT, run:
```
python -m torch.distributed.launch --nproc_per_node 8 --use_env vc.py --epochs=40 --lr=3e-4 --save_dir=youcook --combine_datasets youcook --combine_datasets_val youcook --batch_size=2 --batch_size_val=2 --schedule="cosine_with_warmup"
python -m torch.distributed.launch --nproc_per_node 8 --use_env vc.py --epochs=20 --lr=3e-4 --save_dir=vitt --combine_datasets vitt --combine_datasets_val vitt --batch_size=2 --batch_size_val=2 --schedule="cosine_with_warmup"
```
The zero-shot evaluation can be simply done by using the arguments `--load=$load --eval`.