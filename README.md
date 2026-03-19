# RelationField: Relate Anything in Radiance Fields
## Modifications from Original Repository

The following changes were made on top of the original [RelationField](https://github.com/boschresearch/RelationField) codebase.

### `datasets/replica_preprocess.py`
- Added `--max_dataset_size` command-line argument so the number of training frames can be set without editing the source. Default remains 200.

### `datasets/subsample_replica.py
- Utility script for subsampling an already-preprocessed scene to fewer frames. Creates a new output directory with symlinks to the original images and a downsampled `transforms.json`. Used to generate reduced-view variants (100 / 50 frames) for the reconstruction-quality ablation experiment.

### `eval_relation_queries.py`
- **Accuracy metrics**: added `loc_acc` (localization accuracy, whether the heatmap argmax falls inside the GT mask), `iou_top10` (IoU between top-10% activation pixels and GT mask), and `score_ratio` (mean activation inside GT vs outside) computed against GPT-generated segmentation masks.
- **GT mask loading**: added `load_gt_masks()` which reads per-frame `.npy` mask files and `_tag2class.json` label files produced by the GPT preprocessing step.
- **Class aliases**: added `CLASS_ALIASES` to handle synonym labels (e.g. `screen`/`monitor`, `sofa`/`couch`) when looking up GT masks.
- **Per-query frame support**: each query can specify its own render frame (`"frame"` key in `QUERIES`); a `--use_query_frames` flag enables this mode.
- **Frame cache**: frames are rendered/loaded once and reused across queries that share the same frame index.
- **Original photo background**: uses the downscaled photo from `images_4/` as the RGB background instead of a NeRF render, saving VRAM and render time.
- **Predicate embedding fix**: relation queries now embed the short predicate string (e.g. `"attached to"`) rather than the full sentence, matching the format used during training.
- **Raw relation relevancy**: passes `relation_relevancy_raw` (pre-distance-scaling) through the render pipeline so accuracy metrics are not biased by the spatial distance weighting.
- **Adaptive heatmap thresholding**: `overlay_heatmap` now chooses the threshold from robust quantiles of the activation distribution separately for object and relation queries.
- **Refined query set and positions**: `QUERIES` and `POSITIONS` updated with cluster-centroid–calibrated 3D coordinates and cleaner query/predicate pairs.
- **CSV output extended**: summary CSV now includes `target`, `frame`, `raw_max`, `raw_mean`, `loc_acc`, `iou_top10`, `score_ratio` columns.

### `relationfield/relationfield_model.py`
- **`relation_embedding_from_points`**: added `relation_semantic_feat` branch — when enabled, OpenSeg semantic embeddings at the query and field positions are concatenated with the positional encodings before the relation MLP, giving the network access to semantic context.
- **`get_outputs_for_points`**: made `clip_net` access conditional with a `hasattr` guard; falls back to `openseg_net` if `clip_net` is not present, fixing a crash on models trained without a separate CLIP head.
- **`relation_click_output`**: now also stores `relation_samples` (per-sample embeddings before volume rendering) alongside the already-rendered `relation_map`, enabling sample-level inspection.
- Minor: replaced `.view()` with `.reshape()` where the input tensor may not be contiguous.

### `relationfield/relationfield_interaction.py`
- **`get_max_across_relation`**: uses `renderer_mean` (the model's volume-rendering accumulator) instead of a hand-rolled normalized-weight sum, making relation relevancy compositing consistent with the rest of the pipeline; also reads `relation_samples` when available instead of `relation_map`.
- **`query_position_embedding`**: `clip_net` access is now guarded with `hasattr`; falls back to `openseg_net` for models that expose only OpenSeg features.
- Minor: `.view()` → `.reshape()` for non-contiguous tensors.

### `extract_scene_graph.py` 
- Standalone script that loads a trained RelationField checkpoint, samples points from the scene mesh, clusters them into object instances via DBSCAN, labels each instance using OpenSeg cosine similarity, predicts pairwise relations via the relation field and Jina embeddings, and saves a scene graph visualization.

## Troubleshooting
**Note:** RelationField requires ~32GB of memory during training.  If your system has lower computational resources, consider reducing the number of training rays.

## Installation

#### Install NerfStudio

```
conda create --name relationfield -y python=3.10
conda activate relationfield
python -m pip install --upgrade pip
```

### Install cuda, torch, etc

```
conda install nvidia/label/cuda-11.8.0::cuda
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### Install SAM

```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Install RelationField

```
git clone https://github.com/boschresearch/relationfield
cd relationfield
python -m pip install -e .
```

## Data preparation and Foundation Models

The datasets and saved NeRF models require significant disk space.
Let's link them to some (remote) larger storage:

```
ln -s path/to/large_disk/data data
ln -s path/to/large_disk/models models
ln -s path/to/large_disk/outputs outputs
```

Download the OpenSeg feature extractor model from [here](https://drive.google.com/file/d/1DgyH-1124Mo8p6IUJ-ikAiwVZDDfteak/view?usp=sharing) and unzip it into `./models`.
Download the SAM from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and unzip it into `./models`.

### Replica Dataset

Download the Replica dataset pre-processed by [NICE-SLAM](https://pengsongyou.github.io/nice-slam) and transform it into [nerfstudio](https://docs.nerf.studio) format using these steps:

```
cd data
wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
unzip Replica.zip
cd ..
python datasets/replica_preprocess.py --data <root-replica-folder> --output_dir <nerfstudio-output> --max_dataset_size 200
```

Use `--max_dataset_size` to control how many frames are sampled (default: 200). Use `-1` for all frames. Example for reduced-view experiments:

```
python datasets/replica_preprocess.py --data <root-replica-folder> --output_dir <nerfstudio-output-v100> --max_dataset_size 100
python datasets/replica_preprocess.py --data <root-replica-folder> --output_dir <nerfstudio-output-v50> --max_dataset_size 50
```

To subsample an already-preprocessed scene to fewer frames (for reduced-view experiments):

```
python datasets/subsample_replica.py --src <nerfstudio-output>/<scene> --dst <nerfstudio-output-v100>/<scene> --n 100
python datasets/subsample_replica.py --src <nerfstudio-output>/<scene> --dst <nerfstudio-output-v50>/<scene> --n 50
```

### RIO10 Dataset

Download the RIO10 dataset from [here](https://github.com/WaldJohannaU/RIO10?tab=readme-ov-file) and transform it into [nerfstudio](https://docs.nerf.studio) format using these steps:

```
python datasets/rio_preprocess.py --data <root-rio-folder> --output_dir <nerfstudio-output>
```

To process a single sequence (e.g. `seq01_02`):

```
python datasets/rio_preprocess.py --data <root-rio-folder>/seq01_02 --output_dir <nerfstudio-output>
```

### Preprocess GPT Captions

To caption the preprocessed dataset with GPT run:

```
export OPEN_API_KEY=YOUR_API_KEY
python datasets/preprocess_dataset_gpt.py --data_dir [PATH]
```

In case a one of the captioning steps fails you can manually correct it and run (this happens very rarely):

```
python datasets/preprocess_dataset_gpt.py --data_dir [PATH] --redo img_id1.png,...,img_idk.png
```

## Running RelationField

This repository creates a new Nerfstudio method named "relationfield". To train with it, run the command:

```
ns-train relationfield --data [PATH]
```

To view the optimized NeRF, you can launch the viewer separately:

```
ns-viewer --load-config outputs/path_to/config.yml
```

Interact with the viewer to visualize relationships:

![RelationField visualization](figs/image_3.png)



### Utilizing depth data
RelationField support depth supervision for improved and faster convergence of the NeRF geometry. To activate make sure that depth data is available for your data and run:
```
export NERFACTO_DEPTH=True
ns-train relationfield --data [PATH]
```

## Running RelationField with Gaussian Splatting geometry!
Although RelationField's relation field is optimized using NeRF geometry, it can be
used to relate gaussians in 3D!
```
ns-train relationfield-gauss --data [PATH] --pipeline.relationfield-ckpt outputs/path_to/config.yml
```
## Extracting a 3D Scene Graph

After training, extract a lightweight scene graph from the trained model:

```
python extract_scene_graph.py \
    --config outputs/<scene>/relationfield/<timestamp>/config.yml \
    --out_dir sg_results/<scene>
```

For example:

```
python extract_scene_graph.py \
    --config outputs/replica_office0/relationfield/2026-03-09_231057/config.yml \
    --out_dir sg_results/replica_office0

python extract_scene_graph.py \
    --config outputs/seq01_02/relationfield/2026-03-14_015920/config.yml \
    --out_dir sg_results/seq01_02
```


In `relationfield/relationfield/relationfield_config.py`, you can adjust the following parameters (lines 42-46):

```python
train_num_rays_per_batch=4096,
eval_num_rays_per_batch=4096, 
pixel_sampler=RelationFieldPixelSamplerConfig(
    num_rays_per_image=256,  # 4096/256 = 16 images per batch
),

```

