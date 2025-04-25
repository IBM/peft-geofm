# Parameter-Efficient Fine-Tuning (PEFT) for GeoFMs

This repository explores Parameter-Efficient Fine-Tuning (PEFT) techniques with Geospatial Foundation Models (GeoFM). It contains experimental setups, configuration files, and scripts used for our paper:

Marti Escofet, F., Blumenstiel, B., Scheibenreif, L., Fraccaro, P., and Schindler, K. (2025). 
Fine-tune Smarter, Not Harder: Parameter-Efficient Fine-Tuning for Geospatial Foundation Models.
[arXiv preprint arXiv:2504.17397](https://arxiv.org/abs/2504.17397)

![PEFT_methods.png](assets%2FPEFT_methods.png)

We integrated LoRA, Visual Prompt Tuning (VPT), and ViT-Adapter into [TerraTorch](https://github.com/IBM/terratorch), a fine-tuning toolkit for GeoFMs.
Our results show that LoRA matches or surpasses the performance of full fine-tuning on most datasets while reducing the memory consumption by 30%. 

![radar.png](assets%2Fradar.png)

Furthermore, we propose new train text splits for HLS Burn Scars and reBEN which we share in [datasets_splits](datasets_splits).
Specifically, our HLS Burn Scars split ensures non-overlapping samples between splits to avoid data leakage and includes a validation and test split rather than only validation samples.
For reBEN (BigEarthNet 2.0), we created a smaller subset called reBEN 7k similar to BEN-GE 8k for BEN 1.0. 
Our 7k version reduces label biases, enables faster experiments, and includes a geographic hold-out set (Austria and Ireland) for out-of-distribution (OOD) experiments.  

## PEFT in TerraTorch

We integrated all PEFT methods directly into TerraTorch. We shortly describe the required changes to use PEFT with a standard fine-tuning config.
All settings work with the `EncoderDecoderFactory` and are passed as additional parameters in `model_args`. 

### LoRA

You can provide a `peft_config` parameter for using LoRA. 
This setting was tested with LoRA for Prithvi and Clay, but may also work for other models and methods from the [PEFT](https://github.com/huggingface/peft) package. 
You can specify the name pattern of the LoRA modules in `target_modules`.
LoRA is originally applied only on the queries (Q) and value (V) layers of an attention block.
Often queries, values, and keys are combined in a single linear layer (e.g., in `timm` which is used by Prithvi).
Specify this layer in `replace_qkv` to split the matrix of these layers up into separate linear layers.
Here is an example from [lora.yaml](configs%2Fpeft%2Fprithvi_eo_v2_300%2Fburn_scars%2Flora.yaml). 
Clay works with a similar setting, see [lora.yaml](configs%2Fpeft%2Fclay_gw_clay_bands%2Fm_cashew_plantation%2Flora.yaml).

```yaml
    model_factory: EncoderDecoderFactory
    model_args:
      backbone: prithvi_eo_v2_300
      ...
      peft_config:
        method: LORA
        replace_qkv: qkv
        peft_config_kwargs:
          target_modules:
            - qkv.q_linear
            - qkv.v_linear
            - mlp.fc1
            - mlp.fc2
          lora_alpha: 16
          r: 16
      ...
```

### VPT

Visual Prompt Tuning (VPT) is integrated into the backbone of Prithvi and Clay and adds a few extra parameters:  

```yaml
    model_factory: EncoderDecoderFactory
    model_args:
      backbone: prithvi_eo_v2_300  # Similar setting for clay_v1_base
      ...
      backbone_vpt: true
      backbone_vpt_n_tokens: 100
      backbone_vpt_dropout: 0.1
      ...
```

VPT is implemented in the main branch but not included in the latest TerraTorch release.
You can install it with:
```bash
pip install git+https://github.com/IBM/terratorch.git@main
```

### ViT Adapter

For adding the ViT Adapter to Prithvi models, you simply need to set `backbone_vit_adapter` to `True`. 
TerraTorch automatically add the adapter layers to the model.

```yaml
    model_factory: EncoderDecoderFactory
    model_args:
      backbone: prithvi_eo_v2_300
      ...
      backbone_vit_adapter: true
      ...
```

## Setup

Download or clone this repo and create a new environment with TerraTorch.

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install terratorch==1.0
```

To install the necessary dependencies, run:

```bash
pip install -e .
```

### Additional Dependencies for ViT-Adapter Configurations

If you plan to use ViT-Adapter configurations (which require CUDA to compile deformable attention), install the following package:

```bash
pip install "MultiScaleDeformableAttention @ git+https://github.com/fundamentalvision/Deformable-DETR.git#subdirectory=models/ops"
```

This has been tested with the following versions:
- GCC 11.4.0 / 13.3.0
- G++ 11.4.0 / 13.3.0
- CUDA Toolkit 12.8.1

If you encounter any issues, consider using these versions.

## Repository Structure

- **[configs](configs)**: Contains configuration files for various experimental setups.
    - **[datamodules](configs/datamodules/)**: Configuration files for different datamodules. Naming conventions:
        - `no_metadata`: Only includes the image, without metadata. Used for DeCUR and Prithvi without metadata.
        - `clay_gw`: Includes GSD and wavelengths for Clay. Required for Clay model usage.
        - `clay_tlgw`: Includes GSD, wavelengths, temporal, and location metadata for Clay.
        - `prithvi_tl`: Includes temporal and location metadata for Prithvi.
        - `prithvi_bands`: Includes only the 6 HLS bands.
        - `clay_bands`: Includes all 10 Sentinel-2 (S2) bands used in Clay pre-training.
        - `decur_bands`: Includes all available S2 bands. DeCUR was pre-trained on 13 L1C bands, while some datasets only contain 12 bands.
    - **[decoders](configs/decoders/)**: Configuration files for decoder experiments, including Fully Convolutional Network, UperNet, and UNet configurations for each dataset and model.
    - **[metadata](configs/metadata/)**: Configuration files for metadata-related experiments.
    - **[peft](configs/peft/)**: Configuration files for PEFT experiments, following the same naming conventions as `datamodules`.
- **[datasets_splits](datasets_splits)**: Contains predefined dataset splits for Sen1Floods11, Burn Scars, and reBEN datasets.
- **[src](src)**: Includes datamodules and preprocessing scripts, such as the script for merging Sentinel-2 bands in the reBEN dataset.

## Fine-tuning

We use the TerraTorch CLI for training and testing. The learning rates are selected using hyperparameter optimization with [TerraTorch-iterate](https://github.com/IBM/terratorch-iterate).
Each configuration requires specifying the dataset directory and the logging directory. You can set them in the YAML files or provide them via command-line arguments as follows:

```bash
terratorch fit --config <path_to_config> --data.data_root <path_to_corresponding_dataset> --trainer.default_root_dir <path_to_logger_folder>
```

For testing, provide the model checkpoint with `--ckpt_path`: 

```bash
terratorch test --config <path_to_config> --data.data_root <path_to_corresponding_dataset> --trainer.default_root_dir <path_to_logger_folder> --ckpt_path <path_to_model_checkpoint>
```

E.g., you can fine-tune and test Prithvi 2.0 with LoRA on Burn Scars with:
```bash
terratorch fit --config configs/peft/prithvi_eo_v2_300/burn_scars/lora.yaml --data.data_root data/hls_burn_scars/samples --trainer.default_root_dir output/prithvi_eo_v2_300/burn_scars/lora

terratorch test --config configs/peft/prithvi_eo_v2_300/burn_scars/lora.yaml --data.data_root data/hls_burn_scars/samples --trainer.default_root_dir output/prithvi_eo_v2_300/burn_scars/lora --ckpt_path output/prithvi_eo_v2_300/burn_scars/lora/version_0/checkpoints/epoch=80.ckpt  
```

## Datasets

### Burn Scars
- **Download**: [Hugging Face](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars)
- **Setup**: Combine the training and validation samples into a single directory and provide this path in the configuration.

### Sen1Floods11
- **Download**: [GitHub](https://github.com/cloudtostreet/Sen1Floods11)
- **Setup**: Use the decompressed folder's parent directory (containing `v1.1`) for the configuration.

### Cashew Plantation
- **Download**: [Hugging Face](https://huggingface.co/datasets/recursix/geo-bench-1.0)
- **Setup**: Use the parent directory of the decompressed dataset in the configuration.

### SA Crop Type
- **Download**: [Hugging Face](https://huggingface.co/datasets/recursix/geo-bench-1.0)
- **Setup**: Use the parent directory of the decompressed dataset in the configuration.

### reBEN
- **Download**: [BigEarthNet](https://bigearth.net/)
- **Setup**:
  1. Download the Sentinel-2 (S2) files.
  2. Merge S2 bands using the following script:

     ```bash
     python src/peft_geofm/join_reben_s2_bands.py '[<split_file1>, <split_file2>, ...]' <path_to_reBEN>
     ```

     Here, `<path_to_reBEN>` should be the directory containing **BigEarthNet-S2**. This will create a `S2_merged` directory.
  3. Download the `Reference_Maps` directory and `metadata.parquet` files and place them in `<path_to_reBEN>`.
  4. Use `<path_to_reBEN>` in your configuration.

## Citation

If our research is helpful for you, consider citing our [paper](https://arxiv.org/abs/2504.17397):

```text
@article{martiescofet2025peft,
  title={Fine-tune Smarter, Not Harder: Parameter-Efficient Fine-Tuning for Geospatial Foundation Models},
  author={Marti-Escofet, Francesc and Blumenstiel, Benedikt and Scheibenreif, Linus and Fraccaro, Paolo and Schindler, Konrad},
  journal={arXiv preprint arXiv:2504.17397},
  year={2025}
}
```

If you use TerraTorch, please cite the [paper](https://arxiv.org/abs/2503.20563):
```text
@article{gomes2025terratorch,
  title={TerraTorch: The Geospatial Foundation Models Toolkit},
  author={Gomes, Carlos and Blumenstiel, Benedikt and Almeida, Joao Lucas de Sousa and de Oliveira, Pedro Henrique and Fraccaro, Paolo and Marti-Escofet, Francesc and Szwarcman, Daniela and Simumba, Naomi and Kienzler, Romeo and Zadrozny, Bianca},
  journal={arXiv preprint arXiv:2503.20563},
  year={2025}
}
```

## IBM Public Repository Disclosure

This project is licensed under the [Apache 2.0 License](LICENSE).

All content in this repository including code has been provided by IBM under the associated open source software license and IBM is under no obligation to provide enhancements, updates, or support. IBM developers produced this code as an open source project (not as an IBM product), and IBM makes no assertions as to the level of quality nor security, and will not be maintaining this code going forward.
