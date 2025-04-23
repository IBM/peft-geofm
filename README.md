# PEFT-GeoFM

PEFT-GeoFM is a repository for exploring Parameter-Efficient Fine-Tuning (PEFT) techniques with Geospatial Foundation Models. It contains experimental setups, configuration files, and scripts used for [link to paper].

## Installation

### Install Required Packages

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

- **[plots](plots)**: Contains visualizations and plots generated from experiments.
- **[assets](assets)**: Stores figures and other media assets used in the project.
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

### Configuration Setup

Each configuration requires specifying the dataset directory and the logging directory. You can set them in the YAML files or provide them via command-line arguments as follows:

```bash
terratorch fit --config <path_to_config> --data.data_root <path_to_corresponding_dataset> --trainer.default_root_dir <path_to_logger_folder>
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

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

TODO Add note to files: `Copyright contributors to the PEFT-GeoFM project`

## Citation

TODO Add citation


## IBM Public Repository Disclosure

All content in these repositories including code has been provided by IBM under the associated open source software license and IBM is under no obligation to provide enhancements, updates, or support. IBM developers produced this code as an open source project (not as an IBM product), and IBM makes no assertions as to the level of quality nor security, and will not be maintaining this code going forward.
