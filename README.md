# geo_mlvis
**This is the code repository for "Interactive visualization and representation analysis applied to Glacier Segmentation" project.**


## Motivation
In glacier segmentation, researchers focus on implementing model to automatically identify the glaciers from remote sensing data. However, the interpretation of the process is not enough. In this project, we aim to provide a comprehensive visual interpretation of glacier segmentation via interactive visualization and representation analysis. We develop a [shiny app](https://bruce-zheng.shinyapps.io/glacier_segmententation/) for error analysis where users could easily detect potential issues of the predictions. Here we display two examples below, one for the screenshot of the shinyapp and another one for the visual representation of one satellite image.


## Code Guidance

Raw data can be downloaded as .tiff files using `Data_Preview_Download/download.ipynb`.

Data is modeled with a U-Net in `Train_Pred/train_gpu.py` and predicted with `Train_Pred/save_preds.py`.

### Python-only setup (no R/Shiny)

This repository can be used end-to-end from Python/Jupyter only. Use the conda environment for best geospatial support; a pip fallback is provided for CPU-only.

#### Option A: Conda (Windows, NVIDIA GPU recommended)

1. Install Miniconda or Anaconda.
2. Create the environment:
   ```bash
   conda env create -f environment.yml
   conda activate geo-mlvis
   ```
3. Verify PyTorch and CUDA:
   ```python
   import torch; print(torch.__version__, torch.cuda.is_available())
   ```
   - If `False` on a CUDA-capable machine, install matching CUDA toolkit/driver or switch to CPU by removing `pytorch-cuda` from `environment.yml` and reinstalling.
4. Register the kernel for Jupyter:
   ```bash
   python -m ipykernel install --user --name geo-mlvis --display-name "Python (geo-mlvis)"
   ```

#### Option B: Pip (CPU-only fallback)

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name geo-mlvis --display-name "Python (geo-mlvis)"
```

#### Notes for multispectral TIFFs
- Use `rasterio`/`rioxarray` to read and stack bands (Blue, Green, Red, SWIR, Thermal).
- Models in `Train_Pred/train_gpu.py` expect tensors with channels-first shape `(C, H, W)`.
- Ensure consistent spatial resolution/projection before stacking.





| ![25811638647397_ pic_hd](https://user-images.githubusercontent.com/53232883/144722760-d1a153f8-609c-46f5-b1a5-6dd5b095d43a.jpg) | 
|:--:| 
| *Figure 1: Screenshot of shiny app* |


| ![acts-1-1](https://user-images.githubusercontent.com/53232883/144722811-04a40069-fc36-4ae5-81a3-ef39ca130784.png) | 
|:--:| 
| *Figure 2: Visual representations of one satellite image (Activations of one satellite image across five convolutional layers of the U-Net model. Rows represent the first, third, fifth and seventh downsampling convolutional layers, the first and third upsampling convolutional layers, the last pooling layer and the second middle convolutional layer. For each layer, we randomly plot eight activations in grayscale. We observe that the activations capture basic features at the first layer, become more blurred as down sampling goes deeper, and becomes clearer at the middle layer. Comparing activations in the same layer, we find that the activations look alike in the first layer, but different in the middle layer.)* |





