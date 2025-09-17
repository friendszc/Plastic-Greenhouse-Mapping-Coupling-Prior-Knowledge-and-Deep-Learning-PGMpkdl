# Plastic-Greenhouse-Mapping-Coupling-Prior-Knowledge-and-Deep-Learning-PGMpkdl
This project provides the implementation of a novel framework for large-scale plastic greenhouse mapping. By coupling prior knowledge with deep learning methods, the framework addressed the labeling bottleneck and enables efficient large-scale extraction of plastic greenhouses.
# Overview
This repository provides the implementation of a novel framework for large-scale plastic greenhouse mapping using Sentinel-2 imagery.
- Step 1: Construct pseudo-labels based on prior knowledge.
- Step 2: Train deep learning models with pseudo-labels.
- Step 3: Validate both prior knowledge and deep learning results.
- Step 4: Apply the trained model for large-scale mapping.
The framework addresses the labeling bottleneck and enables efficient and scalable extraction of plastic greenhouses.
# Usage Instructions

1. **Download data**  
   Download the dataset from [Zenodo](https://zenodo.org/records/17139012) and place it under the `/data` folder in the repository.

2. **Construct pseudo-labels based on prior knowledge**  
   Run the following scripts in order:
   - `calAPGI.py`: Calculate the APGI index from the prepared Sentinel-2 data.  
   - `classify_APGI.py`: Classify plastic greenhouses (PG) / non-PG using thresholds and predefined masks.  
   - `InputDataGenerator.py`: Generate deep learning input samples (RGB data paired with pkb labels) for training and validation.

3. **Train deep learning models**  
   Run `main.py` to start training. Model hyperparameters can be adjusted in `config.py` and `train.py`. Trained models will be saved in `.pth` format.

4. **Predict test set**  
   Use `predict.py` with the trained model to generate predictions. Results are saved in the same folder as the model.

5. **Validate results**  
   Run `Evaluation.py` to evaluate both prior knowledge and deep learning outputs. Metrics such as accuracy and recall will be output for analysis.

6. **Apply model for large-scale mapping**  
   Use `Predict_tif.py` to generate maps for other regions using the trained model.

# Citation 
If you use the PGMpk-dl method in your research, please cite: Zhou, C., Huang, J., Xiao, Y., Du, M., Li, S., 2024. A novel approach: Coupling prior knowledge and deep learning methods for large-scale plastic greenhouse extraction using Sentinel-1/2 data. Int. J. Appl. Earth Obs. Geoinf. 132, 104073. https://doi.org/10.1016/j.jag.2024.104073
