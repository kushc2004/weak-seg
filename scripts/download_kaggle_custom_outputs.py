import kaggle

# Replaces the web download and pulls only your specific file
kaggle.api.kernels_output_file(
    'kushchaudhari/weakseg', 
    'weak-seg/outputs/checkpoints/classifier_plain.pt', 
    path='./'
)


import kaggle

# Initialize and authenticate the Kaggle API
api = kaggle.KaggleApi()
api.authenticate()

# Download the complete output structure (including your nested .pt file)
api.kernels_output(
    'kushchaudhari/weakseg', 
    path='./'
)
