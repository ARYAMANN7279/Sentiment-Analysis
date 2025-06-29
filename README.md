Multimodal Sentiment Analysis
A comprehensive machine learning project that combines text and image data for multiclass sentiment analysis using early fusion techniques. This project implements a multimodal deep learning model that analyzes memes and social media content across multiple dimensions including humor, sarcasm, offensiveness, motivational content, and overall sentiment.

🎯 Project Overview
This project employs early fusion techniques for multimodal sentiment analysis, where text (OCR-extracted and corrected) and image features are concatenated and jointly processed to predict multiple sentiment-related classifications. The model leverages the power of both textual understanding through DistilBERT and visual feature extraction through ResNet-18.

Key Features
Multimodal Architecture: Combines text and image modalities using early fusion
Multi-task Learning: Simultaneous prediction of 5 different classification tasks
State-of-the-art Models: DistilBERT for text processing and ResNet-18 for image analysis
Comprehensive Analysis: Includes data exploration, visualization, and model evaluation
📊 Tasks and Classifications
The model performs classification across five different dimensions:

Task	Classes	Description
Humor	4 classes	not_funny, funny, very_funny, hilarious
Sarcasm	4 classes	not_sarcastic, general, twisted_meaning, very_twisted
Offensive	4 classes	not_offensive, slight, very_offensive, hateful_offensive
Motivational	2 classes	not_motivational, motivational
Overall Sentiment	5 classes	very_negative, negative, neutral, positive, very_positive
🏗️ Architecture
Model Components
Text Encoder: DistilBERT (distilbert-base-uncased)

Processes OCR-extracted and corrected text
Outputs 768-dimensional embeddings
Reduced to 128 dimensions via linear layer
Image Encoder: ResNet-18 (pre-trained)

Processes 224x224 RGB images
Modified final layer outputs 128 dimensions
Fusion Layer: Early fusion via concatenation

Combined features: 256 dimensions (128 text + 128 image)
Classification Heads: Separate linear layers for each task

Independent prediction heads for each of the 5 tasks
Training Strategy
Multi-task Learning: Sequential training on each task
Optimizer: Adam with learning rate 1e-3
Loss Function: CrossEntropyLoss for each classification task
Device: CUDA-enabled training when available
📁 Dataset Structure
Multimodal_dataset_assignment3/
├── labels.csv              # Main dataset file with labels and text
└── images/                 # Directory containing all images
    ├── image_1.jpg
    ├── image_2.jpeg
    └── ...
Dataset Features
Images: Various formats (jpg, jpeg, png, JPG)
Text Data:
text_ocr: Raw OCR-extracted text
text_corrected: Manually corrected text
Labels: Both categorical and numerical encodings
Metadata: Image names and full paths
🚀 Quick Start
Install Dependencies

pip install torch torchvision transformers pandas numpy pillow scikit-learn matplotlib seaborn
Prepare Dataset

Ensure your dataset follows the structure shown above
Update data_path and image_folder variables in the notebook
Train the Model

# Run the Jupyter notebook
jupyter notebook train.ipynb
Model Inference

# Load trained model
model = MultimodalModel(num_classes_list=[4, 4, 4, 2, 5])
model.load_state_dict(torch.load("Multimodal_Model.pth"))
model.eval()
📈 Model Performance
The model is evaluated using accuracy metrics for each classification task:

Humor Classification: Multi-class accuracy
Sarcasm Detection: Multi-class accuracy
Offensive Content Detection: Multi-class accuracy
Motivational Content Classification: Binary accuracy
Overall Sentiment Analysis: Multi-class accuracy
Training Configuration
Epochs: 10
Batch Size: 32
Train/Test Split: 70/30
Image Size: 224x224 pixels
Text Max Length: Model default (DistilBERT)
🔧 Code Structure
Core Classes
MemotionDataset: Custom PyTorch Dataset

Handles multimodal data loading
Applies image transformations
Tokenizes text input
MultimodalModel: Main neural network architecture

Implements early fusion strategy
Separate classification heads for each task
Forward pass combining text and image features
Key Functions
Data Loading: CSV reading and image path mapping
Preprocessing: Image transformations and text tokenization
Training Loop: Multi-task learning with separate optimization
Evaluation: Accuracy calculation per task
📊 Data Analysis Features
The notebook includes comprehensive exploratory data analysis:

Class Distribution Visualization: Bar plots for each classification task
Text Analysis: Word count analysis per sentiment class
Data Quality Checks: Missing value detection and handling
Label Encoding: Categorical to numerical conversion
🎨 Visualization
The project includes various visualization components:

Class frequency distributions
Box plots for text length analysis
Training progress tracking
Model performance metrics per task
💾 Model Persistence
The trained model is saved as:

File: Multimodal_Model.pth
Format: PyTorch state dictionary
Loading: Compatible with the MultimodalModel class
🔄 Future Enhancements
Potential improvements and extensions:

Advanced Fusion Techniques: Implement attention-based fusion
Model Architecture: Experiment with larger pre-trained models
Data Augmentation: Add image and text augmentation techniques
Hyperparameter Tuning: Systematic optimization of training parameters
Cross-validation: Implement k-fold cross-validation
Additional Metrics: Include precision, recall, and F1-scores
📚 References
DistilBERT: Sanh, V., et al. "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter." (2019)
ResNet: He, K., et al. "Deep residual learning for image recognition." (2016)
Multimodal Learning: Baltrusaitis, T., et al. "Multimodal machine learning: A survey and taxonomy." (2018)
🤝 Contributing
Contributions are welcome! Please feel free to:

Report bugs and issues
Suggest new features or improvements
Submit pull requests
Share experimental results
This project represents a comprehensive approach to multimodal sentiment analysis, combining the latest advances in natural language processing and computer vision for enhanced understanding of multimedia content.

About
No description, website, or topics provided.
Resources
 Readme
 Activity
Stars
 0 stars
Watchers
 1 watching
Forks
 0 forks
Report repository
Releases
No releases published
Packages
No packages published
Languages
Jupyter Notebook
100.0%
Footer
© 2025 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Docs
Contact
Manage cookies
Do not share my personal information
