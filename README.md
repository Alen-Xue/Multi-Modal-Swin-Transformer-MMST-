Project Overview
This project implements a multi-modal classification model that combines image data (processed by Swin Transformer) and structured data (processed by a custom embedding and attention-based fusion module). The model is designed for tasks where both visual and tabular data are available, such as disaster impact assessment, where images of affected areas and structured data (e.g., evacuation zones, building age, etc.) are used to predict outcomes.

Key features:

Multi-Modal Fusion: Combines image features (from Swin Transformer) and structured data features (from FModel) using a weighted fusion strategy.
Focal Loss: Addresses class imbalance by focusing on hard-to-classify samples.
Attention Mechanism: Uses multi-head attention to capture interactions between structured features.
Model Components
FocalLoss
Overview
FocalLoss is a custom loss function that extends the cross-entropy loss to address class imbalance by down-weighting easy-to-classify samples and focusing on hard-to-classify ones.

Parameters
alpha (float, default: 1): Weighting factor to balance positive and negative samples.
gamma (float, default: 2): Focusing parameter to emphasize hard samples. Higher values increase focus on difficult samples.
logits (bool, default: False): Whether the input is raw logits. If True, applies F.cross_entropy directly; if False, applies softmax first.
reduce (bool, default: True): Whether to reduce the loss. If True, returns the mean loss; if False, returns per-sample losses.
Inputs and Outputs
Inputs:
inputs (Tensor): Model predictions, shape (batch_size, num_classes).
targets (Tensor): Ground truth labels, shape (batch_size,).
Outputs:
If reduce=True: Scalar (mean Focal Loss).
If reduce=False: Tensor of shape (batch_size,) (per-sample Focal Loss).
Forward Logic
Compute the base cross-entropy loss:
If logits=True, use F.cross_entropy(inputs, targets, reduction='none').
If logits=False, apply F.softmax(inputs, dim=1) first, then compute cross-entropy.
Compute the confidence score: pt = torch.exp(-BCE_loss).
Apply the Focal Loss formula: F_loss = alpha * (1-pt)**gamma * BCE_loss.
Reduce the loss if reduce=True (using torch.mean).
Usage Scenario
Suitable for classification tasks with class imbalance, such as medical image classification or object detection.
BasicModel
Overview
BasicModel is a feature embedding and fusion module that processes multiple discrete features (e.g., Evacuation_Zone, YearBuilt) by embedding them into a high-dimensional space and applying multi-head attention for feature interaction.

Parameters
chioce_matrix (list): Feature selection mask. A value of 1 includes the feature, 0 excludes it.
embedding_size (list, default: [6, 113, 38, 115, 124, 120, 14]): Vocabulary sizes for each feature.
num_heads (int, default: 5): Number of heads in the multi-head attention mechanism.
Architecture
Embedding Layers: Seven nn.Embedding layers, each mapping a discrete feature to a 1000-dimensional vector.
Multi-Head Attention: A MultiHeadAttention module (model_dim=1000, num_heads=num_heads) to capture feature interactions.
Inputs and Outputs
Inputs:
Evacuation_Zone (Tensor): Evacuation zone, shape (batch_size,).
YearBuilt (Tensor): Year of construction, shape (batch_size,).
EstimatedValue_level (Tensor): Estimated value level, shape (batch_size,).
dist_track_line (Tensor): Distance to track line, shape (batch_size,).
dist_track_landfall (Tensor): Distance to landfall, shape (batch_size,).
wind_mean (Tensor): Mean wind speed, shape (batch_size,).
flood_mean (Tensor): Mean flood intensity, shape (batch_size,).
Output:
e (Tensor): Fused feature vector, shape (batch_size, 1000).
Forward Logic
Embed each feature using the corresponding embedding layer, resulting in seven tensors of shape (batch_size, 1000).
Concatenate the embeddings along the second dimension: torch.cat((e1, e2, ..., e7), dim=1), shape (batch_size, 7, 1000).
Select features based on chioce_matrix, resulting in shape (batch_size, num_selected_features, 1000).
Apply multi-head attention: e, _ = self.att1(a_e, a_e, a_e), shape (batch_size, num_selected_features, 1000).
Aggregate features by taking the mean along the second dimension: e = torch.mean(e, dim=1), shape (batch_size, 1000).
Usage Scenario
Ideal for tasks requiring embedding and fusion of multiple discrete features with attention-based interaction.
FModel
Overview
FModel builds on BasicModel to perform feature fusion and classification for structured data.

Parameters
embedding_size (list, default: [6, 113, 38, 115, 124, 120, 14]): Vocabulary sizes for each feature.
chioce_matrix (list, default: [1,1,1,1,0,1,0]): Feature selection mask.
num_classes (int, default: 3): Number of classes for classification.
num_heads (int, default: 5): Number of heads in the multi-head attention mechanism.
Architecture
Embedding Module: Uses BasicModel for feature embedding and fusion.
Linear Layer: nn.Linear(1000, num_classes) for classification.
Inputs and Outputs
Inputs: Same as BasicModel.
Output:
x (Tensor): Classification logits, shape (batch_size, num_classes).
Forward Logic
Use BasicModel to embed and fuse the structured features: e = self.embedding(...), shape (batch_size, 1000).
Pass the fused features through the linear layer: x = self.linear(e), shape (batch_size, num_classes).
Usage Scenario
Suitable for classification tasks involving structured data with multiple discrete features.
MixModel
Overview
MixModel is a multi-modal model that combines Swin Transformer (for image data) and FModel (for structured data) with a weighted fusion strategy for classification.

Parameters
embedding_size (list, default: [6, 113, 38, 115, 124, 120, 14]): Vocabulary sizes for structured features.
chioce_matrix (list, default: [1,1,1,1,0,1,0]): Feature selection mask.
pretrained_path (str, default: None): Path to pretrained weights for FModel.
num_classes (int, default: 3): Number of classes for classification.
ratio (float, default: 0.8): Weighting factor for combining image and structured features.
pretrain_path (str, default: "./swin_s-5e29d889.pth"): Path to pretrained weights for Swin Transformer.
chioce (str, default: "None"): Unused parameter (for future extension).
num_heads (int, default: 5): Number of heads in the multi-head attention mechanism.
Architecture
Swin Transformer: swin_s model for image feature extraction.
Feature Fusion Module: FModel for structured data processing.
Dropout: nn.Dropout(p=0.25) to prevent overfitting.
Linear Layer: nn.Linear(1000, num_classes) for classification.
Inputs and Outputs
Inputs:
img (Tensor): Image data, shape (batch_size, channels, height, width).
Structured data inputs (same as BasicModel).
Output:
x (Tensor): Classification logits, shape (batch_size, num_classes).
Forward Logic
Extract image features using Swin Transformer: x = self.swim(img), shape (batch_size, 1000).
Fuse structured features using FModel: e = self.embedding.embedding(...), shape (batch_size, 1000).
Perform weighted fusion: x = x * self.ratio + (1 - self.ratio) * e.
Apply dropout: x = self.dropout(x).
Classify using the linear layer: x = self.linear(x), shape (batch_size, num_classes).
Usage Scenario
Ideal for multi-modal tasks involving both image and structured data, such as disaster impact prediction.
Installation
Prerequisites
Python 3.6+
PyTorch 1.8+
torchvision
Install Dependencies
bash

Collapse

Wrap

Copy
pip install torch torchvision
Download Pretrained Weights
Download the Swin Transformer pretrained weights (swin_s-5e29d889.pth) and place them in the project directory.
If using pretrained weights for FModel, ensure the path is correctly specified.
Usage
Model Instantiation
python

Collapse

Wrap

Copy
import torch
from model import MixModel  # Replace with the actual file name

# Instantiate the model
model = MixModel(
    embedding_size=[6, 113, 38, 115, 124, 120, 14],
    chioce_matrix=[1, 1, 1, 1, 0, 1, 0],
    pretrained_path=None,
    num_classes=3,
    ratio=0.8,
    pretrain_path="./swin_s-5e29d889.pth",
    num_heads=5
)
Inference Example
python

Collapse

Wrap

Copy
# Construct input data
batch_size = 4
img = torch.randn(batch_size, 3, 224, 224)  # Image data
Evacuation_Zone = torch.randint(0, 6, (batch_size,))  # Evacuation zone
YearBuilt = torch.randint(0, 113, (batch_size,))  # Year of construction
EstimatedValue_level = torch.randint(0, 38, (batch_size,))  # Estimated value level
dist_track_line = torch.randint(0, 115, (batch_size,))  # Distance to track line
dist_track_landfall = torch.randint(0, 124, (batch_size,))  # Distance to landfall
wind_mean = torch.randint(0, 120, (batch_size,))  # Mean wind speed
flood_mean = torch.randint(0, 14, (batch_size,))  # Mean flood intensity

# Forward pass
model.eval()
with torch.no_grad():
    output = model(
        img,
        Evacuation_Zone,
        YearBuilt,
        EstimatedValue_level,
        dist_track_line,
        dist_track_landfall,
        wind_mean,
        flood_mean
    )
print(output.shape)  # Output shape: (batch_size, num_classes)
Training Example
python

Collapse

Wrap

Copy
import torch.optim as optim
from model import FocalLoss  # Replace with the actual file name

# Define loss function and optimizer
criterion = FocalLoss(alpha=1, gamma=2, logits=True)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Construct labels
labels = torch.randint(0, 3, (batch_size,))

# Training step
model.train()
optimizer.zero_grad()
output = model(
    img,
    Evacuation_Zone,
    YearBuilt,
    EstimatedValue_level,
    dist_track_line,
    dist_track_landfall,
    wind_mean,
    flood_mean
)
loss = criterion(output, labels)
loss.backward()
optimizer.step()
print(f"Loss: {loss.item()}")
Input Data Requirements
Image Data
Shape: (batch_size, 3, 224, 224) (RGB images, 224x224 resolution).
Preprocessing:
Resize images to 224x224.
Normalize pixel values (e.g., using ImageNet mean and standard deviation).
Structured Data
Feature Ranges:
Evacuation_Zone: Integer in [0, 5].
YearBuilt: Integer in [0, 112].
EstimatedValue_level: Integer in [0, 37].
dist_track_line: Integer in [0, 114].
dist_track_landfall: Integer in [0, 123].
wind_mean: Integer in [0, 119].
flood_mean: Integer in [0, 13].
Shape: Each feature should be a tensor of shape (batch_size,).
Notes and Considerations
Pretrained Weights:
Ensure the pretrain_path and pretrained_path point to valid files.
The Swin Transformer weights must match the swin_s architecture.
Class Imbalance:
Adjust alpha and gamma in FocalLoss to handle class imbalance.
Computational Resources:
The Swin Transformer and multi-head attention require significant GPU memory. Use a GPU for training and inference.
Feature Selection:
The chioce_matrix allows selective inclusion of features. Modify it based on your use case.
