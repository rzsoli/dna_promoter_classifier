# DNA Promoter Detection using Deep Learning

This project implements multiple deep learning models for detecting DNA promoter sequences in E. coli using the UCI Molecular Biology Promoter Gene Sequences dataset.

## Overview

Promoters are DNA sequences that signal where transcription of a gene begins. This project builds and evaluates various neural network architectures to classify DNA sequences as promoters (+) or non-promoters (-) using 57-nucleotide sequences.

## Dataset

- **Source**: UCI Molecular Biology Promoter Gene Sequences dataset (`promoters.data`)
- **Size**: 106 sequences (53 promoters, 53 non-promoters)  
- **Sequence Length**: 57 nucleotides each
- **Classes**: Binary (+ for promoter, - for non-promoter)

## Data Preprocessing

The raw dataset required several preprocessing steps to prepare it for deep learning models:

1. **Loading Data**: The dataset is loaded from a CSV file with columns: Class, Instance_Name, and Sequence
   
2. **Class Label Conversion**: 
   - Original 'Class' column contains '+' (promoter) and '-' (non-promoter)
   - Converted to binary labels: '+' → 1 (promoter), '-' → 0 (non-promoter)
   - Created new 'Label' column with numeric values

3. **Sequence Cleaning**:
   - Converted all sequences to uppercase
   - Removed leading/trailing whitespace using `.strip()`
   - Ensures consistent formatting across all sequences

4. **Feature Engineering**:
   - Calculated GC content: (count of G + count of C) / sequence length
   - Useful biological feature as promoters often have distinct GC patterns

5. **One-Hot Encoding**:
   - Each nucleotide mapped to a 4D binary vector:
     - A → [1,0,0,0]
     - C → [0,1,0,0]  
     - G → [0,0,1,0]
     - T → [0,0,0,1]
   - Final input shape: (num_sequences, 57, 4)

6. **Train/Validation/Test Split**:
   - Initial split: 70% train, 30% temp (stratified)
   - Temp split: 50% validation, 50% test (stratified)
   - Final: 74 train, 16 validation, 16 test sequences

## Exploratory Data Analysis (EDA)

Key insights from the data exploration:

1. **Class Distribution**: Perfectly balanced dataset with 53 promoters and 53 non-promoters (50% each)

2. **Sequence Length**: All sequences are exactly 57 nucleotides long (uniform length)

3. **Nucleotide Frequencies**:
   - Overall nucleotide distribution across all sequences
   - Position-specific nucleotide frequency heatmap reveals conserved regions
   - Clear patterns around positions -35 and -10 (classic promoter consensus sequences)

4. **GC Content Analysis**:
   - GC content ranges from 26.3% to 64.9%
   - Mean GC content: 45.6% (±7.96%)
   - Distribution shows variation between promoter and non-promoter sequences

5. **Positional Analysis**: 
   - Heatmap visualization shows nucleotide preferences at each position
   - Strong conservation patterns at biologically relevant positions
   - Positions are labeled from -50 to +6 following promoter notation conventions

## Models Implemented

### 1. Baseline CNN
- 1D Convolutional Neural Network with global max pooling
- Hyperparameter tuning using Keras Tuner
- Test Accuracy: ~87.5%
- **Note**: Trained on original dataset (106 sequences)

### Data Augmentation & Re-splitting

After evaluating the baseline model, data augmentation was applied to improve model performance:

1. **Reverse Complement Augmentation**:
   - Generated reverse complement for each sequence
   - Biologically meaningful as DNA can be read in both directions
   - Doubled dataset size to 212 sequences

2. **Random Mutation Augmentation**:
   - Applied random mutations to 3 nucleotides per sequence
   - Simulates natural variation and improves robustness
   - Further doubled dataset to 424 sequences

3. **New Data Split**:
   - Re-split augmented data: 70% train, 15% validation, 15% test
   - Final split: 296 train, 64 validation, 64 test sequences
   - Maintained stratification to ensure balanced class distribution

**All subsequent models (CNN + BiLSTM, Inception CNN, Transformer) were trained on this augmented dataset.**

### 2. CNN + BiLSTM with Attention
- Combines CNN feature extraction with bidirectional LSTM
- Attention mechanism for focusing on important regions
- Test Accuracy: ~95.3%

### 3. Inception-style CNN
- Multi-kernel convolutions (3, 5, 7) for capturing patterns at different scales
- Batch normalization for stable training
- Test Accuracy: ~98.4%

### 4. Transformer Encoder
- Self-attention mechanism with positional embeddings
- 2 transformer encoder blocks
- 10-fold CV Accuracy: 93.2% ± 4.2%
- Test Accuracy: ~95.3%

## Key Features

- **One-hot Encoding**: Each nucleotide (A, C, G, T) is encoded as a 4-dimensional binary vector

- **Hyperparameter Tuning**: Keras Tuner with Random Search for optimal architecture selection

- **Interpretability**: Integrated Gradients for visualizing important sequence positions

- **Cross-validation**: 10-fold stratified CV for robust evaluation (Transformer model)

## Training Visualization Analysis

Each model's training process is visualized with accuracy and loss plots showing both training and validation performance over epochs:

### Key Observations from Training Curves:

1. **Baseline CNN**: Shows initial overfitting with training accuracy reaching ~95% while validation plateaus around 87.5%

2. **CNN + BiLSTM**: Rapid convergence with both curves closely aligned, reaching near-perfect training accuracy (100%) with validation at ~95%

3. **Inception CNN**: Excellent generalization with minimal gap between training and validation curves, achieving 98.4% test accuracy

4. **Transformer**: More volatile training initially but stabilizes well, showing good convergence patterns

The plots help identify:
- **Overfitting**: When training accuracy >> validation accuracy
- **Convergence**: When both curves plateau
- **Optimal stopping point**: Where validation performance peaks before degrading

## Performance Summary

| Model | Test Accuracy | AUROC | AUPRC | F1-Score |
|-------|--------------|-------|-------|----------|
| Baseline CNN | 0.875 | 0.969 | 0.969 | 0.872 |
| CNN + BiLSTM | 0.953 | 0.988 | 0.988 | 0.954 |
| Inception CNN | 0.984 | 0.998 | 0.998 | 0.984 |
| Transformer | 0.953 | 0.988 | 0.988 | 0.954 |

## Requirements

```bash
tensorflow>=2.0
numpy
pandas
scikit-learn
matplotlib
seaborn
keras-tuner
```

## Usage

1. Ensure the dataset file `data/promoters.data` is in the correct path
2. Run the Jupyter notebook `promoter_detection copy.ipynb`
3. The notebook will:
   - Load and preprocess the data
   - Train multiple models with hyperparameter tuning
   - Evaluate performance on test set
   - Generate visualizations and interpretability plots
   - Save trained model weights

## Saved Model Weights

All trained models are saved for future use:

- **Baseline CNN**: Model weights saved during training (best validation AUPRC)
- **CNN + BiLSTM**: `models/cnn_bilstm_best.weights.h5` - Best model based on validation AUPRC
- **Inception CNN**: `models/inception_best.weights.h5` - Achieved highest test accuracy (98.4%)
- **Transformer**: `models/simple_transformer_best.weights.h5` - Strong performance with attention mechanism

These weights can be loaded to make predictions on new sequences without retraining.

## Key Findings

- The Inception-style CNN achieved the best performance with 98.4% test accuracy
- Data augmentation significantly improved model generalization
- Integrated Gradients revealed that models focus on regions around the -35 and -10 consensus sequences, which are biologically relevant for promoter recognition
- All deep learning models substantially outperformed traditional machine learning approaches

