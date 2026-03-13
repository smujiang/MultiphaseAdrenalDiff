# Fix for OpenMP library initialization error on macOS
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math


class FeatureTransformerClassifier(nn.Module):
    """
    Transformer-based classification model for tabular data with explicit feature masking.
    
    Takes two inputs:
    - features: (batch_size, num_features) - raw feature values
    - feature_mask: (batch_size, num_features) - boolean mask indicating available features
    
    The model treats each feature as a token and uses attention to learn feature interactions.
    """
    
    def __init__(self,
                 num_features: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 num_classes: int = 2,
                 pooling: str = 'cls',  # 'cls', 'mean', 'max', 'attention'
                 use_positional_encoding: bool = True):
        """
        Args:
            num_features: Number of input features
            d_model: Transformer embedding dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
            num_classes: Number of output classes
            pooling: Pooling strategy ('cls', 'mean', 'max', 'attention')
            use_positional_encoding: Whether to use positional encoding for features
        """
        super().__init__()
        
        self.num_features = num_features
        self.d_model = d_model
        self.pooling = pooling
        self.num_classes = num_classes
        self.use_positional_encoding = use_positional_encoding
        
        # Feature embedding: project each feature to d_model dimensions
        self.feature_embedding = nn.Linear(1, d_model)  # Each feature is a scalar
        
        # Positional encoding for feature positions
        if self.use_positional_encoding:
            self.positional_encoding = nn.Parameter(
                torch.randn(1, num_features, d_model) * 0.02
            )
        
        # CLS token for classification (if using cls pooling)
        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention pooling head (if using attention pooling)
        if pooling == 'attention':
            self.attention_pool = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=1,
                dropout=dropout,
                batch_first=True
            )
            self.attention_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, features: torch.Tensor, feature_mask: torch.Tensor):
        """
        Forward pass
        
        Args:
            features: (batch_size, num_features) - feature values
            feature_mask: (batch_size, num_features) - boolean mask (True = available)
            
        Returns:
            logits: (batch_size, num_classes) - classification logits
            attention_weights: Optional attention weights for interpretability
        """
        batch_size = features.shape[0]
        device = features.device
        
        # Handle missing features by replacing masked positions with 0
        features_masked = features * feature_mask.float()
        
        # Embed each feature: (batch_size, num_features, d_model)
        features_embedded = self.feature_embedding(features_masked.unsqueeze(-1))
        
        # Add positional encoding
        if self.use_positional_encoding:
            features_embedded = features_embedded + self.positional_encoding
        
        # Prepare sequence for transformer
        if self.pooling == 'cls':
            # Prepend CLS token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            sequence = torch.cat([cls_tokens, features_embedded], dim=1)
            
            # Extend mask for CLS token (CLS is always present)
            cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
            extended_mask = torch.cat([cls_mask, feature_mask], dim=1)
        else:
            sequence = features_embedded
            extended_mask = feature_mask
        
        # Create attention mask for transformer 
        # transformer expects src_key_padding_mask where True = masked (ignored)
        src_key_padding_mask = ~extended_mask  # Invert mask
        
        # Apply transformer
        encoded = self.transformer(sequence, src_key_padding_mask=src_key_padding_mask)
        
        # Pooling
        attention_weights = None
        if self.pooling == 'cls':
            pooled = encoded[:, 0, :]  # Use CLS token
        elif self.pooling == 'mean':
            # Masked average over available features
            if self.pooling == 'cls':
                feature_encoded = encoded[:, 1:, :]  # Skip CLS token
                valid_mask = feature_mask
            else:
                feature_encoded = encoded
                valid_mask = feature_mask
                
            mask_expanded = valid_mask.unsqueeze(-1).float()  # (batch_size, num_features, 1)
            masked_features = feature_encoded * mask_expanded
            sum_features = masked_features.sum(dim=1)  # (batch_size, d_model)
            count_features = mask_expanded.sum(dim=1).clamp_min(1)  # (batch_size, 1)
            pooled = sum_features / count_features
            
        elif self.pooling == 'max':
            # Max pooling over available features
            if self.pooling == 'cls':
                feature_encoded = encoded[:, 1:, :]
                valid_mask = feature_mask
            else:
                feature_encoded = encoded
                valid_mask = feature_mask
                
            # Set masked positions to large negative values for max pooling
            mask_expanded = valid_mask.unsqueeze(-1).float()
            masked_features = feature_encoded * mask_expanded + (1 - mask_expanded) * (-1e9)
            pooled, _ = masked_features.max(dim=1)  # (batch_size, d_model)
            
        elif self.pooling == 'attention':
            # Attention-based pooling
            query = self.attention_query.expand(batch_size, -1, -1)
            
            if self.pooling == 'cls':
                feature_encoded = encoded[:, 1:, :]
                key_padding_mask = ~feature_mask
            else:
                feature_encoded = encoded
                key_padding_mask = ~extended_mask
                
            attended, attention_weights = self.attention_pool(
                query, feature_encoded, feature_encoded,
                key_padding_mask=key_padding_mask
            )
            pooled = attended.squeeze(1)  # (batch_size, d_model)
        
        # Apply layer norm
        pooled = self.layer_norm(pooled)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits, attention_weights


class TabularDataset(Dataset):
    """Dataset for tabular data with feature masking"""
    
    def __init__(self, features, labels, feature_mask=None):
        """
        Args:
            features: (num_samples, num_features) - feature matrix
            labels: (num_samples,) - target labels  
            feature_mask: (num_samples, num_features) - boolean mask (optional)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
        if feature_mask is None:
            # If no mask provided, assume all features are available
            self.feature_mask = torch.ones_like(self.features, dtype=torch.bool)
        else:
            self.feature_mask = torch.BoolTensor(feature_mask)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.feature_mask[idx], self.labels[idx]


class TransformerTrainer:
    """Trainer for the transformer classifier"""
    
    def __init__(self, model, device='cpu', learning_rate=1e-3, weight_decay=1e-5):
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        # Use CrossEntropyLoss for all classification tasks (binary and multi-class)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for features, feature_mask, labels in dataloader:
            features = features.to(self.device)
            feature_mask = feature_mask.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            logits, _ = self.model(features, feature_mask)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def evaluate(self, dataloader):
        """Evaluate model with comprehensive metrics"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            for features, feature_mask, labels in dataloader:
                features = features.to(self.device)
                feature_mask = feature_mask.to(self.device)
                labels = labels.to(self.device)
                
                logits, _ = self.model(features, feature_mask)
                loss = self.criterion(logits, labels)
                
                if self.model.num_classes == 2:
                    # For binary classification with CrossEntropyLoss, use softmax probabilities
                    probs = torch.softmax(logits, dim=1)
                    preds = probs[:, 1].cpu().numpy()  # Take probability of positive class
                    labels_np = labels.cpu().numpy()
                else:
                    preds = torch.softmax(logits, dim=1).cpu().numpy()
                    labels_np = labels.cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels_np)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        if self.model.num_classes == 2:
            # Binary classification - comprehensive metrics
            pred_classes = (all_preds > 0.5).astype(int)
            
            # Basic metrics
            auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0
            acc = accuracy_score(all_labels, pred_classes)
            
            # Advanced metrics
            try:
                precision = precision_score(all_labels, pred_classes, zero_division=0)
                recall = recall_score(all_labels, pred_classes, zero_division=0)  # Sensitivity
                f1 = f1_score(all_labels, pred_classes, zero_division=0)
                
                # Calculate specificity from confusion matrix
                tn, fp, fn, tp = confusion_matrix(all_labels, pred_classes).ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                
            except Exception as e:
                print(f"Warning: Error calculating some metrics: {e}")
                precision = recall = f1 = specificity = 0.0
            
            return {
                'loss': avg_loss,
                'auc': auc,
                'accuracy': acc,
                'precision': precision,
                'recall': recall,  # Sensitivity
                'sensitivity': recall,  # Alias for recall
                'specificity': specificity,
                'f1': f1
            }
        else:
            # Multi-class classification
            pred_classes = np.argmax(all_preds, axis=1)
            acc = accuracy_score(all_labels, pred_classes)
            
            # Multi-class metrics (macro average)
            precision = precision_score(all_labels, pred_classes, average='macro', zero_division=0)
            recall = recall_score(all_labels, pred_classes, average='macro', zero_division=0)
            f1 = f1_score(all_labels, pred_classes, average='macro', zero_division=0)
            
            return {
                'loss': avg_loss,
                'accuracy': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
    
    def fit(self, train_loader, val_loader=None, epochs=100, patience=10):
        """Train the model"""
        best_val_metric = -np.inf
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_metric': []}
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                val_loss = val_metrics['loss']
                val_metric = val_metrics.get('auc', val_metrics.get('accuracy', 0))
                
                history['val_loss'].append(val_loss)
                history['val_metric'].append(val_metric)
                
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val AUC: {val_metrics.get('auc', 0):.4f}, "
                      f"Val Acc: {val_metrics.get('accuracy', 0):.4f}, "
                      f"Val F1: {val_metrics.get('f1', 0):.4f}")
                
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
        
        return history




def parse_column_names(columns_str):
    """Parse the column names from the string format"""
    # Remove backslashes and extra whitespace, then split by comma
    clean_str = columns_str[0].replace('\\', '').replace('\n', '').replace('\t', ' ')
    columns = [col.strip() for col in clean_str.split(',')]
    return columns


def load_csv_data(csv_path, columns_def, target_col='malignancy', normalize_features=True):
    """
    Load CSV data and prepare features and masks for transformer model
    
    Args:
        csv_path: Path to CSV file
        columns_def: Column definition (if None, will use all columns)
        target_col: Name of target column
        normalize_features: Whether to normalize feature values
        
    Returns:
        features: numpy array of shape (n_samples, n_features)
        labels: numpy array of shape (n_samples,)
        feature_mask: numpy array of shape (n_samples, n_features) boolean mask
        feature_names: list of feature column names
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Parse column names if provided

    expected_columns = parse_column_names(columns_def)
    print(f"Expected columns: {expected_columns[:10]}...")  # Show first 10
    print(f"Found columns: {list(df.columns)[:10]}...")  # Show first 10

    # Find target column
    if target_col not in expected_columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV")
    
    # Get target column index
    target_idx = list(expected_columns).index(target_col)
    print(f"Target column '{target_col}' found at index {target_idx}")
    
    # Features are columns after the target column
    feature_columns = expected_columns[target_idx + 1:]
    print(f"Number of feature columns: {len(feature_columns)}")
    print(f"First few feature columns: {feature_columns[:5]}")
    
    # Extract labels
    labels = df[target_col].values
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Extract features
    features_df = df[feature_columns]
    
    # Create feature mask (True where data is available, False where missing/NaN)
    feature_mask = ~features_df.isna().values  # True where NOT NaN
    print(f"Feature availability: {feature_mask.sum()}/{feature_mask.size} ({100*feature_mask.mean():.1f}%)")
    
    # Fill missing values with 0 for features
    features = features_df.fillna(0).values.astype(np.float32)
    
    # # Normalize features if requested
    # if normalize_features:
    #     # Only normalize non-zero values (available features)
    #     scaler = StandardScaler()
        
    #     # Create a copy for normalization
    #     features_normalized = features.copy()
        
    #     # For each feature column, normalize only the available values
    #     for i in range(features.shape[1]):
    #         available_mask = feature_mask[:, i]
    #         if available_mask.sum() > 1:  # Need at least 2 values to normalize
    #             available_values = features[available_mask, i].reshape(-1, 1)
    #             normalized_values = scaler.fit_transform(available_values).ravel()
    #             features_normalized[available_mask, i] = normalized_values
        
    #     features = features_normalized
    
    print(f"Final data shapes:")
    print(f"  Features: {features.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Feature mask: {feature_mask.shape}")
    
    return features, labels, feature_mask, feature_columns


def create_data_loaders(features, labels, feature_mask, test_size=0.2, val_size=0.2, 
                       batch_size=32, random_state=42):
    """
    Create train, validation, and test data loaders
    
    Args:
        features: Feature matrix
        labels: Target labels
        feature_mask: Feature availability mask
        test_size: Fraction for test set
        val_size: Fraction of remaining data for validation
        batch_size: Batch size for data loaders
        random_state: Random seed
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    mask_temp, mask_test, _, _ = train_test_split(
        feature_mask, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Second split: separate train and validation from remaining data
    if val_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )
        mask_train, mask_val, _, _ = train_test_split(
            mask_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )
        
        # Create datasets
        train_dataset = TabularDataset(X_train, y_train, mask_train)
        val_dataset = TabularDataset(X_val, y_val, mask_val)
        test_dataset = TabularDataset(X_test, y_test, mask_test)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    else:
        # No validation set
        train_dataset = TabularDataset(X_temp, y_temp, mask_temp)
        test_dataset = TabularDataset(X_test, y_test, mask_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, None, test_loader


# Example usage
if __name__ == "__main__":
    from config import DATA_ROOT_DIR
    output_dir = os.path.join(DATA_ROOT_DIR, "output")

    data_root = os.path.join(DATA_ROOT_DIR, "processed_data")
    # Collect test metrics across multiple runs
    test_metrics_list = {
        'auc': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'specificity': [],
        'f1': []
    }

    # Define a specific dataset and dataloader for testing the model
    Columns_with_washout = ["MRN, StudyDate, group, available_phases, malignancy, \
            area, perimeter,	eccentricity, axis_major_length, axis_minor_length,\
            absolute_washout, relative_washout, absolute_washout_rate, relative_washout_rate,\
            NC_contrast,	NC_correlation,	NC_energy,	NC_homogeneity,	avg_HU_NC,\
            AR_contrast,	AR_correlation,	AR_energy,	AR_homogeneity,	avg_HU_AR,\
            PV_contrast,	PV_correlation,	PV_energy,	PV_homogeneity,	avg_HU_PV,\
            Delay_contrast,	Delay_correlation,	Delay_energy,	Delay_homogeneity,	avg_HU_Delay"]

    # malignancy is the target label.

    Columns_without_washout = ["MRN, StudyDate, group, available_phases, malignancy, \
            area, perimeter,	eccentricity, axis_major_length, axis_minor_length,\
            NC_contrast,	NC_correlation,	NC_energy,	NC_homogeneity,	avg_HU_NC,\
            AR_contrast,	AR_correlation,	AR_energy,	AR_homogeneity,	avg_HU_AR,\
            PV_contrast,	PV_correlation,	PV_energy,	PV_homogeneity,	avg_HU_PV,\
            Delay_contrast,	Delay_correlation,	Delay_energy,	Delay_homogeneity,	avg_HU_Delay"]

    # with_washout = True  # Set to False to test without washout features
    with_washout = False  # Set to False to test without washout features
    seed_list = [42, 123, 456, 789, 1010]  # Example seeds for reproducibility
    for i in range(5): # Run multiple times to check stability
        print(f"\n=== Run {i+1} ===")
        # Set random seeds  
        seed = seed_list[i]
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Load real CSV data 
        csv_path = os.path.join(data_root, "grouped_instance_features_unified", "all_groups_features_unified_normalized.csv")

        print("Loading CSV data...")
        features, labels, feature_mask, feature_names = load_csv_data(
            csv_path, 
            columns_def=Columns_with_washout if with_washout else Columns_without_washout,
            target_col='malignancy',
            normalize_features=True
        )
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            features, labels, feature_mask, 
            test_size=0.2, val_size=0.2, batch_size=32
        )
        
        num_features = features.shape[1]
        print(f"Using real data with {num_features} features")
        

        # Create model
        model = FeatureTransformerClassifier(
            num_features=num_features,
            d_model=128,
            nhead=8,
            num_layers=3,
            dim_feedforward=512,
            dropout=0.1,
            num_classes=2,  # Binary classification for malignancy prediction
            pooling='attention',  # Can be 'cls', 'mean', 'max', 'attention'
            use_positional_encoding=True
        )
        
        # Train model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        trainer = TransformerTrainer(model, device=device, learning_rate=1e-4, weight_decay=1e-5)
        
        print("\nStarting transformer classifier training...")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train with validation if available
        history = trainer.fit(
            train_loader, 
            val_loader if 'val_loader' in locals() else test_loader, 
            epochs=50, 
            patience=5
        )
        
        # Final evaluation
        print("\nFinal evaluation:")
        
        
        test_metrics = trainer.evaluate(test_loader)
        print(f"\nTest Metrics:")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  AUC: {test_metrics['auc']:.4f}")
        print(f"  ACC: {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall(Sens): {test_metrics['recall']:.4f}")
        print(f"  Specificity: {test_metrics['specificity']:.4f}")
        print(f"  F1: {test_metrics['f1']:.4f}")
        

        test_metrics_list['auc'].append(test_metrics['auc'])
        test_metrics_list['accuracy'].append(test_metrics['accuracy'])
        test_metrics_list['precision'].append(test_metrics['precision'])
        test_metrics_list['recall'].append(test_metrics['recall'])
        test_metrics_list['specificity'].append(test_metrics['specificity'])
        test_metrics_list['f1'].append(test_metrics['f1'])
        
        # Store all test metrics for final summary
        test_results = {
            'auc': test_metrics['auc'],
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'specificity': test_metrics['specificity'],
            'f1': test_metrics['f1']
        }



        # Feature importance analysis (if using attention pooling)
        if model.pooling == 'attention':
            print("\nAnalyzing feature attention patterns...")
            model.eval()
            sample_batch = next(iter(test_loader))
            with torch.no_grad():
                features, feature_mask, _ = sample_batch
                features = features.to(device)
                feature_mask = feature_mask.to(device)
                _, attention_weights = model(features, feature_mask)
                if attention_weights is not None:
                    avg_attention = attention_weights.mean(dim=0).cpu().numpy()
                    print(f"Average attention weights shape: {avg_attention.shape}")
                    print(f"Top 5 most attended feature names: {[feature_names[i] for i in np.argsort(avg_attention.ravel())[-5:]]}")


    # Final comprehensive summary across all runs
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY (5 runs)")
    print("="*60)
    print(f"{'Metric':<15} {'Mean':<10} {'Std':<10} {'95% CI':<15}")
    print("-"*50)
    
    for metric_name, values in test_metrics_list.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        ci_lower = mean_val - 1.96 * std_val / np.sqrt(len(values))
        ci_upper = mean_val + 1.96 * std_val / np.sqrt(len(values))
        
        print(f"{metric_name.upper():<15} {mean_val:<10.4f} {std_val:<10.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    print("\n" + "="*60)
    # save final summary to CSV, columns:
    result_csv_columns = {"Strategy": [], "AUC": [], "ACC": [], "Precision": [], "Recall(Sens)": [], "Specificity": [], "F1": []}
    result_csv_columns["Strategy"].append("Transformer")
    result_csv_columns["AUC"].append(f"{np.mean(test_metrics_list['auc']):.4f}±{np.std(test_metrics_list['auc']):.4f}")
    result_csv_columns["ACC"].append(f"{np.mean(test_metrics_list['accuracy']):.4f}±{np.std(test_metrics_list['accuracy']):.4f}")
    result_csv_columns["Precision"].append(f"{np.mean(test_metrics_list['precision']):.4f}±{np.std(test_metrics_list['precision']):.4f}")
    result_csv_columns["Recall(Sens)"].append(f"{np.mean(test_metrics_list['recall']):.4f}±{np.std(test_metrics_list['recall']):.4f}")
    result_csv_columns["Specificity"].append(f"{np.mean(test_metrics_list['specificity']):.4f}±{np.std(test_metrics_list['specificity']):.4f}")
    result_csv_columns["F1"].append(f"{np.mean(test_metrics_list['f1']):.4f}±{np.std(test_metrics_list['f1']):.4f}")


    result_df = pd.DataFrame(result_csv_columns)
    if with_washout:
        save_fn = os.path.join(output_dir, "fusion_strategy_eval", "overall_transformer_results_summary.csv")
    else:
        save_fn = os.path.join(output_dir, "fusion_strategy_eval", "overall_transformer_results_summary_no_washout.csv")
    os.makedirs(os.path.dirname(save_fn), exist_ok=True)
    result_df.to_csv(save_fn, index=False)
    print(f"\nSaved final results summary to {save_fn}")
