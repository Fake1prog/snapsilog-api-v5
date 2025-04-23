import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split


class FoodComponentDataset(Dataset):
    """Dataset for training food component weight prediction model."""

    def __init__(self, dataset_index, image_dir, processed_dir, component_type=None):
        """
        Args:
            dataset_index: Path to dataset index JSON file or loaded dataset
            image_dir: Directory containing original food images
            processed_dir: Directory containing processed masks
            component_type: If specified, only load this component type (e.g., 'egg')
        """
        if isinstance(dataset_index, str):
            # Load dataset index from file
            with open(dataset_index, 'r') as f:
                self.dataset = json.load(f)['samples']
        else:
            # Use provided dataset
            self.dataset = dataset_index

        self.image_dir = image_dir
        self.processed_dir = processed_dir
        self.component_type = component_type

        # Define separate transforms for images and masks
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Create samples list
        self.samples = []
        for item in self.dataset:
            image_id = item['image_id']
            image_path = os.path.join(image_dir, item['image_path'])

            if not os.path.exists(image_path):
                # Try to find the image with different extensions
                for ext in ['.jpg', '.jpeg', '.png']:
                    alt_path = os.path.join(image_dir, f"{image_id}{ext}")
                    if os.path.exists(alt_path):
                        image_path = alt_path
                        break

            if not os.path.exists(image_path):
                print(f"Warning: Image not found for {image_id}")
                continue

            # Handle either all components or just the specified one
            if component_type:
                if component_type in item['components']:
                    comp_data = item['components'][component_type]
                    mask_path = os.path.join(processed_dir, comp_data['mask_path'])

                    if os.path.exists(mask_path):
                        self.samples.append({
                            'image_id': image_id,
                            'image_path': image_path,
                            'mask_path': mask_path,
                            'component': component_type,
                            'weight': comp_data['weight_grams'],
                            'features': {
                                'area': comp_data.get('area_pixels', 0),
                                'width': comp_data.get('width_pixels', 0),
                                'height': comp_data.get('height_pixels', 0),
                                'aspect_ratio': comp_data.get('aspect_ratio', 1.0),
                                'confidence': comp_data.get('confidence', 1.0)
                            }
                        })
            else:
                # Add all components
                for comp_name, comp_data in item['components'].items():
                    mask_path = os.path.join(processed_dir, comp_data['mask_path'])

                    if os.path.exists(mask_path):
                        self.samples.append({
                            'image_id': image_id,
                            'image_path': image_path,
                            'mask_path': mask_path,
                            'component': comp_name,
                            'weight': comp_data['weight_grams'],
                            'features': {
                                'area': comp_data.get('area_pixels', 0),
                                'width': comp_data.get('width_pixels', 0),
                                'height': comp_data.get('height_pixels', 0),
                                'aspect_ratio': comp_data.get('aspect_ratio', 1.0),
                                'confidence': comp_data.get('confidence', 1.0)
                            }
                        })

        print(f"Loaded {len(self.samples)} samples for {'all components' if not component_type else component_type}")

        # Compute component type embeddings
        self.component_types = sorted(list(set(sample['component'] for sample in self.samples)))
        self.component_to_idx = {comp: idx for idx, comp in enumerate(self.component_types)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')

        # Load mask
        mask = Image.open(sample['mask_path']).convert('L')

        # Apply transformations - use separate transforms for image and mask
        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        # Create one-hot encoding for component type
        component_idx = self.component_to_idx[sample['component']]
        component_onehot = torch.zeros(len(self.component_types))
        component_onehot[component_idx] = 1.0

        # Extract manual features
        manual_features = torch.tensor([
            sample['features']['area'],
            sample['features']['width'],
            sample['features']['height'],
            sample['features']['aspect_ratio'],
            sample['features']['confidence']
        ], dtype=torch.float32)

        # Normalize manual features (simple scaling for now)
        if manual_features[0] > 0:  # Skip if area is zero
            # Normalize area by typical mask area
            manual_features[0] = manual_features[0] / 50000  # Typical mask might be around 50k pixels

            # Normalize width and height by typical dimensions
            manual_features[1] = manual_features[1] / 500  # Width normalization
            manual_features[2] = manual_features[2] / 500  # Height normalization

        return {
            'image': image,
            'mask': mask,
            'component_type': component_onehot,
            'manual_features': manual_features,
            'weight': torch.tensor(sample['weight'], dtype=torch.float32)
        }


class FoodWeightCNN(nn.Module):
    """CNN model for predicting food component weights."""

    def __init__(self, num_components, use_manual_features=True):
        """
        Args:
            num_components: Number of different food component types
            use_manual_features: Whether to use hand-crafted features
        """
        super(FoodWeightCNN, self).__init__()

        # Image feature extraction
        self.image_features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )

        # Mask feature extraction
        self.mask_features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )

        # Global average pooling to reduce feature maps
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Component type embedding
        self.component_embedding = nn.Sequential(
            nn.Linear(num_components, 16),
            nn.ReLU()
        )

        # Number of features after pooling
        img_features = 64
        mask_features = 32
        comp_features = 16
        manual_features = 5 if use_manual_features else 0

        total_features = img_features + mask_features + comp_features + manual_features

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(total_features, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

        self.use_manual_features = use_manual_features

    def forward(self, image, mask, component_type, manual_features=None):
        # Extract image features
        img_feats = self.image_features(image)
        img_feats = self.gap(img_feats).squeeze(-1).squeeze(-1)

        # Extract mask features
        mask_feats = self.mask_features(mask)
        mask_feats = self.gap(mask_feats).squeeze(-1).squeeze(-1)

        # Component type embedding
        comp_feats = self.component_embedding(component_type)

        # Concatenate features
        if self.use_manual_features and manual_features is not None:
            combined = torch.cat([img_feats, mask_feats, comp_feats, manual_features], dim=1)
        else:
            combined = torch.cat([img_feats, mask_feats, comp_feats], dim=1)

        # Predict weight
        weight = self.regressor(combined)
        return weight.squeeze()


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc="Training"):
        # Get data
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        component_types = batch['component_type'].to(device)
        manual_features = batch['manual_features'].to(device)
        weights = batch['weight'].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images, masks, component_types, manual_features)
        loss = criterion(outputs, weights)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(train_loader.dataset)


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            # Get data
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            component_types = batch['component_type'].to(device)
            manual_features = batch['manual_features'].to(device)
            weights = batch['weight'].to(device)

            # Forward pass
            outputs = model(images, masks, component_types, manual_features)
            loss = criterion(outputs, weights)

            total_loss += loss.item() * images.size(0)

            # Store predictions and ground truths for metrics
            predictions.extend(outputs.cpu().numpy())
            ground_truths.extend(weights.cpu().numpy())

    # Calculate metrics
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)

    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - ground_truths))

    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((predictions - ground_truths) / ground_truths)) * 100

    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((predictions - ground_truths) ** 2))

    return total_loss / len(val_loader.dataset), mae, mape, rmse, predictions, ground_truths


def plot_training_results(history, output_dir):
    """Plot training and validation loss."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_mae'], label='MAE')
    plt.plot(history['val_rmse'], label='RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('Error (grams)')
    plt.title('Validation Metrics')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

    # Plot prediction vs. ground truth for last epoch
    plt.figure(figsize=(10, 8))
    plt.scatter(history['ground_truths'][-1], history['predictions'][-1], alpha=0.5)

    # Add perfect prediction line
    min_val = min(min(history['ground_truths'][-1]), min(history['predictions'][-1]))
    max_val = max(max(history['ground_truths'][-1]), max(history['predictions'][-1]))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.xlabel('Ground Truth (grams)')
    plt.ylabel('Prediction (grams)')
    plt.title(f'Prediction vs. Ground Truth (MAE: {history["val_mae"][-1]:.2f}g, MAPE: {history["val_mape"][-1]:.2f}%)')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_vs_truth.png'))
    plt.close()

    # Plot component-specific results if available
    if 'component_names' in history and 'component_predictions' in history:
        component_names = history['component_names']
        component_predictions = history['component_predictions'][-1]
        component_ground_truths = history['component_ground_truths'][-1]

        # Calculate how many subplot rows and columns we need
        n_components = len(component_names)
        n_cols = min(3, n_components)
        n_rows = (n_components + n_cols - 1) // n_cols  # Ceiling division

        plt.figure(figsize=(15, 5 * n_rows))
        for i, component in enumerate(component_names):
            plt.subplot(n_rows, n_cols, i + 1)
            pred = component_predictions[component]
            truth = component_ground_truths[component]

            if len(pred) > 0:
                plt.scatter(truth, pred, alpha=0.5)

                # Add perfect prediction line
                min_val = min(min(truth), min(pred))
                max_val = max(max(truth), max(pred))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--')

                # Add component specific metrics
                mae = np.mean(np.abs(np.array(pred) - np.array(truth)))
                mape = np.mean(np.abs((np.array(pred) - np.array(truth)) / np.array(truth))) * 100

                plt.title(f'{component.capitalize()} (MAE: {mae:.2f}g, MAPE: {mape:.2f}%)')
                plt.xlabel('Ground Truth (g)')
                plt.ylabel('Prediction (g)')
                plt.axis('equal')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'prediction_by_component.png'))
        plt.close()


def export_model(model, num_components, output_dir):
    """Export model to TorchScript for mobile deployment."""
    model.eval()

    # Create example inputs
    example_image = torch.randn(1, 3, 224, 224)
    example_mask = torch.randn(1, 1, 224, 224)
    example_component = torch.zeros(1, num_components)
    example_component[0, 0] = 1.0  # One-hot for first component
    example_manual = torch.randn(1, 5)

    # Trace model
    traced_model = torch.jit.trace(
        model,
        (example_image, example_mask, example_component, example_manual)
    )

    # Save model
    model_path = os.path.join(output_dir, 'weight_model_mobile.pt')
    traced_model.save(model_path)
    print(f"Exported TorchScript model to {model_path}")

    # Also save component names for reference
    component_info = {
        'component_names': model.component_types if hasattr(model, 'component_types') else [],
        'input_shape': {
            'image': [1, 3, 224, 224],
            'mask': [1, 1, 224, 224],
            'component_type': [1, num_components],
            'manual_features': [1, 5]
        }
    }

    with open(os.path.join(output_dir, 'model_info.json'), 'w') as f:
        json.dump(component_info, f, indent=2)


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_numpy_types(item) for item in obj.tolist()]
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    else:
        return obj


def generate_history_and_plots(model, dataset, val_dataset, output_dir, device):
    """Generate training history and plots without training."""
    print("Generating history and plots for existing model...")

    # Set up criterion for validation
    criterion = nn.MSELoss()

    # Create validation dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4
    )

    # Validate model
    val_loss, val_mae, val_mape, val_rmse, predictions, ground_truths = validate(
        model, val_loader, criterion, device
    )

    # Print metrics
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val MAE: {val_mae:.2f}g, Val MAPE: {val_mape:.2f}%, Val RMSE: {val_rmse:.2f}g")

    # Create history dictionary with single entry
    history = {
        'train_loss': [0.0],  # Placeholder since we don't have training data
        'val_loss': [float(val_loss)],
        'val_mae': [float(val_mae)],
        'val_mape': [float(val_mape)],
        'val_rmse': [float(val_rmse)],
        'predictions': [[float(p) for p in predictions.tolist()]],
        'ground_truths': [[float(g) for g in ground_truths.tolist()]],
        'component_names': dataset.component_types
    }

    # Get per-component predictions
    component_predictions = {comp: [] for comp in dataset.component_types}
    component_ground_truths = {comp: [] for comp in dataset.component_types}

    model.eval()
    with torch.no_grad():
        for i in range(len(val_dataset)):
            sample = val_dataset[i]
            img = sample['image'].unsqueeze(0).to(device)
            mask = sample['mask'].unsqueeze(0).to(device)
            comp_type = sample['component_type'].unsqueeze(0).to(device)
            features = sample['manual_features'].unsqueeze(0).to(device)
            weight = sample['weight'].item()

            # Get component name from one-hot
            comp_idx = torch.argmax(sample['component_type']).item()
            comp_name = dataset.component_types[comp_idx]

            # Forward pass
            output = model(img, mask, comp_type, features).item()

            # Store by component
            component_predictions[comp_name].append(float(output))
            component_ground_truths[comp_name].append(float(weight))

    history['component_predictions'] = [component_predictions]
    history['component_ground_truths'] = [component_ground_truths]

    # Save history
    serializable_history = convert_numpy_types(history)
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(serializable_history, f, indent=2)

    # Plot results
    plot_training_results(history, output_dir)

    return history


def main():
    parser = argparse.ArgumentParser(description='Train food component weight prediction model')
    parser.add_argument('--dataset_index', type=str, default='processed_dataset/dataset_index.json',
                        help='Path to dataset index JSON file')
    parser.add_argument('--image_dir', type=str, default='.',
                        help='Directory containing original images')
    parser.add_argument('--processed_dir', type=str, default='processed_dataset',
                        help='Directory containing processed masks')
    parser.add_argument('--output_dir', type=str, default='model_output',
                        help='Directory to save model and results')
    parser.add_argument('--component_type', type=str, default=None,
                        help='Train for specific component type (default: all components)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use_manual_features', action='store_true',
                        help='Use manual features (area, width, etc.)')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training if model exists and just generate plots')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    dataset = FoodComponentDataset(
        args.dataset_index,
        args.image_dir,
        args.processed_dir,
        component_type=args.component_type
    )

    # Split into train/validation sets
    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        random_state=args.seed
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # Initialize model
    num_components = len(dataset.component_types)
    model = FoodWeightCNN(
        num_components=num_components,
        use_manual_features=args.use_manual_features
    )

    # Save component types to model for reference
    model.component_types = dataset.component_types
    model.component_to_idx = dataset.component_to_idx

    # Check if we should skip training
    model_path = os.path.join(args.output_dir, 'best_weight_model.pth')
    if os.path.exists(model_path) and args.skip_training:
        print(f"Found existing model at {model_path}, skipping training.")

        # Load the existing model
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        # Generate history and plots
        generate_history_and_plots(model, dataset, val_dataset, args.output_dir, device)

        print("Analysis completed!")
        return

    # Continue with training if not skipping
    model.to(device)

    # Print model summary
    print(f"Model architecture:")
    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Create data loaders for training
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_mape': [],
        'val_rmse': [],
        'predictions': [],
        'ground_truths': [],
        'component_predictions': [],
        'component_ground_truths': [],
        'component_names': dataset.component_types
    }

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_mae, val_mape, val_rmse, predictions, ground_truths = validate(
            model, val_loader, criterion, device
        )

        # Update scheduler
        scheduler.step(val_loss)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Val MAE: {val_mae:.2f}g, Val MAPE: {val_mape:.2f}%, Val RMSE: {val_rmse:.2f}g")

        # Save history - convert numpy values to Python native types
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['val_mae'].append(float(val_mae))
        history['val_mape'].append(float(val_mape))
        history['val_rmse'].append(float(val_rmse))

        # Convert numpy arrays to native Python lists with native Python floats
        history['predictions'].append([float(p) for p in predictions.tolist()])
        history['ground_truths'].append([float(g) for g in ground_truths.tolist()])

        # Get per-component predictions
        component_predictions = {comp: [] for comp in dataset.component_types}
        component_ground_truths = {comp: [] for comp in dataset.component_types}

        model.eval()
        with torch.no_grad():
            for i in range(len(val_dataset)):
                sample = val_dataset[i]
                img = sample['image'].unsqueeze(0).to(device)
                mask = sample['mask'].unsqueeze(0).to(device)
                comp_type = sample['component_type'].unsqueeze(0).to(device)
                features = sample['manual_features'].unsqueeze(0).to(device)
                weight = sample['weight'].item()

                # Get component name from one-hot
                comp_idx = torch.argmax(sample['component_type']).item()
                comp_name = dataset.component_types[comp_idx]

                # Forward pass
                output = model(img, mask, comp_type, features).item()

                # Store by component - ensure these are native Python floats
                component_predictions[comp_name].append(float(output))
                component_ground_truths[comp_name].append(float(weight))

        history['component_predictions'].append(component_predictions)
        history['component_ground_truths'].append(component_ground_truths)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_weight_model.pth'))
            print(f"Saved best model with Val Loss: {val_loss:.4f}")

            # Export model for mobile
            export_model(model, num_components, args.output_dir)

    # Convert any remaining numpy types before saving
    serializable_history = convert_numpy_types(history)

    # Save training history
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(serializable_history, f)

    # Plot training results
    plot_training_results(history, args.output_dir)

    print("Training completed!")


if __name__ == "__main__":
    main()