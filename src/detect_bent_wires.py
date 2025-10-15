import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing, generate_binary_structure, sobel
from scipy import ndimage
from skimage.measure import regionprops, label

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
LATENT_DIM = 64

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]

class DeepAutoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(DeepAutoencoder, self).__init__()
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, latent_dim, 1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True)
        )
        
        # Decoder: 16 -> 32 -> 64 -> 128 -> 256
        self.dec_bottleneck = nn.Sequential(
            nn.Conv2d(latent_dim, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bottleneck = self.bottleneck(e4)
        d = self.dec_bottleneck(bottleneck)
        d4 = self.dec4(d)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        return d1

def train_model(dataset, dataloader):
    model = DeepAutoencoder(latent_dim=LATENT_DIM).to(device)
    model_path = f'bent_wire_detector_latent{LATENT_DIM}_epochs{EPOCHS}.pth'

    if os.path.exists(model_path):
        print(f"\n✓ Loading existing model: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n[Training] Latent={LATENT_DIM}, Epochs={EPOCHS}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for images, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Model saved: {model_path}")
    return model

def compute_anomalies(model, dataset):
    print("\nComputing anomaly scores...")
    model.eval()
    errors = []
    names = []

    with torch.no_grad():
        for images, filenames in tqdm(DataLoader(dataset, batch_size=1, shuffle=False)):
            images = images.to(device)
            outputs = model(images)
            mse = torch.mean((images - outputs) ** 2, dim=[1, 2, 3]).cpu().numpy()
            errors.extend(mse)
            names.extend(filenames)

    errors = np.array(errors)
    
    mean_error = errors.mean()
    std_error = errors.std()
    threshold = mean_error + 3 * std_error
    
    anomaly_mask = errors > threshold
    top_indices = np.where(anomaly_mask)[0]
    
    top_indices = top_indices[np.argsort(errors[top_indices])[::-1]]
    
    num_anomalies = len(top_indices)

    print("\n" + "="*60)
    print(f"ANOMALIES (>{threshold:.6f}, mean+2σ):")
    print(f"Found {num_anomalies} anomalies out of {len(errors)} images")
    print("="*60)
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. {names[idx]}: {errors[idx]:.6f}")

    return top_indices, names

def detect_bent_wires(error_map, original_image):
    threshold_95 = np.percentile(error_map, 95)
    error_map = np.where(error_map >= threshold_95, error_map, 0)
    
    fine = gaussian_filter(error_map, sigma=1.5)
    coarse = gaussian_filter(error_map, sigma=3.5)
    
    fine_mask = fine > np.percentile(fine[fine > 0], 88) if np.any(fine > 0) else fine > 0
    coarse_mask = coarse > np.percentile(coarse[coarse > 0], 85) if np.any(coarse > 0) else coarse > 0
    mask = (fine_mask & coarse_mask) | coarse_mask
    
    struct = generate_binary_structure(2, 2)
    mask = binary_opening(mask, structure=struct, iterations=1)
    mask = binary_closing(mask, structure=struct, iterations=3)
    
    labeled, num = ndimage.label(mask)
    for i in range(1, num + 1):
        if np.sum(labeled == i) < 150:
            mask[labeled == i] = False
    
    regions = regionprops(label(mask))
    filtered = np.zeros_like(mask, dtype=bool)
    
    for region in regions:
        if region.area < 100:
            continue
        
        minr, minc, maxr, maxc = region.bbox
        aspect = (maxc - minc) / (maxr - minr + 1e-8)
        angle = np.abs(np.degrees(region.orientation))
        solidity = region.solidity
        
        is_horizontal = angle < 15 or angle > 165
        is_elongated = aspect > 8
        is_straight = solidity > 0.85
        
        if not (is_horizontal and is_elongated and is_straight) or solidity < 0.75:
            for coord in region.coords:
                filtered[coord[0], coord[1]] = True
    
    gray = np.mean(original_image, axis=2)
    edges_x = sobel(gray, axis=0)
    edges_y = sobel(gray, axis=1)
    edge_mag = np.sqrt(edges_x**2 + edges_y**2)
    edge_mag = (edge_mag - edge_mag.min()) / (edge_mag.max() - edge_mag.min() + 1e-8)
    edge_mask = edge_mag > 0.04
    filtered = filtered & edge_mask
    
    return coarse * filtered

def generate_visualizations(model, top_indices, names, images_dir):
    output_dir = f'bent_wires_latent{LATENT_DIM}'
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    print("\nGenerating visualizations...\n")

    for i, idx in enumerate(top_indices):
        img_path = os.path.join(images_dir, names[idx])
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            recon = model(img_tensor)

        error = torch.abs(img_tensor - recon).squeeze(0).cpu()
        error = torch.mean(error, dim=0).numpy()
        original = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        bent_wires = detect_bent_wires(error, original)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(original)
        axes[0].imshow(bent_wires, cmap='jet', alpha=0.7)
        axes[0].set_title(f'Original\n{names[idx]}', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(original)
        axes[1].imshow(error, cmap='jet', alpha=0.6)
        axes[1].set_title('Raw Errors', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/anomaly_{i+1}_{names[idx]}', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: anomaly_{i+1}_{names[idx]}")

    print("\nCreating summary grid...")
    
    display_indices = top_indices[:9] if len(top_indices) >= 9 else top_indices
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()

    for i, idx in enumerate(display_indices):
        img_path = os.path.join(images_dir, names[idx])
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            recon = model(img_tensor)

        error = torch.abs(img_tensor - recon).squeeze(0).cpu()
        error = torch.mean(error, dim=0).numpy()
        original = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        bent_wires = detect_bent_wires(error, original)

        threshold_95 = np.percentile(error, 95)
        
        # create mask for high-error regions
        high_error_mask = (error >= threshold_95).astype(np.float32)
        high_error_mask_3d = high_error_mask[:, :, np.newaxis]
        
        # darken low-error regions (30% brightness), keep full brightness for anomalies
        darkening_factor = np.where(high_error_mask_3d == 1, 1.0, 0.3)
        enhanced_image = original * darkening_factor
        
        axes[i].imshow(enhanced_image)
        axes[i].set_title(f'#{i+1}: {names[idx]}', fontsize=10, fontweight='bold')
        axes[i].axis('off')
    
    for i in range(len(display_indices), 9):
        axes[i].axis('off')

    plt.suptitle('Top 9 Bent Wire Anomalies - Enhanced View', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/summary.png")
    print(f"\n✅ All results in '{output_dir}/'")

def main():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    print("\n" + "="*70)
    print("BENT WIRE DETECTION")
    print("="*70)
    
    images_dir = '/home/asoultana/infineon-hackathon-bruteforce/Images'
    dataset = ImageDataset(images_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    print(f"Loaded {len(dataset)} images")

    model = train_model(dataset, dataloader)
    top_indices, names = compute_anomalies(model, dataset)
    generate_visualizations(model, top_indices, names, images_dir)

    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)

if __name__ == '__main__':
    main()
