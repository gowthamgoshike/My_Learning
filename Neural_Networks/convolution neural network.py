
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Neural Network Class
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # --- LAYER 1: The Feature Extractor ---
        # Input: 1 channel (grayscale image), Output: 10 feature maps
        # Kernel: 3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3)
        
        # --- LAYER 2: Another Feature Extractor ---
        # Input: 10 channels (from prev layer), Output: 20 feature maps
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3)
        
        # --- POOLING: The Compressor ---
        # 2x2 Max Pooling (cuts dimensions in half)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- LAYER 3: The Classifier ---
        # We need to flatten the 2D maps into a 1D vector.
        # Calculation:
        # Image starts 28x28. 
        # After Conv1 (no padding, 3x3) -> 26x26. 
        # After Pool -> 13x13.
        # After Conv2 (no padding, 3x3) -> 11x11.
        # After Pool -> 5x5.
        # Result: 20 channels * 5 * 5 = 500 inputs
        self.fc1 = nn.Linear(in_features=20 * 5 * 5, out_features=10) # 10 outputs (digits 0-9)

    def forward(self, x):
        # 1. Conv1 -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv1(x)))
        
        # 2. Conv2 -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv2(x)))
        
        # 3. Flatten (Batch_Size, 500)
        x = x.view(-1, 20 * 5 * 5)
        
        # 4. Fully Connected Layer
        x = self.fc1(x)
        return x

if __name__ == "__main__":
    # Simulate a random grayscale image (Batch Size=1, Channels=1, Height=28, Width=28)
    dummy_image = torch.rand(1, 1, 28, 28)
    
    model = SimpleCNN()
    output = model(dummy_image)
    print(f"Input Shape: {dummy_image}")
    print("CNN Architecture")
    print(f"Input Shape: {dummy_image.shape}")
    print(f"Output Shape: {output.shape}") # Should be [1, 10]
    print(f"Raw Predictions: \n{output.detach().numpy()}")