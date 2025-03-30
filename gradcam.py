import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

def save_gradcam_image(img_tensor, heatmap, path):
    img = img_tensor.squeeze()  # [1, H, W] or [H, W]

    if img.dim() == 2:
        img = img.unsqueeze(0)  # [1, H, W]
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)  # Convert to 3-channel

    img = img.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # Normalize to [0, 1]
    img = (img * 255).astype(np.uint8)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Combine heatmap and image
    superimposed = cv2.addWeighted(heatmap_color, 0.4, img, 0.6, 0)

    # Make sure it's a valid image
    if superimposed is not None and superimposed.shape[0] > 0 and superimposed.shape[1] > 0:
        cv2.imwrite(path, superimposed)
    else:
        print("⚠️ Warning: Invalid image. Skipping save.")

def generate_gradcam(model, image_tensor, class_idx, target_layer, save_path):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_backward_hook(backward_hook)

    model.eval()
    output = model(image_tensor) 
    model.zero_grad()
    output[0, class_idx].backward()

    grad = gradients[0]
    act = activations[0]

    weights = grad.mean(dim=1, keepdim=True)  # [B, 1, D]
    cam = (weights * act).sum(dim=2).squeeze().detach().cpu().numpy()  # [L]
    cam = np.maximum(cam, 0)
    cam = cam / cam.max() if cam.max() != 0 else cam
    cam = cam.reshape(16, 16)  

    save_gradcam_image(image_tensor, cam, save_path)

    handle_fw.remove()
    handle_bw.remove()
