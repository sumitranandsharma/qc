import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import io
import base64

class FeatureExtractor:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = models.ResNet18_Weights.DEFAULT
        self.base_model = models.resnet18(weights=weights)
        # Move full base model to self.device for Grad-CAM
        self.base_model.to(self.device)
        self.base_model.eval()
        
        # Keep feature extractor up to the average pooling layer (512-dim output)
        self.features_layer = nn.Sequential(*list(self.base_model.children())[:-1])
        self.features_layer.to(self.device)
        self.features_layer.eval()

        # Target layer for Grad-CAM
        self.target_conv = self.base_model.layer4[-1]

        # Standard transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def apply_clahe(pil_img):
        """Enhance contrast of chest X-ray using CLAHE."""
        img_np = np.array(pil_img.convert("L"))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_np)
        return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))

    def preprocess_image(self, pil_img, use_clahe=True):
        """
        Preprocesses a PIL Image.
        Returns:
            processed_tensor: (1, 3, 224, 224) torch.Tensor
            clahe_img: PIL Image with CLAHE enhancement
        """
        if use_clahe:
            enhanced_pil = self.apply_clahe(pil_img)
        else:
            enhanced_pil = pil_img.convert("RGB")
        
        tensor = self.transform(enhanced_pil).unsqueeze(0).to(self.device)
        return tensor, enhanced_pil

    @torch.no_grad()
    def extract_features(self, tensor):
        """
        Extracts 512-dimensional latent feature vector from preprocessed tensor.
        Returns: numpy array of shape (512,)
        """
        feat = self.features_layer(tensor)
        feat = feat.view(feat.size(0), -1)
        return feat.cpu().numpy().squeeze(0)

    def generate_gradcam(self, tensor, original_pil):
        """
        Computes Grad-CAM attention heatmap overlay for the chest X-ray.
        Returns base64 encoded JPEG string.
        """
        activations = []
        gradients = []

        def forward_hook(module, inp, out):
            activations.append(out)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        h1 = self.target_conv.register_forward_hook(forward_hook)
        h2 = self.target_conv.register_full_backward_hook(backward_hook)

        # Forward pass through base model
        tensor_req = tensor.clone().detach().requires_grad_(True)
        out = self.base_model(tensor_req)
        score = torch.max(out)

        self.base_model.zero_grad()
        score.backward()

        h1.remove()
        h2.remove()

        if not activations or not gradients:
            # Fallback blank heatmap
            return None

        act = activations[0].detach().cpu().numpy()[0]
        grad = gradients[0].detach().cpu().numpy()[0]

        # Channel-wise global pooling of gradients
        weights = np.mean(grad, axis=(1, 2))
        cam = np.zeros(act.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * act[i]

        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        cam = cv2.resize(cam, (224, 224))

        # Overlay on image
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        orig_resized = np.array(original_pil.convert("RGB").resize((224, 224)))
        overlay = np.uint8(0.6 * orig_resized + 0.4 * heatmap)

        overlay_pil = Image.fromarray(overlay)
        buffered = io.BytesIO()
        overlay_pil.save(buffered, format="JPEG", quality=90)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
