import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
from typing import Dict, List, Any, Union
from minimal_preprocessing import MinimalImageProcessor  # Import the new processor

class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class AdvancedResNet50(nn.Module):
    def __init__(self, num_classes: int = 7, pretrained: bool = True, dropout_rate: float = 0.5):
        super(AdvancedResNet50, self).__init__()
        
        # Load pretrained ResNet50 backbone
        if pretrained:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet50(weights=None)
            
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        self.num_features = 2048
        
        # Attention mechanisms
        self.channel_attention = ChannelAttention(self.num_features)
        self.spatial_attention = SpatialAttention()
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature enhancement layers
        self.feature_enhancer = nn.Sequential(
            nn.Linear(self.num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.4)
        )
        
        # Cultural adaptation layer
        self.cultural_adapter = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3)
        )
        
        # Final emotion classifier
        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor, alpha: float = 0.0) -> torch.Tensor:
        # Extract features using ResNet50 backbone
        features = self.features(x)
        
        # Apply attention mechanisms
        channel_att = self.channel_attention(features)
        features = features * channel_att
        
        spatial_att = self.spatial_attention(features)
        features = features * spatial_att
        
        # Global pooling and flattening
        pooled_features = self.global_pool(features)
        flattened_features = pooled_features.view(pooled_features.size(0), -1)
        
        # Feature enhancement
        enhanced_features = self.feature_enhancer(flattened_features)
        
        # Cultural adaptation
        adapted_features = self.cultural_adapter(enhanced_features)
        
        # Final emotion classification
        emotion_output = self.emotion_classifier(adapted_features)
        
        return emotion_output

class EmotionPredictor:
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.emotion_colors = {
            'Angry': '#FF6B6B',
            'Disgust': '#4ECDC4',
            'Fear': '#45B7D1',
            'Happy': '#96CEB4',
            'Neutral': '#FFEAA7',
            'Sad': '#DDA0DD',
            'Surprise': '#FFB347'
        }
        
        # Initialize minimal image processor
        self.image_processor = MinimalImageProcessor()
        
        # Image preprocessing pipeline - matching training exactly
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load the model
        self.model = self._load_model_fixed(model_path)

    def _load_model_fixed(self, model_path: str) -> AdvancedResNet50:
        """Fixed model loading to ensure proper weight loading"""
        try:
            print(f"🔍 Loading model from: {model_path}")
            
            # Try different loading approaches to find the correct one
            
            # Approach 1: Try loading entire model first (if saved with torch.save(model, path))
            try:
                print("🔄 Attempting to load entire model...")
                model = torch.load(model_path, map_location=self.device)
                if hasattr(model, 'eval') and callable(getattr(model, 'eval')):
                    model.eval()
                    print("✅ Successfully loaded entire model")
                    return model
            except Exception as e:
                print(f"ℹ️ Entire model loading failed: {e}")
            
            # Approach 2: Load state dict (your current approach)
            print("🔄 Loading model state dict...")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Check if checkpoint is the state dict itself or contains state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("📋 Found 'model_state_dict' key in checkpoint")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("📋 Found 'state_dict' key in checkpoint")
            else:
                state_dict = checkpoint
                print("📋 Using checkpoint as state dict directly")
            
            # Create model with same parameters as training
            model = AdvancedResNet50(num_classes=7, pretrained=False, dropout_rate=0.5)
            
            # Filter domain classifier weights
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if not key.startswith('domain_classifier'):
                    filtered_state_dict[key] = value
            
            print(f"📊 Filtered state dict keys: {len(filtered_state_dict)}")
            
            # Load with strict=False and check for issues
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️ Missing keys: {len(missing_keys)}")
                print(f"🔍 First few missing: {missing_keys[:3]}")
                
            if unexpected_keys:
                print(f"⚠️ Unexpected keys: {len(unexpected_keys)}")
                print(f"🔍 First few unexpected: {unexpected_keys[:3]}")
            
            # Ensure proper evaluation mode
            model.to(self.device)
            model.eval()
            
            # Critical: Set all batch norm and dropout layers properly
            for module in model.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    module.eval()
                    module.track_running_stats = False
                elif isinstance(module, nn.Dropout):
                    module.eval()
            
            print("✅ Model loaded and configured for inference")
            
            # Validate model produces reasonable outputs
            self._validate_model_predictions(model)
            
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e
    
    def _validate_model_predictions(self, model: AdvancedResNet50):
        """Validate that model produces reasonable and diverse predictions"""
        print("🧪 Validating model predictions...")
        
        with torch.no_grad():
            # Test with multiple random inputs
            predictions = []
            confidences = []
            
            for i in range(10):
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                output = model(dummy_input)
                probabilities = torch.softmax(output, dim=1)
                
                predicted_class = int(torch.argmax(probabilities, dim=1).item())
                confidence = float(probabilities[0][predicted_class].item())
                
                predictions.append(predicted_class)
                confidences.append(confidence)
            
            # Check diversity
            unique_predictions = len(set(predictions))
            avg_confidence = sum(confidences) / len(confidences)
            
            print(f"🎯 Unique predictions out of 10: {unique_predictions}")
            print(f"💯 Average confidence: {avg_confidence:.3f}")
            print(f"📊 Prediction distribution: {predictions}")
            
            # Check for issues
            if unique_predictions <= 2:
                print("⚠️ WARNING: Very low prediction diversity!")
            if avg_confidence < 0.2:
                print("⚠️ WARNING: Very low average confidence!")
            if avg_confidence > 0.9:
                print("⚠️ WARNING: Suspiciously high confidence!")
            
            # Test probability sum
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            output = model(dummy_input)
            probabilities = torch.softmax(output, dim=1)
            prob_sum = float(probabilities.sum().item())
            
            print(f"✅ Probability sum check: {prob_sum:.6f} (should be ~1.0)")
            
            if abs(prob_sum - 1.0) > 0.001:
                print("⚠️ WARNING: Probabilities don't sum to 1!")

    def preprocess_image(self, image_file: Union[bytes, Any]) -> torch.Tensor:
        """Enhanced preprocessing with minimal preprocessing integration"""
        try:
            print("🔄 Starting enhanced image preprocessing...")
            
            # Use minimal preprocessing for better quality
            if hasattr(image_file, 'read'):
                # Flask FileStorage object
                processed_pil = self.image_processor.process_flask_file(image_file)
            else:
                # Handle other formats
                if isinstance(image_file, bytes):
                    image = Image.open(io.BytesIO(image_file)).convert('RGB')
                else:
                    image = Image.open(image_file).convert('RGB')
                processed_pil = image
            
            print(f"📸 Preprocessed image: {processed_pil.mode}, {processed_pil.size}")
            
            # Apply model transforms
            input_tensor = self.transform(processed_pil).unsqueeze(0) # type: ignore
            input_tensor = input_tensor.to(self.device)
            
            return input_tensor
            
        except Exception as e:
            print(f"❌ Enhanced preprocessing failed: {e}")
            # Fallback to basic preprocessing
            try:
                if isinstance(image_file, bytes):
                    image = Image.open(io.BytesIO(image_file)).convert('RGB')
                else:
                    image = Image.open(image_file).convert('RGB')
                
                input_tensor = self.transform(image).unsqueeze(0)
                input_tensor = input_tensor.to(self.device)
                return input_tensor
            except Exception as fallback_error:
                raise ValueError(f"Complete preprocessing failure: {fallback_error}")

    def predict_emotion(self, image_file: Any) -> Dict[str, Any]:
        """Enhanced prediction with proper confidence calculation and minimal preprocessing"""
        try:
            print("🔮 Starting emotion prediction with minimal preprocessing...")
            
            # Preprocess image using minimal preprocessing
            input_tensor = self.preprocess_image(image_file)
            
            # Make prediction with proper evaluation mode
            self.model.eval()
            
            with torch.no_grad():
                # Forward pass
                raw_output = self.model(input_tensor)
                
                print(f"🧠 Raw output shape: {raw_output.shape}")
                print(f"📊 Raw output range: [{raw_output.min():.3f}, {raw_output.max():.3f}]")
                
                # Apply softmax to get proper probabilities that sum to 1
                probabilities = torch.softmax(raw_output, dim=1)
                
                # Verify probabilities sum to 1
                prob_sum = float(probabilities.sum().item())
                print(f"✅ Probability sum: {prob_sum:.6f}")
                
                # Get predicted class and confidence
                predicted_class = int(torch.argmax(probabilities, dim=1).item())
                confidence = float(probabilities[0][predicted_class].item())
                
                # Get all probabilities as percentages that sum to 100%
                all_probs = probabilities[0].cpu().numpy()
                
                # Ensure they sum to exactly 100% (handle floating point precision)
                all_probs_percent = all_probs * 100
                prob_sum_percent = all_probs_percent.sum()
                
                # Normalize to ensure exact 100% sum if needed
                if abs(prob_sum_percent - 100.0) > 0.01:
                    all_probs_percent = all_probs_percent * (100.0 / prob_sum_percent)
                
                print(f"🎯 Predicted: {self.emotions[predicted_class]} ({confidence:.3f})")
                print(f"📊 All probabilities sum: {all_probs_percent.sum():.2f}%")
            
            # Prepare result with proper percentages
            result = {
                'emotion': self.emotions[predicted_class],
                'confidence': round(float(confidence * 100), 2),
                'color': self.emotion_colors[self.emotions[predicted_class]],
                'preprocessing_applied': True,
                'preprocessing_type': 'Minimal Quality Preserving',
                'all_probabilities': [
                    {
                        'emotion': self.emotions[i],
                        'probability': round(float(all_probs_percent[i]), 2),
                        'color': self.emotion_colors[self.emotions[i]]
                    }
                    for i in range(len(self.emotions))
                ]
            }
            
            # Verify the sum is 100%
            total_percent = sum(item['probability'] for item in result['all_probabilities'])
            print(f"📈 Final percentage sum: {total_percent:.2f}%")
            
            return result
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            raise ValueError(f"Error during prediction: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_name': 'Advanced ResNet50',
            'num_classes': len(self.emotions),
            'emotions': self.emotions,
            'accuracy': '96.43%',
            'training_type': 'Cross-Cultural (Japanese → Indian)',
            'preprocessing': 'Minimal Quality Preserving',
            'device': str(self.device),
            'features': [
                'Channel Attention',
                'Spatial Attention', 
                'Cultural Adaptation',
                'Feature Enhancement',
                'Minimal Preprocessing'
            ]
        }

# Utility functions
def create_model_instance(model_path: str, device: str = 'cpu') -> EmotionPredictor:
    return EmotionPredictor(model_path, device)

def validate_model_file(model_path: str) -> bool:
    try:
        import os
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
        
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"📁 Model file size: {file_size:.2f} MB")
        
        if file_size < 5:
            print(f"⚠️ Model file seems too small: {file_size:.2f} MB")
            return False
        
        # Test loading
        torch.load(model_path, map_location='cpu')
        print(f"✅ Model file is valid: {model_path}")
        return True
        
    except Exception as e:
        print(f"❌ Invalid model file: {e}")
        return False
