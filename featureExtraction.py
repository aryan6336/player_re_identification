import torch
import torchvision.transforms as T
import torch.nn as nn
import numpy as np
import cv2
import pytesseract
from torchvision.models import resnet18

class FeatureExtractor:
    def __init__(self, use_cuda=True):
        # Load pretrained ResNet18 model (remove classification head)
        base_model = resnet18(pretrained=True)
        self.model = nn.Sequential(*list(base_model.children())[:-1])  # remove fc layer
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.model.to(self.device)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def extract_appearance_features(self, image):
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(image).squeeze().cpu().numpy()  # (512,)
        return features

    def extract_jersey_number(self, image):
        # Preprocess for better OCR accuracy
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshed = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(threshed, config='--psm 6 digits')
        digits = ''.join(filter(str.isdigit, text))
        if digits.isdigit():
            return int(digits)
        return -1  # jersey number not found

    def jersey_embedding(self, jersey_number):
        # One-hot for 0-99 jersey numbers
        one_hot = np.zeros(100)
        if 0 <= jersey_number < 100:
            one_hot[jersey_number] = 1
        return one_hot

    def extract_features(self, frame, bbox):
        """
        Extracts combined appearance + jersey number features.
        bbox: (x, y, w, h)
        """
        x, y, w, h = bbox
        x, y = max(x, 0), max(y, 0)
        cropped = frame[y:y+h, x:x+w]

        if cropped.size == 0:
            return np.zeros(612)  # 512 (appearance) + 100 (jersey)

        appearance_feat = self.extract_appearance_features(cropped)
        jersey_number = self.extract_jersey_number(cropped)
        jersey_feat = self.jersey_embedding(jersey_number)

        full_feature = np.concatenate([appearance_feat, jersey_feat])  # (612,)
        return full_feature
