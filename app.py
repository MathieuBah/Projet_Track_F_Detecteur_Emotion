import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ── Configuration de la page ──
st.set_page_config(
    page_title="Detecteur d'emotions faciales",
    page_icon="🔬",
    layout="wide"
)

# ── Définir les transforms ──
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ImageNet mean and std for denormalization
mean_t = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std_t  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Classes names
class_names = ['angry', 'happy', 'neutral', 'sad']

# Wrapper pour que GradCAM accède aux logits via pixel_values
class ConvNeXtWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(pixel_values=x).logits

# ── Charger le modèle une seule fois avec caching ──
@st.cache_resource
def load_model_and_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForImageClassification.from_pretrained(
        'facebook/convnext-tiny-224',  # ConvNeXt-Tiny pré-entraîné sur ImageNet
        num_labels=4,                  # 4 émotions : angry, happy, neutral, sad
        ignore_mismatched_sizes=True,  # remplace la tête 1000 classes par une tête 4 classes
    )
    model = model.to(device)

    # Geler tous les paramètres du backbone ConvNeXt (tout sauf la tête)
    for param in model.convnext.parameters():
        param.requires_grad = False

    model.eval() # Set model to evaluation mode
    return model, device

model, device = load_model_and_device()
wrapped_model = ConvNeXtWrapper(model)
# Target layer for ConvNeXt
target_layer = [model.convnext.encoder.stages[-1].layers[-1]]

# ── Sidebar ──
st.sidebar.title("A propos")
st.sidebar.info(
    "Prototype de recherche — la reconnaissance d'emotions par IA est un sujet controverse. Les resultats sont approximatifs et culturellement biaises."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Modele :** ConvNeXt-Tiny")
st.sidebar.markdown("**Classes :** angry, happy, neutral, sad")

# ── Page principale ──
st.title("Detecteur d'emotions faciales")
st.markdown(
    "Uploadez une image pour obtenir une prediction "
    "avec score de confiance et visualisation GradCAM."
)

# ── Upload d'image ──
uploaded_file = st.file_uploader(
    "Choisir une image",
    type=["jpg", "jpeg", "png"],
    help="Formats acceptes : JPG, JPEG, PNG"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Layout en 2 colonnes
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image originale")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Analyse")

        with st.spinner("Analyse en cours..."):
            # 1. Appliquer val_transform a l'image
            input_tensor = val_transform(image).unsqueeze(0).to(device)

            # 2. Predire avec le modele
            with torch.no_grad():
                outputs = model(pixel_values=input_tensor)
                logits = outputs.logits

            # 3. Calculer les probabilites (softmax)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = probs.argmax().item()
            confidence = probs.max().item()
            predicted_class_name = class_names[pred_idx]

            # 4. Generer le GradCAM
            # Temporarily enable gradients for the convnext backbone for GradCAM computation
            original_convnext_requires_grad_state = {}
            for name, param in model.convnext.named_parameters():
                original_convnext_requires_grad_state[name] = param.requires_grad
                param.requires_grad = True

            with GradCAM(model=wrapped_model, target_layers=target_layer, use_cuda=True if device.type=='cuda' else False) as cam:
                grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

            # Restore original requires_grad state
            for name, param in model.convnext.named_parameters():
                param.requires_grad = original_convnext_requires_grad_state[name]

            # Denormalize the original image for display with GradCAM overlay
            # Convert PIL image to numpy array, then normalize for display
            # (image is already PIL.Image in RGB, val_transform expects PIL, then makes it tensor)
            img_display_pil = image.resize((224, 224))
            img_display_np = np.array(img_display_pil).astype(np.float32) / 255.0
            if img_display_np.ndim == 2: # if Grayscale after resize
                img_display_np = np.stack([img_display_np]*3, axis=-1)

            cam_image = show_cam_on_image(img_display_np, grayscale_cam, use_rgb=True)

            # 5. Afficher les resultats
            st.metric(label="Prediction", value=predicted_class_name, delta=f"{confidence:.1%}")
            st.image(cam_image, caption="Heatmap GradCAM", use_container_width=True)

    # ── Disclaimer en bas ──
    st.markdown("---")
    st.warning("Prototype de recherche — la reconnaissance d'emotions par IA est un sujet controverse. Les resultats sont approximatifs et culturellement biaises.")

else:
    st.info("Uploadez une image pour commencer l'analyse.")
