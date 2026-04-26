from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import uvicorn

# ── Model Definition (must match training) ──────────────────────────────────
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(4 * 4 * 128, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# ── Constants ────────────────────────────────────────────────────────────────
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

CLASS_EMOJIS = {
    "airplane": "✈️", "automobile": "🚗", "bird": "🐦", "cat": "🐱",
    "deer": "🦌", "dog": "🐶", "frog": "🐸", "horse": "🐴",
    "ship": "🚢", "truck": "🚛"
}

TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="CIFAR-10 CNN Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
device = torch.device("cpu")
model = CNN().to(device)

try:
    model.load_state_dict(torch.load("cnn_cifar10.pth", map_location=device))
    print("✅ Model loaded from cnn_cifar10.pth")
except FileNotFoundError:
    print("⚠️  cnn_cifar10.pth not found!")

model.eval()

# ── Read index.html once at startup ─────────────────────────────────────────
with open("index.html", "r", encoding="utf-8") as f:
    _RAW_HTML = f.read()

# Patch the API URL so it works on any host (HF Spaces URL changes per user)
_HTML = _RAW_HTML.replace(
    'const API = "http://localhost:8000"',
    'const API = window.location.origin'
)

# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return HTMLResponse(content=_HTML)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        top_idx = torch.argmax(probs).item()
        confidence = probs[top_idx].item()

    top3 = torch.topk(probs, 3)
    top3_preds = [
        {
            "label": CLASSES[i],
            "emoji": CLASS_EMOJIS[CLASSES[i]],
            "confidence": round(probs[i].item() * 100, 2),
        }
        for i in top3.indices.tolist()
    ]

    return {
        "prediction": CLASSES[top_idx],
        "emoji": CLASS_EMOJIS[CLASSES[top_idx]],
        "confidence": round(confidence * 100, 2),
        "top3": top3_preds,
        "all_probs": {c: round(probs[i].item() * 100, 2) for i, c in enumerate(CLASSES)},
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
