import torch
from torchvision import models, transforms
from PIL import Image
import gradio as gr

class_names = ['recyclable-waste', 'general-waste', 'compostable-waste', 'hazardous-waste']

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("waste_classifier.pth", map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict(image):
    image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
    return f"{class_names[pred.item()]} ({conf.item()*100:.2f}%)"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Garbage Classification",
    description="Upload the image of any trash to classify into 4 types."
)

if __name__ == "__main__":
    demo.launch()
