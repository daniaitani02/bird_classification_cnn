from flask import Flask, request, render_template
import torch
from PIL import Image
from torchvision import transforms
from io import BytesIO
from src.model import build_model  
import os

app = Flask(__name__)

def get_class_names(data_directory):
    # List the directories and sort them alphabetically to maintain order
    class_names = sorted(os.listdir(data_directory))
    return class_names
class_names = get_class_names('data/train')
# print('Class Names:',class_names)
def load_model(model_path, num_classes):
    model = build_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model
num_classes = len(class_names) 
model = load_model('models/bird_classification_model.pth', num_classes)



def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def model_predict(image_bytes, mode,class_namesl):
    preprocessed_image = preprocess_image(image_bytes)
    with torch.no_grad():
        outputs = model(preprocessed_image)
    predicted_index = outputs.argmax().item()
    # print("Model's index output: ", predicted_index)
    predicted_class_name = class_names[predicted_index]
    return predicted_class_name


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename:
            image_bytes = file.read()
            prediction = model_predict(image_bytes, model,class_names)
            return render_template('prediction.html', prediction=prediction)
        else:
            return render_template('index.html', error="No file selected or file is not an image.")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')
