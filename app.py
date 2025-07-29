from flask import Flask, request, render_template
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
from scipy.ndimage import center_of_mass, shift
import numpy as np
from io import BytesIO
import base64
from cnn_model import CNNModel



app = Flask(__name__)

# Load PyTorch model
#model = torch.load("best_cnn_model.pt", map_location=torch.device('cpu'))
#model.eval()

model = CNNModel()
model.load_state_dict(torch.load('best_cnn_model.pt', map_location='cpu'))
model.eval()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def preprocess_image(image):
    # Convert to grayscale
    image = image.convert("L")

    # Invert and resize to 28x28 with aspect ratio preserved
    image = ImageOps.invert(image)
    image = ImageOps.fit(image, (28, 28), Image.Resampling.LANCZOS)

    # Convert to numpy array and normalize
    img_array = np.array(image).astype(np.float32) / 255.0    
    # Threshold low values
    img_array[img_array < 0.2] = 0.0

    # Center the mass
    cy, cx = center_of_mass(img_array)
    if np.isnan(cx) or np.isnan(cy):
        cx, cy = 14, 14
    shift_y = np.round(14 - cy).astype(int)
    shift_x = np.round(14 - cx).astype(int)
    img_array = shift(img_array, shift=(shift_y, shift_x), mode='constant', cval=0.0)

    # Normalize as in training
    img_array = (img_array - 0.5) / 0.5
    # Convert to torch tensor
    tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 28, 28]
    return tensor

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    prediction = None
    image_data = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            image = Image.open(file.stream)
            input_tensor = preprocess_image(image)

            # Get prediction
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1).numpy()[0]
                #probabilities-------
                print("Class probabilities:")
                for i, prob in enumerate(probs):
                    print(f"{class_names[i]}: {prob:.4f}")
                #probabilities-------
                top_indices = probs.argsort()[-3:][::-1]

                prediction = [
                    class_names[top_indices[0]], round(probs[top_indices[0]] * 100, 2),
                    class_names[top_indices[1]], round(probs[top_indices[1]] * 100, 2),
                    class_names[top_indices[2]], round(probs[top_indices[2]] * 100, 2),
                    class_names[top_indices[3]], round(probs[top_indices[3]] * 100, 2),
                ]

            # Convert uploaded image to base64 for display
            buffered = BytesIO()
            image.convert("RGB").save(buffered, format="PNG")
            image_data = base64.b64encode(buffered.getvalue()).decode()

    return render_template('index.html', prediction=prediction, image_data=image_data)

if __name__ == '__main__':
    app.run(debug=True)
