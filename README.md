# 🧥 Fashion MNIST Image Classifier – CNN + Flask Web App

This project is a full-stack web application that allows users to upload an image of a fashion item (e.g., shirt, sneaker, bag) and classifies it using a Convolutional Neural Network (CNN) trained on the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset.

The app features:
- 🔍 Real-time image classification
- 🧠 Robust preprocessing (resize, normalize, center the object)
- 📸 Live preview of uploaded image
- 💡 Displays top 3 predictions with confidence scores
- 🎨 Beautiful and responsive HTML/CSS interface

---

## 🚀 Live Demo

> _Want to see it in action?_  
> 👉 [Coming Soon: Deployed on Render or Hugging Face Spaces]

---

## 🧠 Model Details

- **Architecture**: 3-layer CNN with BatchNorm, Dropout, and MaxPooling
- **Input shape**: 28×28 grayscale images
- **Training Dataset**: Fashion MNIST (60,000 training + 10,000 test images)
- **Accuracy**: ~91% on test data

---

## 🖼️ Supported Classes

``` ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] ```

---

## 🧪 Try It Locally

### 1️⃣ Clone the repo

bash
git clone https://github.com/YOUR_USERNAME/fashion-mnist-classifier.git
cd fashion-mnist-classifier

2️⃣ Install requirements

pip install -r requirements.txt

3️⃣ Run the app

python app.py

Open your browser: http://127.0.0.1:5000

---

📁 Project Structure

```Image-Classification-AI/
│
├── Image/                      # Folder for example images or datasets
│
├── __pycache__/               # Python cache files (auto-generated)
│
├── data/
│   └── FashionMNIST/
│       └── raw/               # Raw dataset files (FashionMNIST dataset)
│
├── templates/                 # Templates for web app or reports (if any)
│
├── README.md                  # Project description and documentation
│
├── app.py                    # Main application script (e.g., for running or serving the model)
│
├── best_cnn_model.pt          # Saved trained CNN model weights
│
├── cnn_model.py 
```

🛠 Tech Stack

    Python

    TensorFlow / Keras

    Flask

    PIL (Pillow)

    HTML5 + CSS

    JavaScript (Image preview)

    SciPy (for center-of-mass preprocessing)

📸 Screenshots

![Screenshot](https://github.com/debbrath/ImageClassification/blob/main/Image/2025-07-30%2014_25_37-Window.png)

![Screenshot](https://github.com/debbrath/ImageClassification/blob/main/Image/2025-07-30%2014_27_19-Window.png)

![Screenshot](https://github.com/debbrath/ImageClassification/blob/main/Image/2025-07-30%2014_27_41-Window.png)

![Screenshot](https://github.com/debbrath/ImageClassification/blob/main/Image/2025-07-30%2014_28_25-Window.png)

✍️ Author

Debbrath Debnath

📫 [Connect on LinkedIn](https://www.linkedin.com/in/debbrathdebnath/)

🌐 [GitHub Profile](https://github.com/debbrath)

