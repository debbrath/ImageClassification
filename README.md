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

```├── app.py                  # Flask backend
├── best_cnn_model.keras    # Trained CNN model (add yours here)
├── requirements.txt
├── templates/
│   └── index.html          # Stylish HTML UI
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
![Screenshot](https://github.com/mahisalman/Fashion_MNIST_Classifier/blob/main/Screenshot_2.png)
![Screenshot](https://github.com/mahisalman/Fashion_MNIST_Classifier/blob/main/Screenshot_4.png)
![Screenshot](https://github.com/mahisalman/Fashion_MNIST_Classifier/blob/main/Screenshot_5.png)
![Screenshot](https://github.com/mahisalman/Fashion_MNIST_Classifier/blob/main/Screenshot_6.png)

✍️ Author

Mahi Salman

📫 [Connect on LinkedIn](https://www.linkedin.com/in/mahisalman/)

🌐 [GitHub Profile](https://github.com/mahisalman)

