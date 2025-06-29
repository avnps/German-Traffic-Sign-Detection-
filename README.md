# German-Traffic-Sign-Detection

# 🚦 Traffic Sign Classifier (GTSRB Dataset)

This project implements a Traffic Sign Classifier using a Convolutional Neural Network (CNN) trained on the [GTSRB Dataset](https://benchmark.ini.rub.de/gtsrb_news.html). It includes training, evaluation, exploratory data analysis (EDA), and a Streamlit app for real-time predictions.

---

## 📁 Project Structure

```
.
├── app.py           # Streamlit app to upload and classify traffic sign images
├── train.py         # Training script for CNN on GTSRB dataset
├── evaluate.py      # Evaluate model on test dataset
├── EDA.py           # Data exploration and visualization
├── model.py         # CNN model architecture (TrafficSignNet)
├── traffic_sign_model.pth  # Trained model weights (generated after training)
├── *.p              # Pickle files: train.p, valid.p, test.p
```

---

## 🧠 Model Architecture (TrafficSignNet)

- 2 convolutional layers with ReLU + MaxPooling
- Dropout layer for regularization
- Fully connected layer with 256 units
- Output layer with 43 classes (softmax)

---

## 🧪 Running the Project

### 1. Install Dependencies

```bash
pip install torch torchvision matplotlib seaborn streamlit
```

### 2. Train the Model

```bash
python train.py
```

This will:
- Load `train.p` and `valid.p`
- Train for 10 epochs
- Save the model to `traffic_sign_model.pth`

### 3. Evaluate the Model

```bash
python evaluate.py
```

Outputs:
- Test Loss
- Accuracy on `test.p`

### 4. Explore the Dataset

```bash
python EDA.py
```

Shows:
- Class distribution bar plot
- Random sample image grid with labels

### 5. Run the Streamlit App

```bash
streamlit run app.py
```

Upload a `.jpg`, `.png`, or `.ppm` traffic sign image and get instant classification with label.

---

## 🎯 Dataset

Make sure you have the following pickle files from the GTSRB dataset preprocessing:

- `train.p`
- `valid.p`
- `test.p`

---

## 📜 License

MIT License. Free to use, modify, and share.
