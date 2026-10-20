<div align="center">

# 🚀 Deep Learning & Computer Vision Practical Exam

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=22&pause=900&color=36BCF7&center=true&vCenter=true&width=850&lines=ANN+Diabetes+Prediction;OpenCV+Image+Processing;CNN+CIFAR-10+Classification;SimpleRNN+Airline+Forecasting;Transformer+Sentiment+Analysis" alt="Animated project topics" />

<p>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow Keras" />
  <img src="https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Transformers" />
</p>

**Five practical exercises in neural networks, computer vision, time-series forecasting, and NLP.**

</div>

---

## 📚 Table of Contents

- [✨ Project Overview](#-project-overview)
- [🧠 Task 1 — ANN Diabetes Prediction](#-task-1--ann-diabetes-prediction)
- [🖼️ Task 2 — OpenCV Image Processing](#-task-2--opencv-image-processing)
- [🌈 Task 3 — CNN CIFAR-10 Classification](#-task-3--cnn-cifar-10-classification)
- [📈 Task 4 — SimpleRNN Airline Forecasting](#-task-4--simplernn-airline-forecasting)
- [💬 Task 5 — Transformer Sentiment Analysis](#-task-5--transformer-sentiment-analysis)
- [🛠️ Technologies Used](#-technologies-used)
- [⚙️ Setup and Run](#-setup-and-run)
- [📂 Suggested Project Structure](#-suggested-project-structure)
- [🎯 Key Learnings](#-key-learnings)
- [⚠️ Notes and Limitations](#-notes-and-limitations)

## ✨ Project Overview

| Task | Project | Technique | Main output |
|---|---|---|---|
| 1 | Diabetes prediction | ANN | Binary classification metrics |
| 2 | Image processing | OpenCV | Resizing, grayscale, blur, edges, contours |
| 3 | CIFAR-10 classification | CNN | 10-class image classification |
| 4 | Airline passenger forecasting | SimpleRNN | Time-series predictions |
| 5 | Sentiment analysis | Pre-trained Transformer | Positive/negative text labels |

**Workflow:** Load data → preprocess → build/load model → evaluate → visualize results.

---

## 🧠 Task 1 — ANN Diabetes Prediction

**Goal:** Predict the diabetes outcome using the Pima Indians Diabetes dataset.

### Workflow
1. Load the CSV and assign meaningful column names.
2. Replace zero values in selected medical measurements (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) with the median of non-zero values.
3. Split features and target; use an 80/20 train-test split with stratification.
4. Standardize using `StandardScaler`, fitted on training data only.
5. Train a Sequential ANN: `Dense(16, relu) → Dense(8, relu) → Dense(1, sigmoid)`.
6. Evaluate loss, accuracy, precision, recall, F1-score, and confusion matrix.

### 📊 Test results

| Metric | Result |
|---|---:|
| Test Loss | **0.5167** |
| Test Accuracy | **71.43%** |
| Precision | **60.87%** |
| Recall | **51.85%** |
| F1-score | **56.00%** |

**Confusion matrix** (rows = actual class, columns = predicted class):

```text
[[82 18]
 [26 28]]
```

- 82 class-0 cases and 28 class-1 cases were correctly classified.
- 18 class-0 cases were predicted as class 1.
- 26 class-1 cases were predicted as class 0.
- Recall of 51.85% means the model identified about half of the positive cases in this test set. Accuracy alone is not enough to assess a healthcare model.

---

## 🖼️ Task 2 — OpenCV Image Processing

**Goal:** Process five images using fundamental computer-vision operations.

### Pipeline

```text
Five images → Load → Resize to 128×128 → Grayscale → Gaussian blur
            → Canny edge detection → Find contours → Draw rectangle
            → Display 5×5 grid → Save processed images
```

### Functions used

| Function | Purpose |
|---|---|
| `cv2.imread()` | Read an image |
| `cv2.resize()` | Change image dimensions |
| `cv2.cvtColor()` | Convert color space |
| `cv2.GaussianBlur()` | Reduce noise |
| `cv2.Canny()` | Detect edges |
| `cv2.findContours()` | Detect contours |
| `cv2.boundingRect()` | Calculate a bounding rectangle |
| `cv2.imwrite()` | Save an image |

Place five images in `images/`, for example:

```text
images/
├── apple.jpg
├── dog.jpg
├── car.jpg
├── flower.jpg
└── ball.jpg
```

The notebook displays five stages for every image: **Original, Resized, Grayscale, Canny Edges, Contours**.

> Add a screenshot of your actual 5×5 output grid after running the notebook. The results image is not embedded in this README yet.

---

## 🌈 Task 3 — CNN CIFAR-10 Classification

**Goal:** Classify images into 10 CIFAR-10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

### Architecture

```text
Input (32×32×3) → Conv2D(32) → MaxPooling
                → Conv2D(64) → MaxPooling
                → Flatten → Dense(64) → Dense(10, Softmax)
```

### 📊 Test results

| Metric | Result |
|---|---:|
| Test Loss | **1.1557** |
| Test Accuracy | **60.25%** |
| Test set size | **4,000 images** |

### Per-class classification report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Airplane | 0.66 | 0.67 | 0.67 | 389 |
| Automobile | 0.79 | 0.72 | 0.75 | 392 |
| Bird | 0.55 | 0.42 | 0.48 | 426 |
| Cat | 0.41 | 0.42 | 0.41 | 389 |
| Deer | 0.45 | 0.60 | 0.51 | 400 |
| Dog | 0.55 | 0.42 | 0.47 | 388 |
| Frog | 0.84 | 0.53 | 0.65 | 396 |
| Horse | 0.63 | 0.70 | 0.66 | 397 |
| Ship | 0.58 | 0.90 | 0.70 | 408 |
| Truck | 0.76 | 0.63 | 0.69 | 415 |
| **Macro average** | **0.62** | **0.60** | **0.60** | **4,000** |
| **Weighted average** | **0.62** | **0.60** | **0.60** | **4,000** |

### Observations
- Overall test accuracy was **60.25%**.
- Automobile had the highest class F1-score (**0.75**).
- Ship had the highest recall (**0.90**).
- Cat had the lowest class F1-score (**0.41**) among these reported classes.

---

## 📈 Task 4 — SimpleRNN Airline Forecasting

**Goal:** Predict monthly airline passenger counts from historical time-series data.

### Workflow
1. Load the airline-passengers CSV.
2. Scale values with `MinMaxScaler`.
3. Create sequences using the previous 12 months.
4. Split the data chronologically (do not shuffle time-series samples).
5. Train `SimpleRNN(50)` followed by `Dense(1)`.
6. Inverse-transform predictions and compare them with actual passenger counts.

### 📊 Test result

| Metric | Result |
|---|---:|
| Test RMSE | **40.57 passengers** |

RMSE is in passenger-count units and penalizes larger errors. Interpret it alongside the actual-versus-predicted plot and the scale of the data.

---

## 💬 Task 5 — Transformer Sentiment Analysis

**Model:** `distilbert-base-uncased-finetuned-sst-2-english`

The notebook uses 25 example sentences: 10 positive, 10 negative, and 5 neutral or ambiguous. The pre-trained model predicts **POSITIVE** or **NEGATIVE** and returns a confidence score.

### Sample results

| Sentence | True sentiment | Predicted label | Confidence |
|---|---|---|---:|
| The meeting starts at ten in the morning. | NEU | POSITIVE | 0.967966 |
| I hate waiting for a long time. | NEG | NEGATIVE | 0.986813 |
| The package arrived yesterday. | NEU | POSITIVE | 0.989624 |
| The movie was okay, but nothing special. | NEU | NEGATIVE | 0.993532 |
| My laptop is slow and keeps crashing. | NEG | NEGATIVE | 0.996090 |

### Observations and limitation
- The shown clearly negative examples are predicted as **NEGATIVE**.
- Neutral examples are forced into POSITIVE or NEGATIVE because this model is a binary classifier.
- A high confidence score does not guarantee a correct prediction; the neutral sentence about the package received a high-confidence POSITIVE prediction.
- Neutral or ambiguous examples should be evaluated separately.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy, Pandas | Data handling and preprocessing |
| Matplotlib | Charts and training curves |
| OpenCV | Image processing |
| Scikit-learn | Scaling, data splitting, evaluation metrics |
| TensorFlow / Keras | ANN, CNN, SimpleRNN |
| PyTorch | Transformer model backend |
| Hugging Face Transformers | Pre-trained sentiment pipeline |
| Jupyter Notebook | Interactive development |

## ⚙️ Setup and Run

### 1. Install dependencies

In a terminal:

```bash
pip install numpy pandas matplotlib scikit-learn opencv-python tensorflow torch transformers
```

Or run `%pip install ...` in a Jupyter Notebook cell.

### 2. Provide the data
- **Task 1:** Pima Indians Diabetes CSV.
- **Task 2:** Five images inside `images/`.
- **Task 3:** CIFAR-10, downloaded by Keras when needed.
- **Task 4:** Airline passenger time-series CSV.
- **Task 5:** Internet access for the first download of the Hugging Face model.

### 3. Run notebooks

Open each `.ipynb` in VS Code or Jupyter and execute cells in order. Use the same Python environment for package installation and the notebook kernel.

## 📂 Suggested Project Structure

```text
deep-learning-practical/
├── README.md
├── notebooks/
│   ├── task1_ann_diabetes.ipynb
│   ├── task2_opencv_image_processing.ipynb
│   ├── task3_cnn_cifar10.ipynb
│   ├── task4_simplernn_airline.ipynb
│   └── task5_transformer_sentiment.ipynb
├── data/
│   ├── pima-indians-diabetes.csv
│   └── airline-passengers.csv
├── images/
│   ├── apple.jpg
│   ├── dog.jpg
│   ├── car.jpg
│   ├── flower.jpg
│   └── ball.jpg
└── output_images/
```

*This is a suggested structure; adjust filenames and folders to match your actual repository.*

## 🎯 Key Learnings

- Built and evaluated an ANN for binary classification.
- Practiced resizing, grayscale conversion, blurring, edge detection, and contours with OpenCV.
- Trained a CNN for 10-class image classification.
- Prepared sequential windows and forecasted time-series values with SimpleRNN.
- Used a pre-trained Transformer for sentiment classification.
- Interpreted accuracy, precision, recall, F1-score, confusion matrices, and RMSE.

## ⚠️ Notes and Limitations

- Reported metrics are the results supplied for these runs; results can change with different splits, random seeds, preprocessing, and training settings.
- The diabetes model is an educational exercise, **not a medical diagnostic tool**.
- CIFAR-10 accuracy reflects the reported test run, not a guarantee of performance.
- Airline RMSE depends on the test period and model setup.
- The selected sentiment model is binary and does not natively classify neutral sentiment.
- Include screenshots of your own notebook outputs when publishing the repository.

---

<div align="center">

### ⭐ Thanks for visiting!

**Explore • Learn • Build**

<img src="https://img.shields.io/badge/Keep-Learning-8A2BE2?style=for-the-badge" alt="Keep learning" />

</div>
