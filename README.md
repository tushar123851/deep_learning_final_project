<div align="center">

# 🚀 Deep Learning, Computer Vision & NLP Portfolio

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=22&pause=900&color=36BCF7&center=true&vCenter=true&width=900&lines=Artificial+Neural+Networks;OpenCV+Computer+Vision;CNN+Image+Classification;RNN+Time-Series+Forecasting;Transformer+Sentiment+Analysis" alt="Animated project topics" />

<p>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow Keras" />
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Transformers" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter Notebook" />
</p>

**A hands-on collection of five practical projects covering deep learning, image processing, time-series forecasting, and natural language processing.**

[Project Overview](#-project-overview) • [Results](#-results-at-a-glance) • [Technologies](#️-technology-stack) • [Setup](#️-setup-and-run)

</div>

---

## 📚 Table of Contents

- [✨ Project Overview](#-project-overview)
- [🎯 Project Objectives](#-project-objectives)
- [📊 Results at a Glance](#-results-at-a-glance)
- [🧠 Task 1 — ANN Diabetes Prediction](#-task-1--ann-diabetes-prediction)
- [🖼️ Task 2 — OpenCV Image Processing](#️-task-2--opencv-image-processing)
- [🌈 Task 3 — CNN CIFAR-10 Classification](#-task-3--cnn-cifar-10-classification)
- [📈 Task 4 — SimpleRNN Airline Forecasting](#-task-4--simplernn-airline-forecasting)
- [💬 Task 5 — Transformer Sentiment Analysis](#-task-5--transformer-sentiment-analysis)
- [🧰 Technology Stack](#️-technology-stack)
- [⚙️ Setup and Run](#️-setup-and-run)
- [📂 Suggested Project Structure](#-suggested-project-structure)
- [🧪 Evaluation Approach](#-evaluation-approach)
- [🚀 Potential Improvements](#-potential-improvements)
- [🎓 Key Learnings](#-key-learnings)
- [⚠️ Limitations and Responsible Use](#️-limitations-and-responsible-use)
- [👨‍💻 About Me](#-about-me)

---

## ✨ Project Overview

This repository documents five practical machine-learning and deep-learning exercises. Each task follows a common workflow: understand the problem, prepare data, apply a suitable technique, evaluate the output, and interpret the results.

| Task | Project | Main technique | Problem type | Main output |
|---|---|---|---|---|
| 1 | Diabetes outcome prediction | Artificial Neural Network (ANN) | Binary classification | Accuracy, precision, recall, F1-score, confusion matrix |
| 2 | Image processing pipeline | OpenCV | Computer vision | Resized images, grayscale, blur, edges, contours |
| 3 | CIFAR-10 image classification | Convolutional Neural Network (CNN) | Multi-class classification | Predicted image class and classification report |
| 4 | Airline passenger forecasting | Simple Recurrent Neural Network (SimpleRNN) | Time-series regression | Predicted passenger counts and RMSE |
| 5 | Text sentiment classification | Pre-trained Transformer | NLP classification | Positive/negative labels and confidence scores |

---

## 🎯 Project Objectives

- Practice data loading, inspection, preprocessing, and transformation.
- Understand how ANN, CNN, and recurrent neural networks can be applied to different data types.
- Use OpenCV to process images and identify edges and contours.
- Apply a pre-trained Transformer model to text classification.
- Evaluate model outputs with appropriate metrics instead of relying on a single score.
- Build a well-documented, reproducible learning portfolio using Python and Jupyter Notebook.

---

## 📊 Results at a Glance

| Task | Reported result | Interpretation |
|---|---:|---|
| ANN diabetes prediction | **71.43% accuracy** | Baseline classification result; recall should be improved and checked carefully. |
| OpenCV image processing | **5 images processed** | Five visual stages are generated for each image by the notebook pipeline. |
| CNN CIFAR-10 classification | **60.25% accuracy** | The model learned useful visual patterns across ten classes, with uneven performance by class. |
| SimpleRNN airline forecasting | **40.57 passengers RMSE** | Average prediction error is summarized in passenger-count units by RMSE. |
| Transformer sentiment analysis | **25 example sentences** | Demonstrates binary sentiment predictions with confidence scores. |

> Metrics shown here are from the runs documented in this README. They may change if the data split, random seed, preprocessing, model architecture, or training settings change.

---

## 🧠 Task 1 — ANN Diabetes Prediction

**Objective:** Use the Pima Indians Diabetes dataset to predict the target outcome (`Outcome`) from patient-related numeric features. This is a supervised binary-classification exercise.

### Workflow

1. Load the CSV and assign the expected column names because the dataset file may not contain a header row.
2. Inspect the data and identify implausible zero values in selected medical measurements.
3. Replace zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` with the median of the non-zero values for that column.
4. Separate input features (`X`) from the target (`y`).
5. Create an 80/20 train-test split with stratification to preserve the class proportions.
6. Fit `StandardScaler` on the training features and apply the same transformation to the test features.
7. Train a Sequential ANN with two hidden layers and a sigmoid output.
8. Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.

### Network architecture

```text
Input features
     ↓
Dense(16, activation="relu")
     ↓
Dense(8, activation="relu")
     ↓
Dense(1, activation="sigmoid")
     ↓
Predicted probability for the positive class
```

### 📊 Test results

| Metric | Result |
|---|---:|
| Test loss | **0.5167** |
| Test accuracy | **71.43%** |
| Precision | **60.87%** |
| Recall | **51.85%** |
| F1-score | **56.00%** |

**Confusion matrix** — rows represent actual classes and columns represent predicted classes:

```text
                 Predicted 0   Predicted 1
Actual 0              82            18
Actual 1              26            28
```

- **82 true negatives:** class 0 correctly predicted as class 0.
- **18 false positives:** class 0 predicted as class 1.
- **26 false negatives:** class 1 predicted as class 0.
- **28 true positives:** class 1 correctly predicted as class 1.

### Interpretation

The model achieved 71.43% accuracy on this test split. Recall of 51.85% means it identified around half of the actual positive cases in this run. In a healthcare-related setting, false negatives can be particularly important, so accuracy alone is not enough. This notebook is for learning and is not suitable for clinical decisions.

---

## 🖼️ Task 2 — OpenCV Image Processing

**Objective:** Apply a repeatable image-processing pipeline to five sample images and visualize intermediate outputs to understand what each operation does.

### Processing pipeline

```text
Input images (5)
      ↓
Load image with OpenCV
      ↓
Resize to 128 × 128
      ↓
Convert to grayscale
      ↓
Gaussian blur (noise reduction)
      ↓
Canny edge detection
      ↓
Find contours and draw bounding rectangles
      ↓
Display a 5 × 5 result grid and save outputs
```

### OpenCV functions

| Function | Purpose |
|---|---|
| `cv2.imread()` | Read an image from disk. |
| `cv2.resize()` | Resize an image to a consistent width and height. |
| `cv2.cvtColor()` | Convert an image between color spaces, such as BGR and grayscale. |
| `cv2.GaussianBlur()` | Smooth an image and reduce small-scale noise before edge detection. |
| `cv2.Canny()` | Detect strong intensity changes that may represent edges. |
| `cv2.findContours()` | Find contours in a binary image. |
| `cv2.boundingRect()` | Calculate a rectangular bounding box around a contour. |
| `cv2.rectangle()` | Draw a rectangle around a detected contour. |
| `cv2.imwrite()` | Save an image result to disk. |

### Input image folder

Add five images to an `images/` folder. These are example filenames; use filenames that match your own images and notebook code.

```text
images/
├── apple.jpg
├── dog.jpg
├── car.jpg
├── flower.jpg
└── ball.jpg
```

### Expected output

The notebook can display five stages for each image: **Original**, **Resized**, **Grayscale**, **Canny Edges**, and **Contours**. With five images, these stages form a 5 × 5 visualization grid. Processed images can also be saved in `output_images/`.

**Portfolio tip:** Add a screenshot of the actual grid generated by your notebook to the repository, then embed it here with a relative path, for example `![OpenCV processing results](results/opencv_output_grid.png)`. The screenshot is not embedded yet because the actual image output was not supplied with this README.

---

## 🌈 Task 3 — CNN CIFAR-10 Classification

**Objective:** Classify small color images into ten categories from the CIFAR-10 dataset.

### CIFAR-10 classes

`airplane` · `automobile` · `bird` · `cat` · `deer` · `dog` · `frog` · `horse` · `ship` · `truck`

### Example CNN architecture

```text
Input image (32 × 32 × 3)
          ↓
Convolution + ReLU
          ↓
Max Pooling
          ↓
Convolution + ReLU
          ↓
Max Pooling
          ↓
Flatten
          ↓
Dense(64, ReLU)
          ↓
Dense(10, Softmax)
          ↓
Class probabilities
```

The convolutional layers learn local visual patterns, pooling reduces spatial dimensions, and the final softmax layer returns scores across the ten classes.

### 📊 Test results

| Metric | Result |
|---|---:|
| Test loss | **1.1557** |
| Test accuracy | **60.25%** |
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
- Automobile had the highest reported class F1-score (**0.75**).
- Ship had the highest recall (**0.90**), meaning most actual ship images in this test set were detected.
- Cat had the lowest reported class F1-score (**0.41**), indicating that this class was more challenging for the current model.
- Per-class metrics provide more detail than accuracy alone and help identify which categories may need additional training or model improvements.

---

## 📈 Task 4 — SimpleRNN Airline Forecasting

**Objective:** Forecast monthly airline passenger counts from historical time-series data using sequential input windows.

### Workflow

1. Load the airline-passengers dataset and inspect the time order.
2. Scale passenger counts using `MinMaxScaler` to help neural-network training.
3. Create input sequences from the previous 12 months to predict the next value.
4. Split training and test data chronologically; do not shuffle time-series observations.
5. Train a `SimpleRNN(50)` layer followed by `Dense(1)` for numeric prediction.
6. Convert predictions back to the original passenger-count scale with the scaler's inverse transformation.
7. Compare predictions against actual test values using a plot and RMSE.

### Model flow

```text
Previous 12 monthly observations → SimpleRNN(50) → Dense(1) → Next-month estimate
```

### 📊 Test result

| Metric | Result |
|---|---:|
| Test RMSE | **40.57 passengers** |

RMSE (root mean squared error) measures the typical magnitude of prediction error in the same units as the target, while giving larger errors more weight. Interpret it with the passenger-count range and an actual-versus-predicted plot. Results depend on the selected test period and preprocessing choices.

---

## 💬 Task 5 — Transformer Sentiment Analysis

**Objective:** Use a pre-trained Transformer model to classify text as positive or negative and inspect its confidence score.

**Model:** `distilbert-base-uncased-finetuned-sst-2-english`  
**Library:** Hugging Face Transformers

The notebook evaluates 25 example sentences: 10 positive, 10 negative, and 5 neutral or ambiguous examples. The selected model is trained for binary sentiment classification, so it does not provide a dedicated neutral class.

### Sample predictions

| Sentence | Reference label | Predicted label | Confidence |
|---|---|---|---:|
| The meeting starts at ten in the morning. | NEU | POSITIVE | 0.967966 |
| I hate waiting for a long time. | NEG | NEGATIVE | 0.986813 |
| The package arrived yesterday. | NEU | POSITIVE | 0.989624 |
| The movie was okay, but nothing special. | NEU | NEGATIVE | 0.993532 |
| My laptop is slow and keeps crashing. | NEG | NEGATIVE | 0.996090 |

### Observations and interpretation

- The clearly negative examples shown above were classified as **NEGATIVE**.
- Neutral or factual sentences are still assigned either POSITIVE or NEGATIVE by this binary model.
- Confidence indicates how strongly the model favors its selected label; it does **not** guarantee that the label is correct.
- The high-confidence POSITIVE prediction for a neutral package-delivery sentence illustrates why predictions should be interpreted in context.
- A production application may need a neutral class, domain-specific validation, and human review for uncertain or high-impact cases.

---

## 🧰 Technology Stack

| Technology / library | Role in this project |
|---|---|
| Python | Main programming language |
| NumPy and Pandas | Numerical operations and tabular data handling |
| Matplotlib | Visualizations and prediction plots |
| OpenCV | Image loading and computer-vision operations |
| Scikit-learn | Data splitting, scaling, and evaluation metrics |
| TensorFlow / Keras | ANN, CNN, and SimpleRNN models |
| PyTorch | Backend used by the Transformer pipeline where configured |
| Hugging Face Transformers | Loading and running the pre-trained sentiment model |
| Jupyter Notebook / VS Code | Interactive development and experimentation |

## ⚙️ Setup and Run

### 1. Clone your repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

Replace the placeholders with your actual GitHub repository URL and folder name.

### 2. Create and activate a virtual environment (recommended)

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If you are using Jupyter Notebook, install packages from a notebook cell with `%pip install ...` rather than typing the notebook-only `!pip` syntax directly into PowerShell.

### 3. Install dependencies

```bash
python -m pip install numpy pandas matplotlib scikit-learn opencv-python tensorflow torch transformers jupyter
```

> Deep-learning packages can have Python-version and hardware compatibility requirements. If installation fails, check the package's supported Python version and use the same environment for both installation and your notebook kernel.

### 4. Add the required data

- **Task 1:** Pima Indians Diabetes CSV with the expected eight feature columns and the `Outcome` target. If the file has no header row, read it with `header=None` and assign column names in code.
- **Task 2:** Five image files inside `images/`.
- **Task 3:** CIFAR-10 can be downloaded by Keras when the dataset loader runs for the first time; internet access may be required.
- **Task 4:** Airline passenger time-series CSV with date/month and passenger-count columns matching your notebook.
- **Task 5:** Internet access may be required the first time the Hugging Face model is downloaded.

### 5. Run the notebooks

1. Open the project folder in VS Code or Jupyter Notebook.
2. Select the Python kernel associated with the environment where you installed the packages.
3. Open the notebook for the task you want to run.
4. Update file paths if your dataset or image names differ from the examples.
5. Run cells in order and review the printed metrics and plots.

---

## 🧪 Evaluation Approach

| Metric | Common use | What to keep in mind |
|---|---|---|
| Accuracy | Fraction of predictions that are correct | Can hide poor performance on a minority class. |
| Precision | Fraction of predicted positives that are true positives | Useful when false-positive predictions matter. |
| Recall | Fraction of actual positives that are detected | Important when missing positive cases is costly. |
| F1-score | Harmonic mean of precision and recall | Balances precision and recall in one measure. |
| Confusion matrix | Counts correct and incorrect class predictions | Helps identify the types of errors being made. |
| RMSE | Typical scale of regression error | Expressed in the target's units and sensitive to large errors. |
| Confidence score | Strength of a model's selected classification output | Not the same as a guarantee of correctness or calibrated probability. |

## 🚀 Potential Improvements

- **ANN:** Compare class weighting, threshold tuning, cross-validation, and additional evaluation metrics. For a healthcare use case, investigate recall and false-negative rates carefully.
- **OpenCV:** Save a complete before-and-after grid, compare edge thresholds, and test on images with different lighting and backgrounds.
- **CNN:** Use data augmentation, dropout or batch normalization where appropriate, learning-rate tuning, and a confusion matrix to study class-level errors.
- **SimpleRNN:** Compare against a naive seasonal baseline and other forecasting models; plot actual and predicted passenger counts on the same timeline.
- **Transformer:** Include more examples from the intended domain, evaluate with a labeled test set, and use a model or pipeline that can handle neutral sentiment if that class matters.
- **Reproducibility:** Record random seeds, package versions, data sources, and the final model configuration so results can be compared fairly.
- **Documentation:** Add notebook output screenshots, sample predictions, and a short explanation of each result after running the code in your own environment.

## 🎓 Key Learnings

- Preparing tabular data and training an ANN for binary classification.
- Handling missing or implausible values and scaling numerical features.
- Understanding precision, recall, F1-score, accuracy, and confusion matrices.
- Applying resizing, grayscale conversion, smoothing, edge detection, and contour detection with OpenCV.
- Building a CNN for multi-class image classification.
- Converting time-series data into sequences for recurrent neural-network training.
- Evaluating numerical forecasts using RMSE.
- Applying a pre-trained Transformer to sentiment analysis and recognizing the limitations of binary labels.
- Organizing technical work into a documented, reproducible portfolio.

## ⚠️ Limitations and Responsible Use

- Reported metrics describe the specific runs documented here; results may change with different random seeds, data splits, package versions, preprocessing, and training settings.
- The diabetes prediction exercise is educational only and **must not be used for medical diagnosis, screening, or treatment decisions**.
- CIFAR-10 performance is limited to the documented test run and may vary by class.
- Forecasting performance depends on the selected time period, sequence length, and model setup.
- The sentiment model is a binary classifier and may produce misleading labels for neutral, sarcastic, or domain-specific text.
- Keep datasets and model artifacts organized, respect dataset licenses and privacy, and avoid publishing sensitive information.
- Do not claim that a screenshot, result file, or notebook is included unless it is actually committed to the repository.

---

<div align="center">

## 👨‍💻 About Me

### **Tushar Vala**

**Data Analyst | AI/ML Developer | Data Scientist**

Passionate about data analysis, machine learning, deep learning, computer vision, and natural language processing. I enjoy turning data and practical problems into structured experiments, useful insights, and well-documented projects.

**Explore • Learn • Build • Improve**

<img src="https://img.shields.io/badge/Thank%20You-For%20Visiting-8A2BE2?style=for-the-badge" alt="Thank you for visiting" />

</div>
