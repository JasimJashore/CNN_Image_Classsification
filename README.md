# CNN Image Classification with PyTorch

Geometric Shapes Classification (Circle, Square, Triangle)

Project Overview

This project implements a complete Convolutional Neural Network (CNN) image classification pipeline using PyTorch, trained on a standard geometric shapes dataset and tested on real-world images captured using a smartphone.

Dataset:
Geometric Shapes

* Training Classes:

  * Circle
  * Square
  * Triangle

* Standard Dataset:
  A geometric shapes dataset (drawn/digital shapes)

* **Custom Dataset (Phone Task):**

  * 17 images drawn by hand on paper
  * Shapes photographed using a smartphone
  * Plain background to minimize noise
  * Images resized and processed to match training data format

Repository Structure
```
├── dataset/
│   ├── custom_images/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│
├── model/
│   └── 210136.pth
│
├── 210136.ipynb
├── README.md
```
Folder Description

* **dataset/** → Contains the 10 custom smartphone images
* **model/** → Saved trained model state dictionary (`.pth`)
* **210136.ipynb** → Google Colab notebook (fully automated)
* **README.md** → Project documentation and results

---

Data Preprocessing

All images undergo the **same preprocessing pipeline** to ensure consistency between training and real-world testing.

### Transform Pipeline

* Resize to fixed input size
* Convert to Tensor
* Normalize using dataset mean and standard deviation

```python
transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
```

✔ Custom phone images are converted to the **same format, size, and normalization** as the training dataset.

CNN Model Architecture

The CNN is implemented from scratch using PyTorch and inherits from `nn.Module`.

Architecture Summary

Convolutional Layers:
Feature extraction using `nn.Conv2d`
Activation:*

  * `ReLU`
* **Pooling:**

  * `MaxPool2d`
* **Fully Connected Layers:**

  * Classification into 3 classes

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

Training Details

* **Loss Function:** `CrossEntropyLoss`
* **Optimizer:** `Adam`
* **Batch Size:** 64
* **Epochs:** 10

### Training Loop

* Forward pass
* Loss computation
* Backpropagation
* Weight updates
* Accuracy tracking per epoch

The trained model is saved using:

```python
torch.save(model.state_dict(), "190110.pth")
```
Training Results

Loss vs Epochs

* Rapid decrease in training loss
* Loss converges close to **0.0**, indicating strong learning

Accuracy vs Epochs

* Training accuracy increases from ~75% to **~100%**
* Stable convergence after a few epochs

No signs of instability during training

Confusion Matrix (Standard Test Set)

A confusion matrix was generated on the **standard test dataset**.

**Observations:**

* Near-perfect classification across all classes
* Only **2 misclassifications** observed
* Strong generalization across shapes

Real-World Testing (Custom Images)

The trained model was tested on **10 custom smartphone images**.

### Prediction Gallery Output

Each image displays:

```
Predicted Class (Confidence %)
```

**Examples:**

* Triangle (100.0%)
* Circle (99.9%)
* Square (97.9%)

Model successfully classifies real-world images with **very high confidence**


Visual Error Analysis

* 3 incorrectly classified samples from the test set are displayed
* Each shows:

  * True label
  * Predicted label

This analysis helps understand edge cases and confirms overall robustness.

Automation & Reproducibility

The notebook is **fully automated** and follows assignment constraints:

✔ Uses `git clone` to download custom images
✔ Automatically loads standard dataset via `torchvision`
✔ No manual file uploads required
✔ Can be run using **“Run All”** without path issues


How to Run (Google Colab)

1. Open the Colab link
2. Click **Runtime → Run All**
3. The notebook will:

   * Clone the GitHub repo
   * Load datasets
   * Train or load the model
   * Run predictions on custom images
   * Display all required visual outputs

## Results
<img width="623" height="452" alt="Screenshot 2025-12-25 165337" src="https://github.com/user-attachments/assets/d39a9c57-5f51-4fa9-a311-273290e231a6" />
<img width="622" height="478" alt="Screenshot 2025-12-25 165344" src="https://github.com/user-attachments/assets/f54e38dd-2813-4ee9-a6b1-77ed3d48f4ea" />
<img width="686" height="516" alt="Screenshot 2025-12-25 165238" src="https://github.com/user-attachments/assets/08f44042-94ed-482c-8595-d6deb70362dc" />
<img width="712" height="548" alt="Screenshot 2025-12-25 165257" src="https://github.com/user-attachments/assets/bb95d139-1508-4b98-a639-5e9a5ae771d2" />
<img width="660" height="544" alt="Screenshot 2025-12-25 165303" src="https://github.com/user-attachments/assets/cd2b0f22-68dd-4656-8d18-01aea4229788" />

Submission Links:

* **GitHub Repository:**
  [https://github.com/JasimJashore/CNN_Image_Classsification]

* **Google Colab Notebook:**
  [https://colab.research.google.com/drive/1dS_N-xPHsA_Vuq2yVYRxsZO1A0fNJz3U#scrollTo=ByTykUNPK5Kz]

## 🏁 Conclusion

This project demonstrates a **complete deep learning image classification workflow**, successfully transferring knowledge from a standard dataset to real-world images. The CNN achieves **near-perfect accuracy**, strong generalization, and meets **all assignment requirements**.

