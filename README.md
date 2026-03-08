# 🎨 ChromaSnap — Color Palette Extractor

A machine learning powered web app that extracts the dominant color palette from any image using **K-Means Clustering**. Built with Python, Scikit-learn, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![OpenCV](https://img.shields.io/badge/OpenCV-Vision-green?style=flat-square&logo=opencv)

---

## 📌 What is ChromaSnap?

ChromaSnap takes any image you upload and automatically identifies its **most dominant colors** using an unsupervised machine learning algorithm called **K-Means Clustering**. It shows you each color's:

- **Hex code** (e.g. `#FF5733`)
- **RGB values** (e.g. `R=255, G=87, B=51`)
- **Coverage percentage** — how much of the image that color represents
- **Visual palette strip** — a proportional bar showing all colors at a glance

---

## 🖥️ Demo

> Upload any PNG, JPG, or WEBP image → Get your color palette instantly.

---

## 🧠 How It Works — The ML Behind It

### Step 1 — Image Loading
The image is loaded using **OpenCV** and converted from BGR (OpenCV default) to **RGB** format.

```python
image_bgr = cv2.imread('image.png', cv2.IMREAD_COLOR)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
```

### Step 2 — Pixel Reshaping
An image is a 3D array of shape `(height, width, 3)`. We flatten it into a 2D array where every row is one pixel with 3 values (R, G, B).

```python
pixels = image_rgb.reshape(-1, 3)
# e.g. (1920, 1080, 3) → (2073600, 3)
```

### Step 3 — K-Means Clustering
We run **K-Means** on the pixel array. The algorithm groups similar colors together into `k` clusters. The center of each cluster becomes a **dominant color**.

```python
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(pixels)
dominant_colors = kmeans.cluster_centers_.astype(int)
```

> **Why K-Means?** It naturally groups pixels by color similarity in 3D RGB space — no labels needed. It's an unsupervised learning approach perfect for this task.

### Step 4 — Calculate Coverage
We count how many pixels belong to each cluster to find each color's percentage coverage.

```python
labels = kmeans.labels_
percentage = (np.sum(labels == i) / len(labels)) * 100
```

### Step 5 — Display Results
Colors are sorted by dominance and displayed as swatches with hex codes, RGB values, and percentages.

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed.

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/aviksarkar0204-stack/chromasnap.git
cd chromasnap

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## 📦 Requirements

```
streamlit
opencv-python
numpy
scikit-learn
Pillow
pandas
matplotlib
```

Or install all at once:
```bash
pip install streamlit opencv-python numpy scikit-learn Pillow pandas matplotlib
```

---

## 📁 Project Structure

```
chromasnap/
│
├── app.py              # Main Streamlit web app
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## ✨ Features

- 📤 **Drag & drop image upload** — PNG, JPG, JPEG, WEBP supported
- 🎚️ **Adjustable palette size** — extract 2 to 10 colors using a slider
- 🌈 **Palette strip** — proportional color bar showing visual distribution
- 🟦 **Color cards** — each dominant color shown with hex, RGB, and % coverage
- 📋 **Raw data table** — expandable table with all color data, exportable
- ⚡ **Fast processing** — downsamples large images to 50,000 pixels for speed
- 🌑 **Dark UI** — clean, modern dark theme

---

## 🔬 Technologies Used

| Tool | Purpose |
|------|---------|
| `Python` | Core programming language |
| `OpenCV` | Image loading and color space conversion |
| `NumPy` | Pixel array manipulation |
| `Scikit-learn` | K-Means clustering algorithm |
| `Pillow (PIL)` | Image file handling |
| `Streamlit` | Web app framework |
| `Pandas` | Data display in table format |
| `Matplotlib` | Original visualization (development phase) |

---


## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

Built by **Avik Sarkar** — [@aviksarkar0204-stack](https://github.com/aviksarkar0204-stack)  
Feel free to connect or raise an issue if you find a bug!
