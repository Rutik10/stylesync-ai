# 🎥 StyleSync AI — Video-to-Catalog Product Tagging + Vibe Detection

StyleSync AI is a backend ML pipeline built for the **Flickd AI Hackathon**. It detects fashion items in short-form creator videos, matches them to a product catalog using CLIP embeddings, and classifies the video into 1–3 fashion "vibes" from a predefined taxonomy.

---

## 🧠 What It Does

- Detects clothing and accessories using **YOLOv8**
- Crops detected objects from frames
- Matches objects to a **Shopify catalog** using **CLIP + FAISS**
- Classifies fashion "vibes" using **NLP** on captions
- Outputs a structured **JSON** per video

---

## 📁 Project Structure

```
stylesync-ai/
├── frames/ # Extracted frames for each video
├── cropped_objects/ # Cropped detected fashion objects per frame
├── models/ # Cached catalog embeddings (catalog_embeddings.npy)
├── data/
│ ├── product_data.xlsx # Catalog metadata: id, type, tags
│ ├── images.csv # Product image URLs (id, image_url)
│ ├── vibeslist.json # List of predefined "vibe" labels
│ └── outputs_final/ # Final prediction JSONs per video
├── outputs/ # Object detection results (YOLO JSON per video)
├── notebooks/
│ └── main_pipeline.ipynb # End-to-end notebook pipeline
├── requirements.txt # Python dependencies
└── README.md # You are here
```

## ▶️ Run the Full Pipeline

Run the following notebooks in order:

### 1. [`extract_frames.ipynb`](notebooks/extract_frames.ipynb)
- Extracts frames from input videos
- Saves frames to: `frames/<video_id>/`

### 2. [`detect_objects.ipynb`](notebooks/detect_objects.ipynb)
- Uses YOLOv8 to detect clothing and accessories in frames
- Outputs detection results (with bounding boxes and class labels) to: `outputs/<video_id>.json`

### 3. [`flickd_product_recommendation_pipeline.ipynb`](notebooks/flickd_product_recommendation_pipeline.ipynb)
This notebook performs the following steps:

- 🖼️ **Crop detected items** from frames using YOLO results  
- 🔗 **Embed each crop** using CLIP (OpenAI or HuggingFace)  
- 🛍️ **Match crops to catalog items** using cosine similarity with FAISS  
- 🧠 **Predict vibes** using mean embedding vectors and NLP (e.g., DistilBERT)  
- 📄 **Save final results** as structured JSON in: `data/outputs_final/<video_id>.json`

## 📦 Example Output

Each video generates a structured JSON like this:

```json
{
  "video_id": "2025-06-02_11-31-19_UTC",
  "vibes": ["Clean Girl", "Boho", "Cottagecore"],
  "products": [
    {
      "type": "Dress",
      "color": "Colour: White, Fabric: Cotton",
      "image_url": "https://cdn.shopify.com/...",
      "matched_product_id": 1234,
      "match_type": "similar",
      "confidence": 0.84
    }
  ]
}
```

## 🧠 How It Works

### 🕵️‍♂️ Object Detection
YOLOv8 detects fashion items in video frames and outputs:
- Class labels (e.g., dress, bag, top)
- Bounding boxes (x, y, width, height)
- Confidence scores per detection

### 🧬 Embedding
CLIP (Contrastive Language-Image Pretraining) encodes:
- Each cropped object from the video frame
- Each product image in the catalog  
It generates high-dimensional vector embeddings for visual similarity comparison.

### 🔗 Matching
Embeddings from the detected items are:
- Compared with catalog embeddings using **cosine similarity**
- Accelerated using **FAISS** (Facebook AI Similarity Search)
- Labeled as `exact`, `similar`, or `no match` based on a similarity threshold

### 🎯 Vibe Classification
Captions and hashtags (or optional audio transcript) are:
- Analyzed using transformer-based NLP (e.g., DistilBERT)
- Classified into 1–3 fashion vibes from the predefined taxonomy:  
  `["Coquette", "Clean Girl", "Cottagecore", "Streetcore", "Y2K", "Boho", "Party Glam"]`

## 📹 Loom Demo Video

🎥 Watch the demo of the complete pipeline in action:

## 📽 Demo Video

Watch the full demo on Loom:  
👉 [Click here to watch](https://www.loom.com/share/fea9f75bac024ed4a5fe2bd40860dd47?sid=f0261a7a-5dba-411c-8c6b-89748ab65ba0)

---

## 🛠️ Requirements

Make sure you have **Python 3.8+** installed.

### Required Python Packages:

- `torch`
- `torchvision`
- `transformers`
- `pandas`
- `numpy`
- `opencv-python`
- `scikit-learn`
- `tqdm`
- `ultralytics`  _(for YOLOv8)_

### Install All Dependencies

```bash
pip install -r requirements.txt
```
