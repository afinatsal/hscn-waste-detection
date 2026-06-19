# HSCN Waste Classification

Implementasi **Hierarchical Sibling Classification Network (HSCN)** untuk klasifikasi sampah hierarkis,
diadaptasi dari paper:

> Shin et al., *"Hierarchical Multi-Label Object Detection Framework for Remote Sensing Images"*,
> *Remote Sensing*, 2020.

---

## Struktur Hirarki Label

```
Waste (root)
├── Organic
│   └── Food_Waste                          ← LEAF (tidak ada L3)
├── Recyclable
│   ├── Plastic → [Plastic_Bottle, Plastic_Cup, Plastic_Bag_Film]
│   ├── Metal   → [Metal_Can]
│   ├── Paper   → [Paper_Sheet]
│   ├── Cardboard → [Cardboard_Box]
│   └── Glass   → [Glass_Bottle]
└── Hazardous
    ├── Battery                             ← LEAF (tidak ada L3)
    └── E_Waste                             ← LEAF (tidak ada L3)
```

---

## Arsitektur HSCN

Sesuai paper, HSCN menggunakan **classifier terpisah untuk setiap sibling-set** di setiap level:

```
Input Image (224×224)
        │
   [Backbone]          ← ResNet50/101, EfficientNet-B3, ConvNeXt-Small, dst.
        │
   [GAP + Flatten]     ← Feature vector bersama (shared)
        │
   ┌────┴─────────────────────────────────────────────────────┐
   │                                                          │
[L1 Classifier]                                               │
  SoftMax({Organic, Recyclable, Hazardous})                   │
   │                                                          │
   ├── [L2 Clf: Organic]                                      │
   │     SoftMax({Food_Waste})                                │
   │                                                          │
   ├── [L2 Clf: Recyclable]                                   │
   │     SoftMax({Plastic, Metal, Paper, Cardboard, Glass})   │
   │                                                          │
   └── [L2 Clf: Hazardous]                                    │
         SoftMax({Battery, E_Waste})                          │
                                                              │
[L3 Classifiers] (hanya untuk Recyclable)                    │
   ├── [L3 Clf: Plastic]   SoftMax({Bottle, Cup, Bag_Film})  │
   ├── [L3 Clf: Metal]     SoftMax({Can})                    │
   ├── [L3 Clf: Paper]     SoftMax({Paper_Sheet})            │
   ├── [L3 Clf: Cardboard] SoftMax({Box})                    │
   └── [L3 Clf: Glass]     SoftMax({Bottle})                 │
```

**Keunggulan desain ini:**
- Menghindari kompetisi SoftMax antar kelas yang tidak se-level
- Mendukung *partial annotation* (sampel hanya perlu anotasi sampai level tertentu)
- Loss L2/L3 hanya dihitung untuk sampel yang memiliki label pada level tersebut

---

## Struktur Dataset yang Diharapkan

```
dataset_hscn/
├── train/
│   ├── image/
│   │   ├── train_0001.jpg
│   │   └── ...
│   └── labels.json
├── valid/
│   ├── image/
│   └── labels.json
└── test/
    ├── image/
    └── labels.json
```

Format `labels.json`:
```json
[
  {"image": "train_0001.jpg", "L1": "Recyclable", "L2": "Plastic", "L3": "Plastic_Bottle"},
  {"image": "train_0002.jpg", "L1": "Organic",    "L2": "Food_Waste"},
  {"image": "train_0003.jpg", "L1": "Hazardous",  "L2": "E_Waste"}
]
```

---

## Instalasi

```bash
pip install -r requirements.txt
```

---

## Training

```bash
# Training dengan konfigurasi default (ResNet50, 80 epoch)
python train.py

# Ganti backbone
python train.py --backbone resnet101
python train.py --backbone efficientnet_b3

# Konfigurasi lengkap
python train.py \
  --data_dir dataset_hscn \
  --backbone resnet50 \
  --epochs 80 \
  --batch_size 32 \
  --lr 1e-3 \
  --backbone_lr 1e-4 \
  --scheduler cosine \
  --lambda_l1 1.0 \
  --lambda_l2 1.0 \
  --lambda_l3 0.5 \
  --freeze_backbone_epochs 5 \
  --run_name hscn_waste_v1

# Lanjut dari checkpoint
python train.py --resume checkpoints/hscn_waste_best.pth
```

### Rekomendasi Hyperparameter

| Setting              | Nilai       | Keterangan                           |
|----------------------|-------------|--------------------------------------|
| backbone             | resnet50    | Keseimbangan kecepatan & akurasi     |
| epochs               | 80          | Dengan cosine annealing              |
| batch_size           | 32          | Sesuaikan dengan VRAM GPU            |
| lr (heads)           | 1e-3        | Adam/AdamW untuk classifier heads   |
| backbone_lr          | 1e-4        | 10x lebih kecil untuk fine-tuning   |
| lambda_l1/l2         | 1.0         | Loss L1 dan L2 setara                |
| lambda_l3            | 0.5         | L3 lebih rendah (partial annotation)|
| freeze_backbone      | 5 epoch     | Warmup sebelum fine-tune backbone    |
| label_smoothing      | 0.1         | Regularisasi                         |

---

## Inference

```bash
# Prediksi satu gambar
python predict.py \
  --checkpoint checkpoints/hscn_waste_best.pth \
  --image path/to/image.jpg

# Prediksi seluruh folder
python predict.py \
  --checkpoint checkpoints/hscn_waste_best.pth \
  --image_dir path/to/folder \
  --output results.json

# Evaluasi pada test set
python predict.py \
  --checkpoint checkpoints/hscn_waste_best.pth \
  --eval_test \
  --data_dir dataset_hscn
```

---

## Struktur File

```
hscn_waste/
├── hierarchy.py     # Definisi struktur hirarki label
├── dataset.py       # WasteHSCNDataset + DataLoader
├── model.py         # Arsitektur HSCN
├── loss.py          # Hierarchical Sibling Cross-Entropy Loss
├── metrics.py       # Evaluasi per-level accuracy
├── train.py         # Script training utama
├── predict.py       # Script inference
└── requirements.txt
```

---

## Metrik Evaluasi

| Metrik              | Keterangan                                               |
|---------------------|----------------------------------------------------------|
| acc_l1              | Akurasi klasifikasi L1 (Organic/Recyclable/Hazardous)    |
| acc_l2              | Akurasi L2 (pada sampel yang punya label L2)             |
| acc_l3              | Akurasi L3 (pada sampel yang punya label L3)             |
| acc_mean            | Rata-rata acc_l1 + acc_l2 + acc_l3                       |
| acc_hier_l1l2       | Akurasi hierarkis: benar di L1 DAN L2                    |
| acc_hier_all        | Akurasi hierarkis: benar di L1, L2, DAN L3               |

---

## Referensi

Su-Jin Shin, Seyeob Kim, Youngjung Kim, Sungho Kim.
*"Hierarchical Multi-Label Object Detection Framework for Remote Sensing Images"*.
Remote Sensing, 12(17), 2734, 2020.
https://doi.org/10.3390/rs12172734
