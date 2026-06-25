"""
hierarchy.py
============
Mendefinisikan struktur hirarki label dataset sampah untuk HSCN.

Struktur:
    L1 (root siblings): Waste → [Organic, Recyclable, Hazardous]
    L2 (sibling sets per L1):
        Organic      → [Food_Waste]
        Recyclable   → [Plastic, Metal, Paper, Cardboard, Glass]
        Hazardous    → [Battery, E_Waste]
    L3 (sibling sets per L2, hanya untuk Recyclable):
        Plastic      → [__none__, Plastic_Bottle, Plastic_Cup, Plastic_Bag_Film]
        Metal        → [__none__, Metal_Can]
        Paper        → [__none__, Paper_Sheet]
        Cardboard    → [__none__, Cardboard_Box]
        Glass        → [__none__, Glass_Bottle]

Catatan:
    - "__none__" di L3 berarti objek hanya diklasifikasikan sampai L2,
      tidak ada sub-tipe L3 yang lebih spesifik.
    - Organic dan Hazardous hanya memiliki L1 dan L2 (tidak ada L3).
    - HSCN melatih classifier terpisah untuk setiap "sibling set".
    - Loss L3 dihitung untuk semua sampel Recyclable (termasuk yang __none__).
"""

# ─── Label khusus "tidak ada L3" ──────────────────────────────────────────────
L3_NONE_LABEL = "__none__"   # SELALU di index 0 setiap sibling-set L3

# ─── L1: root-level siblings ──────────────────────────────────────────────────
L1_CLASSES = ["Organic", "Recyclable", "Hazardous"]
L1_TO_IDX  = {c: i for i, c in enumerate(L1_CLASSES)}

# ─── L2: per-parent sibling sets ──────────────────────────────────────────────
L2_SIBLINGS = {
    "Organic":     ["Food_Waste"],
    "Recyclable":  ["Plastic", "Metal", "Paper", "Cardboard", "Glass"],
    "Hazardous":   ["Battery", "E_Waste"],
}
# flat list & mapping
L2_ALL    = [cls for siblings in L2_SIBLINGS.values() for cls in siblings]
L2_TO_IDX = {c: i for i, c in enumerate(L2_ALL)}

# reverse: L2 class → L1 parent
L2_PARENT = {}
for l1, children in L2_SIBLINGS.items():
    for c in children:
        L2_PARENT[c] = l1

# ─── L3: per-parent sibling sets (hanya Recyclable) ───────────────────────────
# PENTING: "__none__" SELALU menjadi elemen PERTAMA (index 0) di setiap
# sibling-set L3. Ini merepresentasikan "objek ini hanya sampai L2, tidak ada L3".
L3_SIBLINGS = {
    "Plastic":   [L3_NONE_LABEL, "Plastic_Bottle", "Plastic_Cup", "Plastic_Bag_Film"],
    "Metal":     [L3_NONE_LABEL, "Metal_Can"],
    "Paper":     [L3_NONE_LABEL, "Paper_Sheet"],
    "Cardboard": [L3_NONE_LABEL, "Cardboard_Box"],
    "Glass":     [L3_NONE_LABEL, "Glass_Bottle"],
}

# L2 nodes yang TIDAK memiliki L3 children (Organic & Hazardous subtypes)
L2_NO_L3 = {"Food_Waste", "Battery", "E_Waste"}

# flat list & mapping — EXCLUDE __none__ dari L3_ALL karena bukan kelas nyata
# L3_ALL hanya berisi kelas L3 yang benar-benar ada (bukan __none__)
L3_ALL    = [
    cls
    for siblings in L3_SIBLINGS.values()
    for cls in siblings
    if cls != L3_NONE_LABEL
]
L3_TO_IDX = {c: i for i, c in enumerate(L3_ALL)}

# Mapping lengkap termasuk __none__ per sibling-set (untuk loss & dataset)
# Key: l2_name, Value: dict {class_name → local_index}
L3_LOCAL_IDX = {
    l2: {cls: local_i for local_i, cls in enumerate(children)}
    for l2, children in L3_SIBLINGS.items()
}

# reverse: L3 class → L2 parent (exclude __none__)
L3_PARENT = {}
for l2, children in L3_SIBLINGS.items():
    for c in children:
        if c != L3_NONE_LABEL:
            L3_PARENT[c] = l2

# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_l2_sibling_indices(l1_class: str):
    """Kembalikan indeks global L2 untuk sibling set milik l1_class."""
    return [L2_TO_IDX[c] for c in L2_SIBLINGS[l1_class]]


def get_l3_sibling_indices(l2_class: str):
    """Kembalikan indeks global L3 untuk sibling set milik l2_class.
    Kembalikan list kosong jika l2_class tidak memiliki L3.
    CATATAN: __none__ tidak dimasukkan karena tidak ada di L3_TO_IDX."""
    if l2_class not in L3_SIBLINGS:
        return []
    return [
        L3_TO_IDX[c]
        for c in L3_SIBLINGS[l2_class]
        if c != L3_NONE_LABEL
    ]


def num_l1(): return len(L1_CLASSES)
def num_l2(): return len(L2_ALL)
def num_l3(): return len(L3_ALL)  # tidak termasuk __none__

def num_l3_with_none(l2_class: str) -> int:
    """Jumlah kelas L3 untuk l2_class TERMASUK __none__."""
    return len(L3_SIBLINGS.get(l2_class, []))


# ─── Summary ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== HSCN Waste Hierarchy ===")
    print(f"L1 classes ({num_l1()}): {L1_CLASSES}")
    print(f"L2 classes ({num_l2()}): {L2_ALL}")
    print(f"L3 classes ({num_l3()}, excl. __none__): {L3_ALL}")
    print()
    for l1 in L1_CLASSES:
        print(f"  [{l1}]")
        for l2 in L2_SIBLINGS[l1]:
            l3s = L3_SIBLINGS.get(l2, [])
            suffix = f"→ {l3s}" if l3s else "(leaf)"
            print(f"    └─ {l2}  {suffix}")
    print()
    print("L3_LOCAL_IDX (termasuk __none__ di index 0):")
    for l2, idx_map in L3_LOCAL_IDX.items():
        print(f"  {l2}: {idx_map}")