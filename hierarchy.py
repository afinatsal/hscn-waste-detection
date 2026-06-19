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
        Plastic      → [Plastic_Bottle, Plastic_Cup, Plastic_Bag_Film]
        Metal        → [Metal_Can]
        Paper        → [Paper_Sheet]
        Cardboard    → [Cardboard_Box]
        Glass        → [Glass_Bottle]

Catatan:
    - Organic dan Hazardous hanya memiliki L1 dan L2 (tidak ada L3).
    - HSCN melatih classifier terpisah untuk setiap "sibling set".
    - Loss L3 hanya dihitung untuk sampel yang memiliki anotasi L3.
"""

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
L3_SIBLINGS = {
    "Plastic":   ["Plastic_Bottle", "Plastic_Cup", "Plastic_Bag_Film"],
    "Metal":     ["Metal_Can"],
    "Paper":     ["Paper_Sheet"],
    "Cardboard": ["Cardboard_Box"],
    "Glass":     ["Glass_Bottle"],
}
# L2 nodes yang TIDAK memiliki L3 children
L2_NO_L3 = {"Food_Waste", "Battery", "E_Waste"}

# flat list & mapping
L3_ALL    = [cls for siblings in L3_SIBLINGS.values() for cls in siblings]
L3_TO_IDX = {c: i for i, c in enumerate(L3_ALL)}

# reverse: L3 class → L2 parent
L3_PARENT = {}
for l2, children in L3_SIBLINGS.items():
    for c in children:
        L3_PARENT[c] = l2

# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_l2_sibling_indices(l1_class: str):
    """Kembalikan indeks global L2 untuk sibling set milik l1_class."""
    return [L2_TO_IDX[c] for c in L2_SIBLINGS[l1_class]]


def get_l3_sibling_indices(l2_class: str):
    """Kembalikan indeks global L3 untuk sibling set milik l2_class.
    Kembalikan list kosong jika l2_class tidak memiliki L3."""
    if l2_class not in L3_SIBLINGS:
        return []
    return [L3_TO_IDX[c] for c in L3_SIBLINGS[l2_class]]


def num_l1():  return len(L1_CLASSES)
def num_l2():  return len(L2_ALL)
def num_l3():  return len(L3_ALL)


# ─── Summary ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== HSCN Waste Hierarchy ===")
    print(f"L1 classes ({num_l1()}): {L1_CLASSES}")
    print(f"L2 classes ({num_l2()}): {L2_ALL}")
    print(f"L3 classes ({num_l3()}): {L3_ALL}")
    print()
    for l1 in L1_CLASSES:
        print(f"  [{l1}]")
        for l2 in L2_SIBLINGS[l1]:
            l3s = L3_SIBLINGS.get(l2, [])
            suffix = f"→ {l3s}" if l3s else "(leaf)"
            print(f"    └─ {l2}  {suffix}")
