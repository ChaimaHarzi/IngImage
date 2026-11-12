import json
from pathlib import Path

nb_path = Path(r"c:\Users\MSI\OneDrive\Desktop\IGL4\SEM2\IngImage\Projet3\projet_detection_maladies.ipynb")

code_lines = [
    "# Interface Gradio pour la prediction de maladie sur feuille de tomate",
    "import numpy as np",
    "import gradio as gr",
    "from PIL import Image",
    "",
    # Selection du modele par defaut (MobileNetV2 TL si disponible)",
    "_MODEL = None",
    "if 'mobilenet' in globals():",
    "    _MODEL = mobilenet",
    "elif 'cnn_scratch' in globals():",
    "    _MODEL = cnn_scratch",
    "",
    "# Labels d'affichage",
    "_LABELS = None",
    "if 'short_names' in globals():",
    "    _LABELS = short_names",
    "elif 'le' in globals():",
    "    _LABELS = [c.replace('Tomato__', '').replace('Tomato_', '') for c in le.classes_]",
    "",
    "def _preprocess_image(pil_img):",
    "    if pil_img.mode != 'RGB':",
    "        pil_img = pil_img.convert('RGB')",
    "    pil_img = pil_img.resize(IMG_SIZE, Image.BILINEAR)",
    "    arr = np.asarray(pil_img, dtype=np.float32) / 255.0",
    "    return np.expand_dims(arr, axis=0)",
    "",
    "def predict_disease(pil_img, top_n=3):",
    "    if _MODEL is None:",
    "        return 'Aucun modele charge. Executez les cellules d entrainement.', {}",
    "    if _LABELS is None:",
    "        return 'Labels indisponibles. Verifiez le LabelEncoder.', {}",
    "    x = _preprocess_image(pil_img)",
    "    probs = _MODEL.predict(x, verbose=0)[0]",
    "    top_n = int(np.clip(top_n, 1, len(_LABELS)))",
    "    top_idx = np.argsort(probs)[::-1][:top_n]",
    "    pred_label = _LABELS[top_idx[0]]",
    "    prob_dict = {_LABELS[i]: float(probs[i]) for i in top_idx}",
    "    return pred_label, prob_dict",
    "",
    "with gr.Blocks(title='Detection Maladies Tomate') as demo:",
    "    gr.Markdown('# Detection et Classification des Maladies des Plantes')",
    "    gr.Markdown('Chargez une image de feuille de tomate et lancez la prediction.')",
    "    with gr.Row():",
    "        inp = gr.Image(type='pil', label='Image')",
    "        out_label = gr.Textbox(label='Prediction')",
    "    top_n = gr.Slider(minimum=1, maximum=10, value=3, step=1, label='Top-N classes')",
    "    out_probs = gr.Label(label='Probabilites', num_top_classes=3)",
    "    btn = gr.Button('Predire')",
    "    btn.click(fn=predict_disease, inputs=[inp, top_n], outputs=[out_label, out_probs])",
    "",
    "demo.launch(share=False)",
]

with nb_path.open("r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb.get("cells", [])
cell_index = 62  # 0-based index for cell 63
if cell_index >= len(cells):
    raise SystemExit(f"Cell index {cell_index} out of range. Total cells: {len(cells)}")

if cells[cell_index].get("cell_type") != "code":
    raise SystemExit(f"Cell {cell_index + 1} is not a code cell.")

cells[cell_index]["source"] = code_lines

with nb_path.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=True, indent=4)

print("Updated cell #VSC-ae34b43c")
