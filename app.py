from pathlib import Path

import cv2
import gradio as gr
import joblib
import numpy as np
import tensorflow as tf
from scipy import ndimage
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


BASE_DIR = Path(__file__).resolve().parent

IMG_SIZE_G = (224, 224)
MORPH_KERNEL_G = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
HSV_GREEN_LOWER_G = np.array([40, 40, 40])
HSV_GREEN_UPPER_G = np.array([80, 255, 255])

SELECTED_CLASSES_G = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]

HEALTHY_LABEL = "healthy"

_gb_model = joblib.load(BASE_DIR / "gradient_boosting_model.joblib")
_gb_scaler = joblib.load(BASE_DIR / "gradient_boosting_scaler.joblib")
_le = joblib.load(BASE_DIR / "label_encoder.joblib")
_dl_model = tf.keras.models.load_model(BASE_DIR / "mobilenetv2_model.keras")


def _preprocess_array(img_rgb_uint8):
    img_res = cv2.resize(img_rgb_uint8, IMG_SIZE_G, interpolation=cv2.INTER_AREA)
    img_blur = cv2.GaussianBlur(img_res, (5, 5), 0)
    img_hsv = cv2.cvtColor(img_res, cv2.COLOR_RGB2HSV)
    img_gray = cv2.cvtColor(img_res, cv2.COLOR_RGB2GRAY)
    img_gray_blur = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
    img_norm = img_res / 255.0
    return {
        "rgb": img_res,
        "blur": img_blur,
        "hsv": img_hsv,
        "gray": img_gray,
        "gray_blur": img_gray_blur,
        "norm": img_norm,
    }


def _get_disease_mask(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, HSV_GREEN_LOWER_G, HSV_GREEN_UPPER_G)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL_G)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL_G)
    return mask


def _apply_clahe(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _extract_color_features(img_rgb, img_hsv, disease_mask):
    features = {}
    hist_rgb, hist_hsv = [], []
    for ch in range(3):
        h_rgb = cv2.calcHist([img_rgb], [ch], None, [16], [0, 256]).flatten()
        h_hsv = cv2.calcHist([img_hsv], [ch], None, [16], [0, 256]).flatten()
        hist_rgb.append(h_rgb / (h_rgb.sum() + 1e-7))
        hist_hsv.append(h_hsv / (h_hsv.sum() + 1e-7))
    features["hist_rgb"] = np.concatenate(hist_rgb)
    features["hist_hsv"] = np.concatenate(hist_hsv)

    stats_rgb, stats_hsv = [], []
    for ch in range(3):
        ch_rgb = img_rgb[:, :, ch].astype(float)
        ch_hsv = img_hsv[:, :, ch].astype(float)
        stats_rgb += [ch_rgb.mean(), ch_rgb.std(), skew(ch_rgb.flatten())]
        stats_hsv += [ch_hsv.mean(), ch_hsv.std(), skew(ch_hsv.flatten())]
    features["stats_rgb"] = np.array(stats_rgb)
    features["stats_hsv"] = np.array(stats_hsv)

    total = img_rgb.shape[0] * img_rgb.shape[1]
    features["disease_ratio"] = np.array([1 - (disease_mask.sum() / 255) / total])
    return features


def _extract_glcm_features(img_gray):
    img_q = (img_gray // 4).astype(np.uint8)
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(
        img_q, distances=[1], angles=angles, levels=64, symmetric=True, normed=True
    )
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    return np.concatenate([graycoprops(glcm, p).flatten() for p in props])


def _extract_lbp_features(img_gray, n_points=8, radius=1, n_bins=26):
    lbp = local_binary_pattern(img_gray, n_points, radius, method="uniform")
    hist, _ = np.histogram(
        lbp.flatten(), bins=n_bins, range=(0, n_points + 2), density=True
    )
    return hist


def _extract_shape_features(img_rgb, disease_mask):
    contours, _ = cv2.findContours(
        disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    total_px = img_rgb.shape[0] * img_rgb.shape[1]
    if len(contours) == 0:
        return np.zeros(7)

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True) + 1e-7
    circularity = (4 * np.pi * area) / (perimeter**2)

    _, _, w, h = cv2.boundingRect(cnt)
    extent = area / (w * h + 1e-7)

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull) + 1e-7
    solidity = area / hull_area

    eccentricity = 0.0
    if len(cnt) >= 5:
        (_, _), (ma, mi), _ = cv2.fitEllipse(cnt)
        a, b = max(ma, mi) / 2, min(ma, mi) / 2
        eccentricity = np.sqrt(1 - (b / (a + 1e-7)) ** 2)

    n_lesions = len([c for c in contours if cv2.contourArea(c) > 50])

    return np.array(
        [
            area / total_px,
            perimeter / (2 * total_px**0.5),
            circularity,
            extent,
            solidity,
            eccentricity,
            n_lesions / 100,
        ]
    )


def _build_feature_vector(img_rgb_uint8):
    prep = _preprocess_array(img_rgb_uint8)
    img_rgb = prep["rgb"]
    img_hsv = prep["hsv"]
    img_gray_b = prep["gray_blur"]
    disease_mask = _get_disease_mask(img_rgb)

    color_feats = _extract_color_features(img_rgb, img_hsv, disease_mask)
    glcm_feats = _extract_glcm_features(img_gray_b)
    lbp_feats = _extract_lbp_features(img_gray_b)
    shape_feats = _extract_shape_features(img_rgb, disease_mask)
    hu_feats = cv2.HuMoments(cv2.moments(img_gray_b)).flatten()
    hu_feats = -np.sign(hu_feats) * np.log10(np.abs(hu_feats) + 1e-10)

    parts = [
        color_feats["hist_rgb"],
        color_feats["hist_hsv"],
        color_feats["stats_rgb"],
        color_feats["stats_hsv"],
        color_feats["disease_ratio"],
        glcm_feats,
        lbp_feats,
        shape_feats,
        hu_feats,
    ]
    return np.concatenate(parts)


def _predict_ml(img_rgb_uint8):
    feat = _build_feature_vector(img_rgb_uint8)
    feat_s = _gb_scaler.transform(feat.reshape(1, -1))
    pred = _gb_model.predict(feat_s)[0]
    proba = _gb_model.predict_proba(feat_s)[0]

    class_name = _le.inverse_transform([pred])[0]
    confidence = float(proba.max())
    is_healthy = HEALTHY_LABEL in class_name.lower()
    return class_name, is_healthy, confidence


def _predict_dl(img_rgb_uint8):
    img = cv2.resize(img_rgb_uint8, IMG_SIZE_G, interpolation=cv2.INTER_AREA)
    img_f = img.astype(np.float32) / 255.0
    inp = np.expand_dims(img_f, axis=0)

    proba = _dl_model.predict(inp, verbose=0)[0]
    pred_idx = int(np.argmax(proba))
    class_name = _le.inverse_transform([pred_idx])[0]
    confidence = float(proba[pred_idx])
    is_healthy = HEALTHY_LABEL in class_name.lower()
    return class_name, is_healthy, confidence


def predict_leaf(image, method):
    if image is None:
        return "Please upload a leaf image first."

    img_rgb = np.array(image)
    if img_rgb.ndim == 2:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    elif img_rgb.shape[2] == 4:
        img_rgb = img_rgb[:, :, :3]

    if "Gradient Boosting" in method:
        class_name, is_healthy, confidence = _predict_ml(img_rgb)
        model_label = "Gradient Boosting (ML)"
    else:
        class_name, is_healthy, confidence = _predict_dl(img_rgb)
        model_label = "MobileNetV2 (DL)"

    short_name = (
        class_name.replace("Tomato__", "")
        .replace("Tomato_", "")
        .replace("_", " ")
    )
    status_icon = "✅" if is_healthy else "🔴"
    status_text = "HEALTHY" if is_healthy else "DISEASED"
    bar_color = "#2ecc71" if is_healthy else "#e74c3c"
    bar_width = int(confidence * 100)

    html = f"""
    <div style="font-family: Arial, sans-serif; padding: 16px; border-radius: 10px;
                border: 2px solid {bar_color}; background: #fafafa;">
        <h2 style="color:{bar_color}; margin-top:0;">{status_icon} Leaf {status_text}</h2>
        <table style="width:100%; border-collapse:collapse;">
            <tr>
                <td style="padding:4px; color:#555;"><b>Model used</b></td>
                <td style="padding:4px;">{model_label}</td>
            </tr>
            <tr>
                <td style="padding:4px; color:#555;"><b>Predicted class</b></td>
                <td style="padding:4px;">{short_name}</td>
            </tr>
            <tr>
                <td style="padding:4px; color:#555;"><b>Confidence</b></td>
                <td style="padding:4px;">
                    <div style="background:#eee; border-radius:6px; width:100%; height:18px; position:relative;">
                        <div style="background:{bar_color}; width:{bar_width}%; height:100%;
                                    border-radius:6px; transition: width 0.4s;"></div>
                    </div>
                    <small>{confidence*100:.1f}%</small>
                </td>
            </tr>
        </table>
        <hr style="margin:10px 0; border-color:#ddd;">
        <small style="color:#999;">
            Full class: <i>{class_name}</i>
        </small>
    </div>
    """
    return html


with gr.Blocks(
    title="Tomato Leaf Disease Detection",
    theme=gr.themes.Soft(primary_hue="green"),
    css="""
        .gradio-container { max-width: 820px; margin: auto; }
        #result-box { min-height: 160px; }
    """,
) as demo:
    gr.Markdown(
        """
        # Tomato Leaf Disease Detection
        Upload a tomato leaf image, choose a prediction method,
        then click **Analyze** to see if the leaf is **healthy** or **diseased**.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Leaf image",
                height=300,
            )

            method_radio = gr.Radio(
                choices=[
                    "🌿 Gradient Boosting (ML)",
                    "🧠 MobileNetV2 (DL)",
                ],
                value="🌿 Gradient Boosting (ML)",
                label="Prediction method",
            )

            predict_btn = gr.Button("Analyze", variant="primary", size="lg")

        with gr.Column(scale=1):
            result_html = gr.HTML(
                value="<p style='color:#aaa; padding:40px; text-align:center;'>"
                "The result will appear here after analysis.</p>",
                label="Result",
                elem_id="result-box",
            )

    gr.Markdown("### How to use")
    gr.Markdown(
        """
        1. Upload a tomato leaf image (JPG, PNG).
        2. Choose the method: Gradient Boosting (ML, fast) or MobileNetV2 (DL, more accurate).
        3. Click *Analyze*.
        4. The result shows: ✅ **HEALTHY** or 🔴 **DISEASED**, the predicted class, and confidence.

        Note: The model was trained on 10 tomato classes (PlantVillage).
        The main output indicates healthy vs diseased, not the detailed disease name.
        """
    )

    predict_btn.click(
        fn=predict_leaf,
        inputs=[image_input, method_radio],
        outputs=result_html,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
