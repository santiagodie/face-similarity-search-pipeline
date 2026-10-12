import pickle
import urllib.request
import zipfile
import hashlib
import numpy as np
import cv2
import dlib
import gradio as gr
from collections import OrderedDict
from pathlib import Path
from deepface import DeepFace
from PIL import Image
import vptree
import bz2
# ── 1. VP-Tree + paths + vecs_arr desde GitHub ────────────────────────────
print("[INFO] descargando vptree.pkl...")
urllib.request.urlretrieve("VPTREE_URL","vptree.pkl")
with open("vptree.pkl", "rb") as f:
    data = pickle.load(f)

paths = data["paths"]
vecs_arr = data["vecs_arr"]
print(f"[INFO] {len(paths)} caras cargadas")

# ── 2. Caras recortadas desde GitHub releases ──────────────────────────────
print("[INFO] descargando carasRecortadas...")
urllib.request.urlretrieve("FACE_CROPS_URL","carasRecortadas.zip")
with zipfile.ZipFile("carasRecortadas.zip", "r") as z:
    z.extractall("carasRecortadas")
print("[INFO] carasRecortadas listas")

CARAS = Path("carasRecortadas")
# ── 3. VP-Tree ─────────────────────────────────────────────────────────────
class Vec:
    def __init__(self, idx, arr):
        self.idx = idx
        self.arr = arr
    def __lt__(self, other): return self.idx <  other.idx
    def __le__(self, other): return self.idx <= other.idx
    def __eq__(self, other): return self.idx == other.idx
    def __ge__(self, other): return self.idx >= other.idx
    def __gt__(self, other): return self.idx >  other.idx

def distancia_l2(a, b):
    return float(np.linalg.norm(a.arr - b.arr))

print("[INFO] construyendo VP-Tree...")
items = [Vec(i, vecs_arr[i]) for i in range(len(paths))]
vp    = vptree.VPTree(items, distancia_l2)
print("[INFO] VP-Tree listo")
# ── 4. Detectores dlib: HOG (rápido) + CNN (fallback) ─────────────────────
print("[INFO] inicializando detector HOG...")
detectorHOG = dlib.get_frontal_face_detector()

print("[INFO] descargando modelo dlib CNN (fallback)...")
urllib.request.urlretrieve("https://github.com/davisking/dlib-models/raw/master/mmod_human_face_detector.dat.bz2","mmod_human_face_detector.dat.bz2")
with bz2.open("mmod_human_face_detector.dat.bz2") as src, \
     open("mmod_human_face_detector.dat", "wb") as dst:
    dst.write(src.read())

detectorCNN = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
print("[INFO] detectores listos")

# ── 5. Warm-up Facenet ─────────────────────────────────────────────────────
print("[INFO] calentando Facenet...")
DeepFace.build_model("Facenet")
_dummy = np.zeros((160, 160, 3), dtype=np.uint8)
DeepFace.represent(_dummy, model_name="Facenet", enforce_detection=False)
print("[INFO] Facenet caliente")

# ── 6. Cache LRU de embeddings (por recorte 160x160) ──────────────────────
_EMB_CACHE_MAX = 200
_emb_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

def _hash_image(img_bgr: np.ndarray) -> str:
    return hashlib.blake2b(img_bgr.tobytes(), digest_size=16).hexdigest()

def _cache_get(h: str):
    if h in _emb_cache:
        _emb_cache.move_to_end(h)
        return _emb_cache[h]
    return None

def _cache_put(h: str, emb: np.ndarray):
    _emb_cache[h] = emb
    _emb_cache.move_to_end(h)
    if len(_emb_cache) > _EMB_CACHE_MAX:
        _emb_cache.popitem(last=False)

# ── 7. Funciones dlib ──────────────────────────────────────────────────────
def convert_and_trim_bb(image, rect):
    startX = max(0, rect.left())
    startY = max(0, rect.top())
    endX   = min(rect.right(),  image.shape[1])
    endY   = min(rect.bottom(), image.shape[0])
    return (startX, startY, endX - startX, endY - startY)

def detectarCaras(imagen_bgr):
    rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    rects_hog = detectorHOG(rgb, 1)
    if len(rects_hog) > 0:
        return [convert_and_trim_bb(imagen_bgr, r) for r in rects_hog], "HOG"
    results_cnn = detectorCNN(rgb, 1)
    return [convert_and_trim_bb(imagen_bgr, r.rect) for r in results_cnn], "CNN"

def recortarYNormalizar(imagen_bgr, x, y, w, h):
    ih, iw      = imagen_bgr.shape[:2]
    margen_px_w = int(w * 0.20)
    margen_px_h = int(h * 0.20)
    x1 = max(0,  x - margen_px_w)
    y1 = max(0,  y - margen_px_h)
    x2 = min(iw, x + w + margen_px_w)
    y2 = min(ih, y + h + margen_px_h)
    if (x2 - x1) < 20 or (y2 - y1) < 20:
        return None
    return cv2.resize(imagen_bgr[y1:y2, x1:x2], (160, 160),
                      interpolation=cv2.INTER_CUBIC)

# ── 8. Embeddings con batch + cache ───────────────────────────────────────
def _embedding_uno(recorte_bgr: np.ndarray) -> np.ndarray:
    h = _hash_image(recorte_bgr)
    hit = _cache_get(h)
    if hit is not None:
        return hit
    emb = DeepFace.represent(
        recorte_bgr, model_name="Facenet", enforce_detection=False
    )[0]["embedding"]
    emb = np.array(emb, dtype=np.float32)
    _cache_put(h, emb)
    return emb

def _embeddings_batch(recortes_bgr: list) -> list:
    embs = [None] * len(recortes_bgr)
    miss_idx, miss_crops, miss_hashes = [], [], []

    for i, r in enumerate(recortes_bgr):
        h = _hash_image(r)
        hit = _cache_get(h)
        if hit is not None:
            embs[i] = hit
        else:
            miss_idx.append(i)
            miss_crops.append(r)
            miss_hashes.append(h)

    if miss_crops:
        results = DeepFace.represent(
            miss_crops, model_name="Facenet", enforce_detection=False
        )
        if results and isinstance(results[0], dict):
            results = [results]
        for j, res in enumerate(results):
            emb = np.array(res[0]["embedding"], dtype=np.float32)
            embs[miss_idx[j]] = emb
            _cache_put(miss_hashes[j], emb)

    return embs

# ── 9. Búsqueda VP-Tree ────────────────────────────────────────────────────
def _vp_query(emb_vec, k=5):
    query   = Vec(-1, np.array(emb_vec, dtype=np.float32))
    vecinos = vp.get_n_nearest_neighbors(query, k)
    return [(float(d), paths[v.idx]) for d, v in vecinos]

def resultados_a_galeria(results):
    salida = []
    for dist, nombre in results:
        ruta = CARAS / nombre
        img  = Image.open(ruta) if ruta.exists() else Image.new("RGB", (160, 160))
        salida.append((img, f"d={dist:.3f}"))
    return salida

# ── 10. Opción A: cara recortada ────────────────────────────────────────────
def buscar_cara(imagen_pil, progress=gr.Progress()):
    if imagen_pil is None:
        return []
    progress(0.1, desc="Preparando imagen...")
    img_bgr = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, (160, 160), interpolation=cv2.INTER_CUBIC)
    progress(0.5, desc="Calculando embedding...")
    emb = _embedding_uno(img_bgr)
    progress(0.85, desc="Buscando en VP-Tree...")
    res = _vp_query(emb)
    progress(1.0, desc="Listo")
    return resultados_a_galeria(res)

# ── 11. Opción B: foto completa ────────────────────────────────────────────
MAX_FACES_POR_FOTO = 10
CHUNK_SIZE         = 3

def buscar_foto(imagen_pil, progress=gr.Progress()):
    if imagen_pil is None:
        yield [], ""
        return

    progress(0.05, desc="Preparando imagen...")
    img_bgr = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    if max(h, w) > 600:
        scale   = 600 / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

    progress(0.15, desc="Detectando caras (HOG)...")
    caras, detector_usado = detectarCaras(img_bgr)
    if not caras:
        yield [], "No se detectó ninguna cara en la imagen."
        return

    caras_ord        = sorted(caras, key=lambda c: c[2] * c[3], reverse=True)
    detectadas_total = len(caras_ord)

    recortes = []
    for (x, y, w_, h_) in caras_ord:
        r = recortarYNormalizar(img_bgr, x, y, w_, h_)
        if r is not None:
            recortes.append(r)
    if not recortes:
        yield [], "Caras detectadas pero no se pudieron recortar."
        return

    capeado = False
    if len(recortes) > MAX_FACES_POR_FOTO:
        recortes = recortes[:MAX_FACES_POR_FOTO]
        capeado  = True
    n = len(recortes)

    hits_iniciales = sum(
        1 for r in recortes if _cache_get(_hash_image(r)) is not None
    )
    msg_cap = f" (capado a {MAX_FACES_POR_FOTO} de {detectadas_total})" if capeado else ""
    yield [], f"{n} cara(s) detectada(s) por {detector_usado}{msg_cap} — procesando..."

    todas = []
    for i in range(0, n, CHUNK_SIZE):
        chunk     = recortes[i:i + CHUNK_SIZE]
        chunk_end = min(i + CHUNK_SIZE, n)
        progress(0.2 + 0.7 * (i / n), desc=f"Embeddings caras {i+1}-{chunk_end}/{n}...")
        embs = _embeddings_batch(chunk)
        for emb in embs:
            todas += resultados_a_galeria(_vp_query(emb))
        yield todas, f"{chunk_end}/{n} cara(s) procesadas..."

    progress(1.0, desc="Listo")
    cache_msg = f" ({hits_iniciales}/{n} desde cache)" if hits_iniciales else ""
    msg_final = (
        f"{n} cara(s) detectada(s) por {detector_usado}{msg_cap}{cache_msg} "
        f"— top 5 más similares por cara."
    )
    yield todas, msg_final

# ── 12. Interfaz Gradio ────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(), title="Motor de Búsqueda Facial") as demo:
    gr.Markdown("# Motor de Búsqueda Facial — Trabajo IA - Grupo 4")
    gr.Markdown(
        "Busca las caras más similares en el dataset del Gobierno de La Rioja. "
        "Usa la pestaña según el tipo de imagen que tengas."
    )

    with gr.Tabs():
        with gr.Tab("📷 Cara recortada"):
            gr.Markdown("Sube directamente un recorte de cara.")
            with gr.Row():
                inp_a = gr.Image(type="pil", label="Cara recortada")
                out_a = gr.Gallery(label="Resultados", columns=5, height="auto")
            gr.Button("Buscar", variant="primary").click(
                fn=buscar_cara, inputs=inp_a, outputs=out_a, concurrency_limit=1,
            )
            gr.Examples(
                examples=[
                    ["examples/cara1.jpg"], ["examples/cara2.jpg"],
                    ["examples/cara3.jpg"], ["examples/cara4.jpg"],
                    ["examples/cara5.jpg"], ["examples/cara6.jpg"],
                    ["examples/cara7.jpg"],
                ],
                inputs=inp_a, cache_examples=False,
            )

        with gr.Tab("🖼️ Foto completa"):
            gr.Markdown(
                "Sube una foto. Un detector dlib HOG detectará las caras. "
                "Cada cara se recorta y normaliza igual que en el Notebook 2."
            )
            with gr.Row():
                inp_b = gr.Image(type="pil", label="Foto completa")
                out_b = gr.Gallery(label="Resultados", columns=5, height="auto")
            msg_b = gr.Textbox(label="Detecciones", interactive=False)
            gr.Button("Buscar", variant="primary").click(
                fn=buscar_foto, inputs=inp_b, outputs=[out_b, msg_b], concurrency_limit=1,
            )
            gr.Markdown("**Examples**")
            with gr.Row():
                ex1 = gr.Image(value="examples/foto1.jpg", type="pil",
                               interactive=False, width=80, height=80)
                ex2 = gr.Image(value="examples/foto2.jpg", type="pil",
                               interactive=False, width=80, height=80)
            ex1.select(fn=lambda x: x, inputs=ex1, outputs=inp_b)
            ex2.select(fn=lambda x: x, inputs=ex2, outputs=inp_b)

demo.queue(default_concurrency_limit=1, max_size=10)
demo.launch(ssr_mode=False)
