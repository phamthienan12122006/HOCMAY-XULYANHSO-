"""
PHAN VIEC CUA KHIEM - Dong 2 (kiem tra anh mau) va Dong 5 (tien xu ly + ROI)
Chay tren Google Colab sau khi nhan manifest.csv / split.csv tu Kim.

manifest.csv cua Kim (repo HOCMAY-XULYANHSO-): cot 'relative_path' (data/raw/KL_x/xxx.png)
va 'kl_grade' (0-4). Chay file nay tu thu muc goc cua repo sau khi da chay
scripts/download_dataset.py de co anh trong data/raw/.

Dau ra (trong OUT_DIR):
  kiem_tra_anh_mau.csv, van_de_phat_hien.csv, anh_mau_5_lop.png   (dong 2)
  A/ B/ C/ (3 phien ban anh), roi.csv, nhat_ky_loi.csv, minh_hoa_roi.png (dong 5)
"""
import os, cv2, numpy as np, pandas as pd

# ---------------- CAU HINH ----------------
MANIFEST = "data/metadata/manifest.csv"
PATH_COL, LABEL_COL = "relative_path", "kl_grade"
OUT_DIR  = "xu_ly_anh"
SIZE     = 224          # kich thuoc anh dau ra (vuong)
CLAHE_CLIP, CLAHE_TILE = 2.0, (8, 8)
BILATERAL = dict(d=7, sigmaColor=50, sigmaSpace=50)   # hoac dung Gaussian: xem denoise()
USE_BILATERAL = True
N_SAMPLE_PER_CLASS = 20
# ------------------------------------------

def read_gray(p):
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None: return None
    if img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img

# ======== DONG 2: KIEM TRA ANH MAU ========
def kiem_tra_anh_mau(df):
    rows, issues = [], []
    for lb, g in df.groupby(LABEL_COL):
        for p in g.sample(min(N_SAMPLE_PER_CLASS, len(g)), random_state=0)[PATH_COL]:
            im = read_gray(p)
            if im is None:
                rows.append(dict(path=p, label=lb, doc_duoc=False)); issues.append((p, lb, "Khong doc duoc anh")); continue
            h, w = im.shape
            lap = cv2.Laplacian(im, cv2.CV_64F).var()          # do net
            r = dict(path=p, label=lb, doc_duoc=True, h=h, w=w, mean=im.mean(), std=im.std(), do_net=lap)
            rows.append(r)
            if im.std() < 20:   issues.append((p, lb, f"Tuong phan thap (std={im.std():.1f})"))
            if lap < 20:        issues.append((p, lb, f"Anh mo (Laplacian var={lap:.1f})"))
            if im.mean() < 30 or im.mean() > 225: issues.append((p, lb, f"Qua toi/qua sang (mean={im.mean():.0f})"))
            if min(h, w) < 100: issues.append((p, lb, f"Kich thuoc nho ({w}x{h})"))
    rep = pd.DataFrame(rows)
    rep.to_csv(f"{OUT_DIR}/kiem_tra_anh_mau.csv", index=False)
    pd.DataFrame(issues, columns=["path", "label", "van_de"]).to_csv(f"{OUT_DIR}/van_de_phat_hien.csv", index=False)
    print(rep.groupby("label")[["h", "w", "mean", "std", "do_net"]].agg(["min", "mean", "max"]).round(1))
    print(f"Phat hien {len(issues)} van de / {len(rep)} anh mau")
    # luoi 5 lop x 5 anh
    labs = sorted(df[LABEL_COL].unique())
    tiles = []
    for lb in labs:
        ims = [cv2.resize(read_gray(p), (150, 150)) for p in df[df[LABEL_COL] == lb][PATH_COL].head(5) if read_gray(p) is not None]
        ims += [np.zeros((150, 150), np.uint8)] * (5 - len(ims))
        row = np.hstack(ims); cv2.putText(row, f"KL{lb}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, 255, 2); tiles.append(row)
    cv2.imwrite(f"{OUT_DIR}/anh_mau_5_lop.png", np.vstack(tiles))

# ======== DONG 5: TIEN XU LY + ROI ========
def clahe(g): return cv2.createCLAHE(CLAHE_CLIP, CLAHE_TILE).apply(g)
def denoise(g):
    if USE_BILATERAL: return cv2.bilateralFilter(g, **BILATERAL)
    return cv2.GaussianBlur(g, (5, 5), 1.0)

def tim_roi(g):
    """ROI khe khop: tim duong khe khop (vung toi nam ngang giua xuong dui va xuong chay)
    bang profile do sang theo hang o vung giua anh. Tra ve (x,y,w,h, phuong_phap)."""
    H, W = g.shape
    x0, x1 = int(W * .25), int(W * .75)
    prof = cv2.GaussianBlur(g[:, x0:x1].astype(np.float32).mean(axis=1, keepdims=True), (1, 15), 0).ravel()
    lo, hi = int(H * .30), int(H * .70)                       # khe khop thuong o giua anh
    seg = prof[lo:hi]
    # khe khop = cuc tieu cuc bo ro nhat (sau hai cuc dai cua xuong)
    cy = lo + int(np.argmin(seg))
    ok = (seg.max() - seg.min()) > 8                          # du tuong phan moi tin
    method = "profile" if ok else "fallback_giua"
    if not ok: cy = H // 2
    rw, rh = int(W * .70), int(H * .35)
    x = max(0, (W - rw) // 2); y = int(np.clip(cy - rh // 2, 0, H - rh))
    return x, y, rw, rh, method

def tien_xu_ly(df):
    for v in "ABC": os.makedirs(f"{OUT_DIR}/{v}", exist_ok=True)
    roi_rows, err_rows = [], []
    for i, r in df.iterrows():
        p = r[PATH_COL]; g = read_gray(p)
        if g is None: err_rows.append((p, "Khong doc duoc anh", "Bo qua")); continue
        try:
            g = cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            A = g                                  # A: anh xam goc
            B = clahe(g)                           # B: xam + CLAHE
            C = denoise(B)                         # C: xam + CLAHE + loc nhieu
            x, y, w, h, m = tim_roi(B)
            if m != "profile": err_rows.append((p, "Khong xac dinh duoc khe khop", "Dung ROI mac dinh giua anh"))
            name = os.path.splitext(os.path.basename(p))[0] + ".png"
            for v, im in zip("ABC", (A, B, C)): cv2.imwrite(f"{OUT_DIR}/{v}/{name}", im)
            roi_rows.append(dict(path=p, file=name, label=r.get(LABEL_COL), x=x, y=y, w=w, h=h, phuong_phap=m))
        except Exception as e:
            err_rows.append((p, f"Loi xu ly: {e}", "Bo qua"))
    pd.DataFrame(roi_rows).to_csv(f"{OUT_DIR}/roi.csv", index=False)
    pd.DataFrame(err_rows, columns=["path", "loi", "cach_xu_ly"]).to_csv(f"{OUT_DIR}/nhat_ky_loi.csv", index=False)
    print(f"Xu ly xong {len(roi_rows)}/{len(df)} anh, {len(err_rows)} dong nhat ky loi")
    # minh hoa: A | B | C | C+ROI
    roi = pd.DataFrame(roi_rows).head(5); tiles = []
    for _, r in roi.iterrows():
        a, b, c = (cv2.imread(f"{OUT_DIR}/{v}/{r.file}", 0) for v in "ABC")
        d = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR); cv2.rectangle(d, (r.x, r.y), (r.x + r.w, r.y + r.h), (0, 255, 0), 2)
        tiles.append(np.hstack([cv2.cvtColor(t, cv2.COLOR_GRAY2BGR) for t in (a, b, c)] + [d]))
    if tiles: cv2.imwrite(f"{OUT_DIR}/minh_hoa_roi.png", np.vstack(tiles))

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(MANIFEST)
    df = df[df["status"] == "ok"] if "status" in df else df
    kiem_tra_anh_mau(df)     # Dong 2
    tien_xu_ly(df)           # Dong 5
