"""
ENG401 - Veri tasnifi (İP2: veri sözlüğü + kayıt envanteri + ilk kalite kontrolü)

Çalıştırma:  python src/veri_tasnif.py
Çıktılar  :  src/output/  klasörüne CSV tabloları ve örnek grafikler

Gerekli paketler:  pip install h5py numpy pandas matplotlib
"""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import mlab


def welch(x, fs, nperseg):
    """Welch PSD (scipy.signal.welch yerine; ek kütüphane gerektirmez)."""
    P, f = mlab.psd(x, NFFT=nperseg, Fs=fs, noverlap=nperseg // 2,
                    window=mlab.window_hanning, detrend="mean")
    return f, P

PROJE = Path(__file__).resolve().parents[1]
DATA = PROJE / "dataset"
CIKTI = PROJE / "src" / "output"
CIKTI.mkdir(parents=True, exist_ok=True)

RT345_H5 = DATA / "Rt345Bridge.h5"
SDOF_OUT = DATA / "output" / "output"
SDOF_IN = DATA / "input" / "input"

G = 9.81
SENSOR_LIMIT = 2 * G  # LIS2L02AL tam ölçek ±2 g -> doygunluk eşiği


def _txt(v):
    """h5 attribute'ları bytes / numpy array gelebiliyor; düz Python değerine çevir."""
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray):
        v = v.tolist()
        return v[0] if len(v) == 1 else v
    return v


# =============================================================================
# 1) ROUTE 345 BRIDGE  (tek .h5 dosyası)
# =============================================================================
# Hiyerarşi:  /<senaryo>/acceleration/<test_k>/<A01..A30>/{y,z}
#             /<senaryo>/strain/<west|center|east>_lane/<S01..S20>
#             /<senaryo>/photographs/<1..>
RT345_SENARYO = {  # makaledeki Table 2 sırası
    "baseline_1": (1, "saglikli"),
    "damage_1": (2, "hasarli"),
    "damage_2": (3, "hasarli"),
    "baseline_2": (4, "saglikli"),
    "damage_3": (5, "hasarli"),
    "damage_4": (6, "hasarli"),
    "damage_5": (7, "hasarli"),
    "damage_6": (8, "hasarli"),
}


def h5_agaci_yazdir(path, max_derinlik=4):
    """h5 dosyasının klasör yapısını ekrana basar (dosyayı tanımak için ilk adım)."""
    def gez(obj, derinlik=0):
        if derinlik > max_derinlik:
            return
        for i, (ad, alt) in enumerate(obj.items()):
            if isinstance(alt, h5py.Dataset):
                print("  " * derinlik + f"- {ad}  [dataset] shape={alt.shape} dtype={alt.dtype}")
            else:
                print("  " * derinlik + f"+ {ad}/  ({len(alt)} öğe)")
                if i == 0:  # kardeşler aynı yapıda: sadece ilkinin içini aç
                    gez(alt, derinlik + 1)
    with h5py.File(path, "r") as f:
        print(f"\n{path.name}")
        for k, v in f.attrs.items():
            print(f"  @{k}: {_txt(v)}")
        gez(f)


def rt345_ivme_oku(senaryo, test, sensor, yon="z"):
    """Tek bir ivme kaydını (m/s^2) ve zaman eksenini döndürür."""
    with h5py.File(RT345_H5, "r") as f:
        ds = f[senaryo]["acceleration"][test][sensor]
        x = ds[yon][()].ravel()  # (1, N) -> (N,)
        fs = float(_txt(ds.attrs["sample_rate"]))
    return np.arange(x.size) / fs, x, fs


def rt345_kayit_matrisi(senaryo, test):
    """Bir testteki tüm sensörleri tek matrise koyar: shape = (30 sensör, 2 yön, N)."""
    with h5py.File(RT345_H5, "r") as f:
        T = f[senaryo]["acceleration"][test]
        sensorler = sorted(T.keys())
        X = np.stack([[T[s]["y"][()].ravel(), T[s]["z"][()].ravel()] for s in sensorler])
    return sensorler, X


def rt345_envanter():
    """Her (senaryo, test, sensör, yön) için bir satır: veri sözlüğü + basit kalite bayrakları."""
    satirlar = []
    with h5py.File(RT345_H5, "r") as f:
        for sc, (case_no, durum) in RT345_SENARYO.items():
            g = f[sc]
            for test, T in g["acceleration"].items():
                zaman = _txt(T.attrs["time"])
                for sensor, S in T.items():
                    pos = S.attrs["position"]
                    fs = float(_txt(S.attrs["sample_rate"]))
                    for yon in ("y", "z"):
                        x = S[yon][()].ravel()
                        sabit = np.allclose(x, x[0]) if x.size else True
                        satirlar.append(dict(
                            senaryo=sc, case_no=case_no, durum=durum,
                            hasar_tipi=_txt(g.attrs["damage_type"]),
                            hasar_yeri=_txt(g.attrs["damage_location"]),
                            hasar_siddeti=_txt(g.attrs["damage_severity"]),
                            test=test, kayit_id=f"{sc}/{test}", zaman=zaman,
                            sensor=sensor, yon=yon,
                            konum=_txt(S.attrs["location"]),
                            x_m=pos[0], y_m=pos[1], z_m=pos[2],
                            fs_hz=fs, n_ornek=x.size, sure_s=x.size / fs,
                            birim=_txt(S.attrs["unit"]),
                            nan_sayisi=int(np.isnan(x).sum()),
                            std=float(np.nanstd(x)), rms=float(np.sqrt(np.nanmean(x**2))),
                            max_abs=float(np.nanmax(np.abs(x))),
                            kalite_sabit_sinyal=bool(sabit),
                            kalite_doygunluk=bool(np.nanmax(np.abs(x)) >= 0.98 * SENSOR_LIMIT),
                        ))
    df = pd.DataFrame(satirlar).sort_values(["case_no", "test", "sensor", "yon"])
    # Kalite: aynı sensör/yön için std, diğer kayıtların medyanından çok farklıysa işaretle
    med = df.groupby(["sensor", "yon"])["std"].transform("median")
    df["kalite_std_aykiri"] = (df["std"] > 5 * med) | (df["std"] < med / 5)
    df["kalite_ok"] = ~(df.kalite_sabit_sinyal | df.kalite_doygunluk
                        | df.kalite_std_aykiri | (df.nan_sayisi > 0))
    return df


def rt345_kayit_tablosu(env):
    """Kayıt (test) bazında özet + önerilen eğitim/kalibrasyon/test rolü.

    Pencereler kayıt bazında ayrılmalı (veri sızıntısı olmasın). Öneri:
      baseline_1 -> eğitim (sağlıklı davranışı öğren)
      baseline_2 -> kalibrasyon/test (bağımsız sağlıklı: yanlış alarm oranı + eşik τ)
      damage_*   -> test (tespit oranı)
    """
    k = (env.groupby(["case_no", "senaryo", "test", "kayit_id", "durum", "hasar_tipi",
                      "hasar_yeri", "hasar_siddeti", "zaman", "fs_hz", "n_ornek", "sure_s"])
         .agg(sensor_sayisi=("sensor", "nunique"),
              bozuk_kanal=("kalite_ok", lambda s: int((~s).sum())),
              bozuk_kanal_listesi=("kalite_ok", lambda s: ", ".join(
                  sorted(set(env.loc[s[~s].index, "sensor"] + "-" + env.loc[s[~s].index, "yon"]))))
              ).reset_index())
    rol = {"baseline_1": "egitim", "baseline_2": "kalibrasyon/test-saglikli"}
    k["onerilen_rol"] = k.senaryo.map(rol).fillna("test-hasarli")
    return k


def rt345_ornek_grafik():
    """Makaledeki Fig. 6a'ya karşılık: A21 düşey ivme + PSD (sağlıklı vs hasarlı)."""
    fig, ax = plt.subplots(2, 1, figsize=(10, 7))
    for sc in ("baseline_1", "damage_4"):
        t, x, fs = rt345_ivme_oku(sc, "test_1", "A21", "z")
        ax[0].plot(t, x, lw=0.5, label=sc)
        f, P = welch(x, fs=fs, nperseg=2048)
        ax[1].semilogy(f, P, lw=1, label=sc)
    ax[0].set(xlabel="Zaman (s)", ylabel="İvme (m/s²)", title="A21 - z (düşey), test_1")
    ax[1].set(xlabel="Frekans (Hz)", ylabel="PSD", xlim=(0, 64))
    for a in ax:
        a.legend(); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CIKTI / "rt345_A21_ornek.png", dpi=130)
    plt.close(fig)


# =============================================================================
# 2) SDOF SHM BENCHMARK  (her saat için ayrı küçük .h5 + txt etiketler)
# =============================================================================
# acceleration/<Dx>/accNNNNN-k.h5  ->  içinde tek dataset: 'acc', shape (1, 18000) = 3 dk @ 100 Hz
# NNNNN = saat indeksi (0 = 2020-01-01 00:00), labels/*.txt satır 'Index' ile birebir eşleşir.
SDOF_SENARYO = {  # klasör: (etiket dosyası, açıklama)
    "D1": ("1_D1_labels.txt", "Hasarsız, sadece çevresel/operasyonel değişkenlik (EOV)"),
    "D2_1": ("2_D2.1_labels.txt", "Ani rijitlik kaybı -10%"),
    "D2_2": ("3_D2.2_labels.txt", "Ani hasar, aylık -1% artan (-18%e kadar)"),
    "D2_3": ("4_D2.3_labels.txt", "Ani hasar, ikiye katlanan (-1..-32%)"),
    "D3_1": ("5_D3.1_labels.txt", "Yavaş hasar (korozyon) L1"),
    "D3_2": ("6_D3.2_labels.txt", "Yavaş hasar (korozyon) L2"),
    "D3_3": ("7_D3.3_labels.txt", "Yavaş hasar (korozyon) L3"),
    "D4": ("8_D4_labels.txt", "Sensör arızaları (SFM) - yapı hasarsız"),
    "D5": ("9_D5_labels.txt", "Ani + yavaş hasar birlikte"),
}
SDOF_SINIF = {0: "saglikli", 1: "hasar_ani", 2: "hasar_yavas", 4: "hasar_ani+yavas", 7: "sensor_arizasi"}


def sdof_ivme_oku(senaryo, saat_index):
    path = next((SDOF_OUT / "acceleration" / senaryo).glob(f"acc{saat_index:05d}-*.h5"))
    with h5py.File(path, "r") as f:
        x = f["acc"][()].ravel().astype(np.float64)
    fs = 100.0
    return np.arange(x.size) / fs, x, fs


def sdof_envanter():
    """Her saatlik dosya için bir satır: tarih, etiket, sıcaklık, trafik yükü."""
    yuk = pd.read_csv(SDOF_IN / "load" / "1_totLoad.txt", sep="\t", parse_dates=["Date"])
    sic = pd.read_csv(SDOF_IN / "temperature" / "1_temperature.txt", sep="\t", parse_dates=["Date"])
    yuk = yuk.merge(sic, on="Date", how="left")
    yuk["Index"] = np.arange(len(yuk))  # etiket dosyalarındaki Index ile aynı saat ızgarası

    parcalar = []
    for sc, (lab_dosya, aciklama) in SDOF_SENARYO.items():
        dosyalar = sorted((SDOF_OUT / "acceleration" / sc).glob("acc*.h5"))
        idx = [int(p.stem[3:8]) for p in dosyalar]
        d = pd.DataFrame({"senaryo": sc, "Index": idx, "dosya": [p.name for p in dosyalar]})
        lab = pd.read_csv(SDOF_OUT / "labels" / lab_dosya, sep="\t")
        d = d.merge(lab, on="Index", how="left").merge(yuk, on="Index", how="left")
        d["senaryo_aciklama"] = aciklama
        parcalar.append(d)
    df = pd.concat(parcalar, ignore_index=True)
    df["sinif"] = df["Label"].map(SDOF_SINIF)
    return df.rename(columns={"Date": "tarih", "Temperature": "sicaklik_C",
                              "TotLoad": "toplam_yuk", "Description": "aciklama",
                              "Level/Type": "seviye"})


def sdof_ornek_grafik(env):
    fig, ax = plt.subplots(2, 1, figsize=(10, 7))
    for sc, idx in (("D1", 20000), ("D2_3", 20000)):
        t, x, fs = sdof_ivme_oku(sc, idx)
        r = env[(env.senaryo == sc) & (env.Index == idx)].iloc[0]
        etiket = f"{sc} #{idx} ({r.sinif}, {r.seviye}, {r.sicaklik_C:.1f}°C)"
        ax[0].plot(t, x, lw=0.5, label=etiket)
        f, P = welch(x, fs=fs, nperseg=4096)
        ax[1].semilogy(f, P, lw=1, label=etiket)
    ax[0].set(xlabel="Zaman (s)", ylabel="İvme", title="SDOF - aynı saat, sağlıklı vs hasarlı")
    ax[1].set(xlabel="Frekans (Hz)", ylabel="PSD", xlim=(0, 20))
    for a in ax:
        a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CIKTI / "sdof_ornek.png", dpi=130)
    plt.close(fig)


# =============================================================================
if __name__ == "__main__":
    # --- Route 345 ---
    h5_agaci_yazdir(RT345_H5)
    env = rt345_envanter()
    env.to_csv(CIKTI / "rt345_kanal_envanteri.csv", index=False, encoding="utf-8-sig")
    kayit = rt345_kayit_tablosu(env)
    kayit.to_csv(CIKTI / "rt345_kayit_tablosu.csv", index=False, encoding="utf-8-sig")
    print("\nRoute 345 - kayıt tablosu:")
    print(kayit[["kayit_id", "durum", "sure_s", "sensor_sayisi", "bozuk_kanal",
                 "bozuk_kanal_listesi", "onerilen_rol"]].to_string(index=False))
    rt345_ornek_grafik()

    # --- SDOF ---
    senv = sdof_envanter()
    senv.to_csv(CIKTI / "sdof_kayit_envanteri.csv", index=False, encoding="utf-8-sig")
    ozet = (senv.groupby(["senaryo", "sinif"])
            .agg(dosya=("dosya", "size"), ilk=("tarih", "min"), son=("tarih", "max"))
            .reset_index())
    ozet.to_csv(CIKTI / "sdof_ozet.csv", index=False, encoding="utf-8-sig")
    print("\nSDOF - senaryo/sınıf özeti:")
    print(ozet.to_string(index=False))
    sdof_ornek_grafik(senv)
    print(f"\nÇıktılar: {CIKTI}")
