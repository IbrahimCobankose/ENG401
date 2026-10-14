"""
ENG401 - Veri tasnifi: iki veri setinin kayıt envanteri

Çalıştırma:  python src/veri_tasnif.py
Çıktılar  :  src/output/  klasörüne CSV tabloları ve örnek grafikler

Gerekli paketler:  pip install h5py numpy pandas matplotlib

Kaynak kısaltmaları (yorumlarda):
  [V] veri dosyasının kendisi (h5 attribute'ları, txt/csv sütunları)
  [M] Whelan vd. (2025) makalesi, J. Bridge Eng. 30(7)
  [Z] SDOF SHM benchmark Zenodo sayfası (doi:10.5281/zenodo.17900302)
  [N] SDOF veri setiyle gelen notebook'lar (dataset/code/*.ipynb)
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

# Zenodo v2 (doi:10.5281/zenodo.15497673), MD5 5c3e2aff20e1984fce15bd18f57f62eb.
# v1'de strain verisi tüm senaryolarda aynı senaryodan kopyalanmıştı; v2'de düzeltildi.
RT345_H5 = DATA / "Rt345Bridge.h5.v2" / "Rt345Bridge.h5"
SDOF_OUT = DATA / "output" / "output"
SDOF_IN = DATA / "input" / "input"


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
# Hiyerarşi [V][M Fig. 7]:
#   /<senaryo>/acceleration/<test_k>/<A01..A30>/{y,z}
#   /<senaryo>/strain/<west|center|east>_lane/<S01..S20>
#   /<senaryo>/photographs/<1..>

# [M Table 2] Deney sırası ve case numaraları. h5'te gruplar alfabetik sıralı
# (baseline_1, baseline_2, damage_1 ...); deneyin gerçek sırası sadece makalede var.
RT345_MAKALE_TABLO2 = {
    "baseline_1": (1, "Initial baseline"),
    "damage_1": (2, "Damage scenario 1"),
    "damage_2": (3, "Damage scenario 2"),
    "baseline_2": (4, "Second baseline"),
    "damage_3": (5, "Damage scenario 3"),
    "damage_4": (6, "Damage scenario 4"),
    "damage_5": (7, "Damage scenario 5"),
    "damage_6": (8, "Damage scenario 6"),
}

# [M] "The full-scale range of the sensor is ±2 g". h5'te sensör tipi (LIS2L02AL)
# yazıyor ama ölçüm aralığı yazmıyor; bu yüzden değer makaleden alındı.
OLCUM_ARALIGI = 2 * 9.81  # m/s^2

# [M] "the wireless sensor node interfaced to the accelerometer at Location A14 did not
# respond ... data from this location are not contained in the archived data for
# Damage scenario 6." Dosyada bu kanal sıfırlarla dolu; nedeni makalede.
RT345_MAKALE_NOTU = {("damage_6", "A14"): "Makale: A14 düğümü bu senaryoda veri göndermedi"}


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
    """Tek bir ivme kaydını ve zaman eksenini döndürür (birim h5'teki 'unit' attribute'u)."""
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
    """Her (senaryo, test, sensör, yön) için bir satır: h5 metadata'sı + sinyal istatistikleri."""
    satirlar = []
    with h5py.File(RT345_H5, "r") as f:
        for sc, (case_no, makale_adi) in RT345_MAKALE_TABLO2.items():
            g = f[sc]
            for test, T in g["acceleration"].items():
                for sensor, S in T.items():
                    pos = S.attrs["position"]
                    fs = float(_txt(S.attrs["sample_rate"]))
                    for yon in ("y", "z"):
                        x = S[yon][()].ravel()
                        max_abs = float(np.nanmax(np.abs(x)))
                        satirlar.append(dict(
                            senaryo=sc,
                            case_no=case_no,                                  # [M Table 2]
                            makale_senaryo_adi=makale_adi,                    # [M Table 2]
                            damage_state=_txt(g.attrs["damage_state"]),       # [V]
                            damage_type=_txt(g.attrs["damage_type"]),         # [V]
                            damage_location=_txt(g.attrs["damage_location"]), # [V]
                            damage_severity=_txt(g.attrs["damage_severity"]), # [V]
                            test=test, kayit_id=f"{sc}/{test}",
                            zaman=_txt(T.attrs["time"]),                      # [V]
                            sensor=sensor, yon=yon,
                            sensor_name=_txt(S.attrs["name"]),                # [V]
                            sensor_type=_txt(S.attrs["type"]),                # [V]
                            location=_txt(S.attrs["location"]),               # [V]
                            x_m=pos[0], y_m=pos[1], z_m=pos[2],               # [V]
                            sample_rate_hz=fs,                                # [V]
                            internal_sample_rate_hz=float(_txt(S.attrs["internal_sample_rate"])),  # [V]
                            unit=_txt(S.attrs["unit"]),                       # [V]
                            h5_number_of_samples=int(_txt(T.attrs["number of samples"])),  # [V]
                            n_ornek=x.size, sure_s=x.size / fs,
                            nan_sayisi=int(np.isnan(x).sum()),
                            tum_degerler_sifir=bool(np.all(x == 0)),
                            std=float(np.nanstd(x)),
                            rms=float(np.sqrt(np.nanmean(x**2))),
                            max_abs=max_abs,
                            olcum_araligi_asildi=max_abs > OLCUM_ARALIGI,     # [M] ±2 g
                            makale_notu=RT345_MAKALE_NOTU.get((sc, sensor), ""),
                        ))
    return pd.DataFrame(satirlar).sort_values(["case_no", "test", "sensor", "yon"])


def rt345_kayit_tablosu(env):
    """Kayıt (senaryo/test) bazında özet."""
    def kanal_listesi(maske):
        return ", ".join(sorted(set(env.loc[maske.index[maske], "sensor"] + "-"
                                    + env.loc[maske.index[maske], "yon"])))
    return (env.groupby(["case_no", "senaryo", "makale_senaryo_adi", "test", "kayit_id",
                         "damage_type", "damage_location", "damage_severity", "zaman",
                         "sample_rate_hz", "h5_number_of_samples", "n_ornek", "sure_s"])
            .agg(sensor_sayisi=("sensor", "nunique"),
                 tum_degerler_sifir_kanallar=("tum_degerler_sifir", kanal_listesi),
                 olcum_araligi_asan_kanallar=("olcum_araligi_asildi", kanal_listesi),
                 makale_notu=("makale_notu", lambda s: "; ".join(sorted(set(s) - {""}))))
            .reset_index())


# [M] ST350 strain ölçerlerin tam ölçek aralığı: "a full-scale range of ±4,000 με".
# h5'te sensör tipi var ama aralık yok; bu yüzden değer makaleden alındı.
STRAIN_OLCUM_ARALIGI = 4000.0  # microstrain


def rt345_strain_envanteri():
    """Her (senaryo, şerit, strain sensörü) için bir satır: h5 metadata'sı + sinyal istatistikleri.
    Strain, kamyon üç şerit konumunda dururken alınmış statik yükleme ölçümüdür [M]."""
    satirlar = []
    with h5py.File(RT345_H5, "r") as f:
        for sc, (case_no, makale_adi) in RT345_MAKALE_TABLO2.items():
            g = f[sc]
            for serit, L in g["strain"].items():
                for sensor, S in L.items():
                    x = S[()].ravel()
                    fs = float(_txt(S.attrs["sample_rate"]))
                    pos = S.attrs["position"]
                    satirlar.append(dict(
                        senaryo=sc, case_no=case_no, makale_senaryo_adi=makale_adi,  # [M Table 2]
                        damage_type=_txt(g.attrs["damage_type"]),                  # [V]
                        damage_location=_txt(g.attrs["damage_location"]),          # [V]
                        damage_severity=_txt(g.attrs["damage_severity"]),          # [V]
                        serit=serit, sensor=sensor,
                        sensor_name=_txt(S.attrs["name"]),                         # [V]
                        sensor_type=_txt(S.attrs["type"]),                         # [V]
                        location=_txt(S.attrs["location"]),                        # [V]
                        x_m=pos[0], y_m=pos[1], z_m=pos[2],                        # [V]
                        sample_rate_hz=fs,                                         # [V]
                        internal_sample_rate_hz=float(_txt(S.attrs["internal_sample_rate"])),  # [V]
                        unit=_txt(S.attrs["unit"]),                                # [V]
                        n_ornek=x.size, sure_s=x.size / fs,
                        nan_sayisi=int(np.isnan(x).sum()),
                        tum_degerler_sifir=bool(np.all(x == 0)),
                        min=float(np.nanmin(x)), max=float(np.nanmax(x)),
                        ortalama=float(np.nanmean(x)), std=float(np.nanstd(x)),
                        olcum_araligi_asildi=float(np.nanmax(np.abs(x))) > STRAIN_OLCUM_ARALIGI,  # [M]
                    ))
    return pd.DataFrame(satirlar).sort_values(["case_no", "serit", "sensor"])


def rt345_fotograf_listesi():
    """photographs grubundaki görüntülerin listesi [V]. Görüntülerde attribute yok."""
    satirlar = []
    with h5py.File(RT345_H5, "r") as f:
        for sc, (case_no, makale_adi) in RT345_MAKALE_TABLO2.items():
            if "photographs" not in f[sc]:
                satirlar.append(dict(senaryo=sc, case_no=case_no, fotograf="",
                                     not_="Bu senaryoda photographs grubu yok"))
                continue
            for ad, ds in f[sc]["photographs"].items():
                satirlar.append(dict(senaryo=sc, case_no=case_no, fotograf=ad,
                                     h5_yolu=f"{sc}/photographs/{ad}",
                                     shape=str(ds.shape), dtype=str(ds.dtype), not_=""))
    df = pd.DataFrame(satirlar)
    df["fotograf_sira"] = pd.to_numeric(df["fotograf"], errors="coerce")
    return (df.sort_values(["case_no", "fotograf_sira"]).drop(columns="fotograf_sira")
            .rename(columns={"not_": "not"}))


def rt345_genel_bilgi():
    """Dosyanın kök (root) attribute'ları [V]."""
    with h5py.File(RT345_H5, "r") as f:
        return pd.DataFrame([dict(attribute=k, deger=_txt(v)) for k, v in f.attrs.items()])


def rt345_ornek_grafik():
    """A21 düşey ivme. A21 seçildi çünkü makalede örnek veri olarak gösterilen
    sensör bu [M Fig. 6a]. Üstte baseline_1/test_1 zaman serisi, altta 8 senaryonun
    test_1 kayıtlarının PSD'si."""
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    t, x, fs = rt345_ivme_oku("baseline_1", "test_1", "A21", "z")
    ax[0].plot(t, x, lw=0.5)
    ax[0].set(xlabel="Zaman (s)", ylabel="İvme (h5 'unit': m/s²)",
              title="A21 - z (düşey), baseline_1 / test_1")
    for sc in RT345_MAKALE_TABLO2:
        _, x, fs = rt345_ivme_oku(sc, "test_1", "A21", "z")
        f, P = welch(x, fs=fs, nperseg=2048)
        ax[1].semilogy(f, P, lw=0.9, label=sc)
    ax[1].set(xlabel="Frekans (Hz)", ylabel="PSD", xlim=(0, 64),
              title="A21 - z, test_1, tüm senaryolar")
    ax[1].legend(fontsize=8, ncol=2)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CIKTI / "rt345_A21_ornek.png", dpi=130)
    plt.close(fig)


# =============================================================================
# 2) SDOF SHM BENCHMARK  (her saat için ayrı küçük .h5 + txt etiketler)
# =============================================================================
# acceleration/<Dx>/accNNNNN-k.h5 -> içinde tek dataset 'acc', shape (1, 18000) [V]
# 3 dakika @ 100 Hz [Z]
# Klasör -> etiket dosyası eşleşmesi dosya adlarından [V]: 1_D1, 2_D2.1, ... 9_D5
SDOF_ETIKET_DOSYASI = {
    "D1": "1_D1_labels.txt",
    "D2_1": "2_D2.1_labels.txt",
    "D2_2": "3_D2.2_labels.txt",
    "D2_3": "4_D2.3_labels.txt",
    "D3_1": "5_D3.1_labels.txt",
    "D3_2": "6_D3.2_labels.txt",
    "D3_3": "7_D3.3_labels.txt",
    "D4": "8_D4_labels.txt",
    "D5": "9_D5_labels.txt",
}
SDOF_FS = 100.0  # [Z] "3-minute signals at 100 Hz"


def sdof_zaman_izgarasi():
    """Saat indeksi -> tarih. Notebook'lardaki tanımın aynısı [N 2_SDOF, 3_SHM]:
    2020-01-01 00:00'dan saatlik, 29 Şubat'lar çıkarılmış ('erase feb 29').
    3_SHM.ipynb dosya adını bu dizinin sırası ile veriyor: acc{i:05d}-1.h5"""
    t = pd.date_range("2020-01-01 00:00", "2023-12-31 23:00", freq="h")
    t = t[~((t.month == 2) & (t.day == 29))]
    return pd.DataFrame({"Index": np.arange(len(t)), "Date": t})


def sdof_ivme_oku(senaryo, saat_index):
    path = next((SDOF_OUT / "acceleration" / senaryo).glob(f"acc{saat_index:05d}-*.h5"))
    with h5py.File(path, "r") as f:
        x = f["acc"][()].ravel().astype(np.float64)
    return np.arange(x.size) / SDOF_FS, x, SDOF_FS


# Senaryo klasörü -> km_pairs dosyası. Eşleşme dosya adlarından [V] (D21 -> D2_1 ...).
# Sütun 0 = rijitlik k, sütun 1 = kütle m [N 2_SDOF, korozyon hücreleri].
# D4 için km dosyası yok. Varyant dosyalar (1_km_D1.1, 3_km_D22_1 ...) hiçbir yerde
# açıklanmadığı için satırlara bağlanmıyor; sadece sdof_dosya_envanteri.csv'de listeleniyor.
SDOF_KM_DOSYASI = {
    "D1": "1_km_D1.txt", "D2_1": "2_km_D21.txt", "D2_2": "3_km_D22.txt", "D2_3": "4_km_D23.txt",
    "D3_1": "5_km_D31.txt", "D3_2": "6_km_D32.txt", "D3_3": "7_km_D33.txt", "D5": "8_km_D5.txt",
}
# 1_deflection_D1-5.txt sütun sırası [N 2_SDOF hücre 58-59]: D1, D21, D22, D23, D31, D32, D33, D5.
# Satır i = saat indeksi i; birim mm ("delta = P / k * 1000 # [mm]") [N].
SDOF_DEFLECTION_SUTUN = ["D1", "D2_1", "D2_2", "D2_3", "D3_1", "D3_2", "D3_3", "D5"]


def _sdof_km(sc):
    """km dosyasını saat indeksine bağlar. 1_km_D1 saat 0'dan, diğerleri 13104'ten
    (indexDataset2Start) başlar ve 26279'da biter [N 2_SDOF]."""
    if sc not in SDOF_KM_DOSYASI:
        return pd.DataFrame(columns=["Index", "k", "m"])
    km = np.loadtxt(SDOF_IN / "km_pairs" / SDOF_KM_DOSYASI[sc])
    bas = 26280 - len(km)
    return pd.DataFrame({"Index": np.arange(bas, bas + len(km)), "k": km[:, 0], "m": km[:, 1]})


def _sinyal_istatistik(path):
    with h5py.File(path, "r") as f:
        x = f["acc"][()].ravel().astype(np.float64)
    return dict(n_ornek=x.size, nan_sayisi=int(np.isnan(x).sum()),
                tum_degerler_sifir=bool(np.all(x == 0)),
                ortalama=float(np.nanmean(x)), std=float(np.nanstd(x)),
                rms=float(np.sqrt(np.nanmean(x**2))), max_abs=float(np.nanmax(np.abs(x))))


def sdof_girdi_tablosu():
    """Sıcaklık (2019-2023) ve yük (2020-2023) dosyalarının tamamı, tarih üzerinden [V].
    Saat ızgarasına düşen satırlara Index yazılır [N]; 2019 satırlarının Index'i boştur."""
    yuk = pd.read_csv(SDOF_IN / "load" / "1_totLoad.txt", sep="\t", parse_dates=["Date"])
    sic = pd.read_csv(SDOF_IN / "temperature" / "1_temperature.txt", sep="\t", parse_dates=["Date"])
    t = sic.merge(yuk, on="Date", how="outer").merge(sdof_zaman_izgarasi(), on="Date", how="left")
    t["Index"] = t["Index"].astype("Int64")
    return t[["Date", "Index", "Temperature", "TotLoad", "LiveLoadLT", "LiveLoadST"]].sort_values("Date")


def sdof_envanter(havuz):
    """Her saatlik ivme dosyası için bir satır: etiket [V] + tarih [N] + sıcaklık/yük [V]
    + sinyal istatistikleri [V] + k, m [V][N] + sehim [V][N]."""
    zaman = sdof_zaman_izgarasi().merge(sdof_girdi_tablosu().drop(columns="Date").dropna(subset=["Index"])
                                        .astype({"Index": int}), on="Index", how="left")
    sehim = np.loadtxt(SDOF_OUT / "deflection" / "1_deflection_D1-5.txt")

    parcalar = []
    for sc, lab_dosya in SDOF_ETIKET_DOSYASI.items():
        dosyalar = sorted((SDOF_OUT / "acceleration" / sc).glob("acc*.h5"))
        d = pd.DataFrame({"senaryo": sc,
                          "Index": [int(p.stem[3:8]) for p in dosyalar],
                          "dosya": [p.name for p in dosyalar]})
        # keep_default_na=False: dosyadaki "N/A" metni boş hücreye çevrilmesin
        lab = pd.read_csv(SDOF_OUT / "labels" / lab_dosya, sep="\t", keep_default_na=False)
        d = d.merge(lab, on="Index", how="left").merge(zaman, on="Index", how="left")
        d = d.merge(_sdof_km(sc), on="Index", how="left")
        if sc in SDOF_DEFLECTION_SUTUN:
            sutun = sehim[:, SDOF_DEFLECTION_SUTUN.index(sc)]
            d["deflection_mm"] = [sutun[i] if i < len(sutun) else np.nan for i in d["Index"]]
        else:
            d["deflection_mm"] = np.nan
        ist = pd.DataFrame(havuz.map(_sinyal_istatistik, dosyalar, chunksize=200))
        parcalar.append(pd.concat([d, ist], axis=1))
        print(f"  {sc}: {len(dosyalar)} dosya okundu")
    return pd.concat(parcalar, ignore_index=True)


def sdof_etiket_tablosu():
    """Etiket dosyalarının bütün satırları (ivme dosyası olsun olmasın) + tarih [N]
    + o saatin ivme dosyası var mı [V]."""
    zaman = sdof_zaman_izgarasi()
    parcalar = []
    for sc, lab_dosya in SDOF_ETIKET_DOSYASI.items():
        lab = pd.read_csv(SDOF_OUT / "labels" / lab_dosya, sep="\t", keep_default_na=False)
        var = {int(p.stem[3:8]) for p in (SDOF_OUT / "acceleration" / sc).glob("acc*.h5")}
        lab.insert(0, "senaryo", sc)
        lab = lab.merge(zaman, on="Index", how="left")
        lab["ivme_dosyasi_var"] = lab["Index"].isin(var)
        parcalar.append(lab)
    return pd.concat(parcalar, ignore_index=True)


def sdof_dosya_envanteri():
    """input/ ve output/ altındaki ivme dışı her dosya için bir satır [V]."""
    satirlar = []
    dosyalar = sorted(p for p in list(SDOF_IN.rglob("*")) + list(SDOF_OUT.rglob("*"))
                      if p.is_file() and "acceleration" not in p.parts)
    km_ana = {v: np.loadtxt(SDOF_IN / "km_pairs" / v) for v in SDOF_KM_DOSYASI.values()}
    for p in dosyalar:
        sat = dict(klasor=str(p.parent.relative_to(DATA)).replace("\\", "/"), dosya=p.name,
                   boyut_bayt=p.stat().st_size)
        if p.suffix == ".csv":  # termo_20xx.csv: ';' ayraçlı, ',' ondalıklı [N 1_EOVs]
            t = pd.read_csv(p, sep=";", decimal=",")
            tarih = pd.to_datetime(t["Date"], format="%d/%m/%Y %H:%M")
            sat.update(satir=len(t), sutun=t.shape[1], sutun_adlari=", ".join(t.columns),
                       ilk=tarih.min(), son=tarih.max(), bos_hucre=int(t.isna().sum().sum()))
        elif "km_pairs" in p.parts or "deflection" in p.parts:
            a = np.loadtxt(p)
            sat.update(satir=a.shape[0], sutun=a.shape[1] if a.ndim > 1 else 1,
                       bos_hucre=int(np.isnan(a).sum()))
            if "km_pairs" in p.parts:
                ana = next((k for k in km_ana if p.name != k and p.name.startswith(k[:-4])), None)
                sat["kullanildigi_senaryo"] = next((s for s, k in SDOF_KM_DOSYASI.items() if k == p.name), "")
                if ana is not None:
                    sat["ana_dosya"] = ana
                    sat["ana_dosyayla_ayni"] = bool(np.array_equal(a, km_ana[ana]))
        else:  # labels, 1_temperature, 1_totLoad: başlıklı, tab ayraçlı
            t = pd.read_csv(p, sep="\t", keep_default_na=False)
            sat.update(satir=len(t), sutun=t.shape[1], sutun_adlari=", ".join(t.columns),
                       bos_hucre=int((t == "").sum().sum()))
            if "Date" in t:
                sat.update(ilk=t["Date"].min(), son=t["Date"].max())
        satirlar.append(sat)
    return pd.DataFrame(satirlar)


def sdof_ornek_grafik():
    """Üstte acc00000-1.h5 (veri setinin 4_SFM.ipynb notebook'unda örnek olarak açılan
    dosya [N]). Altta saat 26278'in PSD'si: D4 dışındaki tüm klasörlerde bulunan son
    saat [V]; D4 klasöründe bu saatin dosyası yok."""
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    t, x, fs = sdof_ivme_oku("D1", 0)
    ax[0].plot(t, x, lw=0.5)
    ax[0].set(xlabel="Zaman (s)", ylabel="acc", title="SDOF - D1 / acc00000-1.h5")
    idx = 26278
    for sc in SDOF_ETIKET_DOSYASI:
        if not list((SDOF_OUT / "acceleration" / sc).glob(f"acc{idx:05d}-*.h5")):
            continue
        _, x, fs = sdof_ivme_oku(sc, idx)
        f, P = welch(x, fs=fs, nperseg=4096)
        ax[1].semilogy(f, P, lw=0.9, label=sc)
    ax[1].set(xlabel="Frekans (Hz)", ylabel="PSD", xlim=(0, 20),
              title=f"SDOF - saat {idx}, tüm senaryolar")
    ax[1].legend(fontsize=8, ncol=2)
    for a in ax:
        a.grid(alpha=0.3)
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
    rt345_strain_envanteri().to_csv(CIKTI / "rt345_strain_envanteri.csv", index=False, encoding="utf-8-sig")
    rt345_fotograf_listesi().to_csv(CIKTI / "rt345_fotograf_listesi.csv", index=False, encoding="utf-8-sig")
    rt345_genel_bilgi().to_csv(CIKTI / "rt345_genel_bilgi.csv", index=False, encoding="utf-8-sig")
    print("\nRoute 345 - kayıt tablosu:")
    print(kayit[["kayit_id", "sure_s", "sensor_sayisi", "tum_degerler_sifir_kanallar",
                 "olcum_araligi_asan_kanallar", "makale_notu"]].to_string(index=False))
    rt345_ornek_grafik()

    # --- SDOF ---
    from multiprocessing import Pool
    print("\nSDOF ivme dosyaları okunuyor...")
    with Pool() as havuz:
        senv = sdof_envanter(havuz)
    senv.to_csv(CIKTI / "sdof_kayit_envanteri.csv", index=False, encoding="utf-8-sig")
    sdof_etiket_tablosu().to_csv(CIKTI / "sdof_etiket_tablosu.csv", index=False, encoding="utf-8-sig")
    sdof_girdi_tablosu().to_csv(CIKTI / "sdof_girdi_tablosu.csv", index=False, encoding="utf-8-sig")
    sdof_dosya_envanteri().to_csv(CIKTI / "sdof_dosya_envanteri.csv", index=False, encoding="utf-8-sig")
    ozet = (senv.groupby(["senaryo", "Label", "Description"])
            .agg(dosya=("dosya", "size"), ilk=("Date", "min"), son=("Date", "max"),
                 nan_iceren_dosya=("nan_sayisi", lambda s: int((s > 0).sum())),
                 tum_degerler_sifir_dosya=("tum_degerler_sifir", "sum"))
            .reset_index())
    ozet.to_csv(CIKTI / "sdof_ozet.csv", index=False, encoding="utf-8-sig")
    print("\nSDOF - senaryo/etiket özeti:")
    print(ozet.to_string(index=False))
    sdof_ornek_grafik()
    print(f"\nÇıktılar: {CIKTI}")
