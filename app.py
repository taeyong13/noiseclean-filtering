"""NoiseClean: aplikasi demonstrasi filtering / noise reduction untuk tugas PSD."""

from io import BytesIO
import wave

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.io import wavfile
from scipy.signal import wiener


st.set_page_config(page_title="NoiseClean | PSD", page_icon="N", layout="wide")


def inject_css() -> None:
    st.markdown(
        """
        <style>
          /* ---------- Base canvas: soft gradient + blurred color blobs ---------- */
          .stApp {
            background: linear-gradient(135deg, #dce9ff 0%, #e4e3ff 45%, #f3e6ff 100%);
          }
          .stApp::before {
            content: ''; position: fixed; z-index: -1; width: 30rem; height: 30rem; border-radius: 50%;
            top: -13rem; right: -8rem; background: rgba(140, 178, 255, .55); filter: blur(70px);
          }
          .stApp::after {
            content: ''; position: fixed; z-index: -1; width: 26rem; height: 26rem; border-radius: 50%;
            bottom: -12rem; left: 18rem; background: rgba(197, 150, 255, .45); filter: blur(70px);
          }
          [data-testid="stMainBlockContainer"] { max-width: 1360px; padding-top: 1.3rem; padding-bottom: 2.5rem; }
          [data-testid="stAppViewContainer"] { position: relative; }
          [data-testid="stHeader"] { background: transparent; }

          /* ---------- Reusable glass surface ---------- */
          .glass, .topbar, .flow-card, .info-card, .experiment-strip, .insight,
          .metric-card, [data-testid="stExpander"], [data-testid="stVerticalBlockBorderWrapper"],
          [data-testid="stFileUploaderDropzone"] {
            background: rgba(255, 255, 255, .5) !important;
            border: 1px solid rgba(255, 255, 255, .7) !important;
            backdrop-filter: blur(18px) saturate(160%);
            -webkit-backdrop-filter: blur(18px) saturate(160%);
            box-shadow: 0 8px 28px rgba(76, 92, 158, .12);
            border-radius: 18px;
          }

          /* ---------- Sidebar ---------- */
          [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, .38);
            border-right: 1px solid rgba(255, 255, 255, .6);
            backdrop-filter: blur(22px) saturate(160%);
            -webkit-backdrop-filter: blur(22px) saturate(160%);
          }
          /* Streamlit reserves top space for the (hidden) header inside the sidebar by default,
             which pushes all sidebar content down. Strip that out and set a small, deliberate gap instead. */
          [data-testid="stSidebar"] > div:first-child {
            background: transparent; padding-top: 0 !important; height: 100vh !important;
            overflow-y: auto !important; overscroll-behavior: contain;
          }
          [data-testid="stSidebarUserContent"] { padding-top: 1rem !important; }
          [data-testid="stSidebarUserContent"] > div:first-child { margin-top: 0 !important; padding-top: 0 !important; }
          [data-testid="stSidebarHeader"] { display: none; }
          [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
          [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
          [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stCaption { color: #0f172a !important; }
          [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #1e293b !important; }
          [data-testid="stSidebar"] hr { border-color: rgba(148, 163, 184, .35); }
          .control-title { margin: 1.1rem 0 .15rem; color: #64748b; font-size: .70rem;
                           font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }

          /* ---------- Streamlit input widgets, glassed to match ---------- */
          [data-baseweb="select"] > div, .stNumberInput input, .stTextInput input,
          [data-testid="stFileUploaderDropzone"] {
            border-radius: 12px !important;
          }
          .stSlider [data-baseweb="slider"] > div > div { background: rgba(124, 108, 255, .18); }
          .stSlider [role="slider"] { background: #7c6cff !important; box-shadow: 0 2px 8px rgba(124, 108, 255, .5); }
          [data-testid="stRadio"] label { color: #334155; }

          [data-testid="stFileUploaderDropzone"] button {
            background: linear-gradient(135deg, #7c6cff, #a855f7) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 999px !important;
            font-weight: 650 !important;
          }
          [data-testid="stFileUploaderDropzone"] button p { color: #fff !important; }
          [data-testid="stFileUploaderDropzoneInstructions"] svg { fill: #64748b !important; }
          [data-testid="stFileUploaderDropzoneInstructions"] span,
          [data-testid="stFileUploaderDropzoneInstructions"] small { color: #475569 !important; }

          /* ---------- Buttons: gradient pill, matches hero-tag/CTA in reference ---------- */
          .stButton > button, .stDownloadButton > button {
            background: linear-gradient(135deg, #7c6cff, #a855f7) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 999px !important;
            padding: .5rem 1.1rem !important;
            font-weight: 650 !important;
            box-shadow: 0 8px 20px rgba(124, 108, 255, .35);
            transition: transform .15s ease, box-shadow .15s ease;
          }
          .stButton > button:hover, .stDownloadButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 26px rgba(124, 108, 255, .45);
          }

          /* ---------- Top bar ---------- */
          .topbar { display:flex; align-items:center; justify-content:space-between; padding: .75rem 1rem; margin-bottom: 1.2rem; }
          .brand { display:flex; align-items:center; gap:.65rem; color:#1e293b; font-weight:800; font-size:1rem; }
          .brand-mark { width:29px; height:29px; display:grid; place-items:center; border-radius:9px; color:#fff; font-size:.76rem; background:linear-gradient(135deg,#7c6cff,#a855f7); }
          .topbar-right { color:#64748b; font-size:.78rem; padding:.35rem .65rem; background:rgba(244,240,255,.7); border-radius:999px; }

          /* ---------- Hero ---------- */
          .hero { padding: .15rem 0 .95rem; color: #172554; margin: 0; }
          .hero h1 { margin: .12rem 0; font-size: 2rem; letter-spacing: -.8px; color: #172554; }
          .hero p { margin: .25rem 0 0; color: #64748b; font-size: .94rem; }
          .eyebrow { text-transform: uppercase; letter-spacing: 1.2px; font-size: .68rem; font-weight: 800; color: #7c3aed; line-height: 1.5; }
          .hero-tag { display: inline-block; margin-top: .8rem; padding: .25rem .6rem; border-radius: 999px;
                      background: rgba(124,108,255,.16); color: #5145cd; font-size: .75rem; font-weight: 700;
                      border: 1px solid rgba(124,108,255,.25); }

          /* ---------- Flow / info cards ---------- */
          .flow { display: grid; grid-template-columns: repeat(4, 1fr); gap: .6rem; margin: .9rem 0 1.1rem; }
          .flow-card { padding: .7rem .8rem; position: relative; }
          .flow-card:not(:last-child)::after { content: '→'; position: absolute; right: -.55rem; top: 34%; z-index: 2;
                                                color: #7c3aed; font-size: 1.2rem; font-weight: 800; }
          .flow-num { color: #7c3aed; font-weight: 800; font-size: .72rem; letter-spacing: .7px; }
          .flow-title { color: #0f172a; font-weight: 750; font-size: .91rem; margin-top: .12rem; }
          .flow-detail { color: #64748b; font-size: .75rem; margin-top: .12rem; }

          .info-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .7rem; margin: .6rem 0 1rem; }
          .info-card { padding: .85rem 1rem; }
          .info-label { color: #64748b; font-size: .72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .65px; }
          .info-value { color: #0f172a; font-size: 1rem; font-weight: 750; margin-top: .25rem; }
          .info-note { color: #64748b; font-size: .76rem; margin-top: .1rem; }

          /* ---------- Experiment strip / insight ---------- */
          .experiment-strip { padding: .8rem 1rem; color: #475569; margin: .7rem 0 1rem; font-size: .86rem; }
          .experiment-strip strong { color: #5145cd; }
          .insight { padding: .9rem 1rem; color: #155e75; margin: .9rem 0 1rem;
                     background: rgba(214, 255, 244, .5) !important; }

          /* ---------- Metric cards ---------- */
          .metric-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1rem; margin: 0 0 1rem; }
          .metric-card { min-height: 136px; box-sizing: border-box; padding: 1rem 1.05rem; transition: transform .18s ease, box-shadow .18s ease; }
          .metric-card:hover { transform: translateY(-3px); box-shadow: 0 15px 30px rgba(76, 92, 158, .18); }
          .metric-label { color: #64748b; font-size: .76rem; font-weight: 750; }
          .metric-value { color: #172554; font-size: 1.55rem; line-height: 1.25; margin: .45rem 0 .3rem; font-weight: 780; letter-spacing: -.4px; }
          .metric-note { color: #64748b; font-size: .74rem; line-height: 1.32; }
          .metric-positive { display: inline-block; color: #047857; background: rgba(16,185,129,.16); border-radius: 999px; padding: .18rem .45rem; font-size: .72rem; font-weight: 750; }

          .section-title { color: #0f172a; font-size: 1.15rem; font-weight: 750; margin: .9rem 0 .15rem; }
          .section-subtitle { color: #64748b; font-size: .88rem; margin-bottom: .65rem; }

          /* ---------- Tabs ---------- */
          .stTabs [data-baseweb="tab-list"] { gap: .45rem; border-bottom: 1px solid rgba(148,163,184,.35); }
          .stTabs [data-baseweb="tab"] { height: 42px; padding: 0 .9rem; border-radius: 9px 9px 0 0;
                                          color: #64748b; font-weight: 650; }
          .stTabs [aria-selected="true"] { color: #6d28d9 !important; background: rgba(124,108,255,.12); }

          /* ---------- Expander ---------- */
          [data-testid="stExpander"] summary { color: #1e293b !important; font-weight: 650; }

          /* ---------- Audio player ---------- */
          audio { border-radius: 999px; }

          .footer { text-align: center; color: #94a3b8; font-size: .78rem; padding: 1.6rem 0 .35rem; }
          @media (max-width: 850px) {
            .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .flow { grid-template-columns: repeat(2, 1fr); }
            .info-grid { grid-template-columns: 1fr; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def to_float_audio(samples: np.ndarray) -> np.ndarray:
    """Normalise common WAV integer/float types to a mono float signal in [-1, 1]."""
    raw = np.asarray(samples)
    # Normalisasi HARUS dilakukan sebelum audio stereo dirata-ratakan.
    # Jika dirata-ratakan lebih dulu, dtype integer berubah menjadi float dengan
    # nilai PCM mentah (mis. ±32768) dan SNR akan salah terbaca sekitar 0 dB.
    if np.issubdtype(raw.dtype, np.integer):
        scale = max(abs(np.iinfo(raw.dtype).min), np.iinfo(raw.dtype).max)
        values = raw.astype(np.float64) / scale
    else:
        values = raw.astype(np.float64)
    if values.ndim == 2:
        values = values.mean(axis=1)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(values, -1.0, 1.0)


def make_test_signal(sample_rate: int, duration: float, frequency: float) -> np.ndarray:
    time = np.arange(int(sample_rate * duration)) / sample_rate
    # Komponen harmonik membuat bentuk gelombang dan spektrum lebih menarik untuk diamati.
    signal = (
        0.72 * np.sin(2 * np.pi * frequency * time)
        + 0.20 * np.sin(2 * np.pi * 2 * frequency * time)
        + 0.08 * np.sin(2 * np.pi * 3 * frequency * time)
    )
    return signal / np.max(np.abs(signal)) * 0.8


def add_awgn(clean: np.ndarray, noise_percent: float, seed: int) -> np.ndarray:
    if noise_percent == 0:
        return clean.copy()
    signal_rms = np.sqrt(np.mean(clean**2))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, signal_rms * noise_percent / 100, clean.size)
    return np.clip(clean + noise, -1.0, 1.0)


def apply_wiener_filter(signal: np.ndarray, window: int) -> np.ndarray:
    """Reduce local noise with a Wiener filter using an odd-sized sample window."""
    return wiener(signal, mysize=window)


def snr_db(reference: np.ndarray, tested: np.ndarray) -> float:
    error = reference - tested
    numerator = np.mean(reference**2)
    denominator = np.mean(error**2)
    if denominator < 1e-15:
        return float("inf")
    return 10 * np.log10(numerator / denominator)


def wav_bytes(signal: np.ndarray, sample_rate: int) -> bytes:
    clipped = np.clip(signal, -1, 1)
    pcm = (clipped * 32767).astype(np.int16)
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


def plot_waveforms(clean: np.ndarray, noisy: np.ndarray, filtered: np.ndarray, sample_rate: int) -> go.Figure:
    max_points = min(clean.size, int(sample_rate * 0.025))
    time_ms = np.arange(max_points) / sample_rate * 1000
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=time_ms, y=clean[:max_points], mode="lines", name="Sinyal awal", line={"color": "#2563eb", "width": 3}, hovertemplate="Waktu: %{x:.2f} ms<br>Amplitudo: %{y:.3f}<extra>Sinyal awal</extra>"))
    figure.add_trace(go.Scatter(x=time_ms, y=noisy[:max_points], mode="lines", name="Ber-noise", line={"color": "#fb7185", "width": 1.4}, opacity=.72, hovertemplate="Waktu: %{x:.2f} ms<br>Amplitudo: %{y:.3f}<extra>Ber-noise</extra>"))
    figure.add_trace(go.Scatter(x=time_ms, y=filtered[:max_points], mode="lines", name="Hasil filter", line={"color": "#0f9f8a", "width": 3}, hovertemplate="Waktu: %{x:.2f} ms<br>Amplitudo: %{y:.3f}<extra>Hasil filter</extra>"))
    figure.update_layout(
        title={"text": "Domain waktu - 25 ms pertama", "x": .02, "xanchor": "left"},
        height=360, margin={"l": 54, "r": 20, "t": 62, "b": 48},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.35)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": .98},
        hovermode="x unified",
    )
    figure.update_xaxes(title="Waktu (ms)", gridcolor="rgba(148,163,184,.35)", zeroline=False)
    figure.update_yaxes(title="Amplitudo", gridcolor="rgba(148,163,184,.35)", zerolinecolor="#cbd5e1")
    return figure


def plot_spectrum(clean: np.ndarray, noisy: np.ndarray, filtered: np.ndarray, sample_rate: int) -> go.Figure:
    n = min(clean.size, 32768)
    window = np.hanning(n)
    frequencies = np.fft.rfftfreq(n, 1 / sample_rate)

    def magnitude_db(values: np.ndarray) -> np.ndarray:
        return 20 * np.log10(np.maximum(np.abs(np.fft.rfft(values[:n] * window)), 1e-10))

    figure = go.Figure()
    for values, name, color, width, opacity in [(clean, "Awal", "#2563eb", 2.4, 1), (noisy, "Ber-noise", "#fb7185", 1, .72), (filtered, "Terfilter", "#0f9f8a", 2.4, 1)]:
        figure.add_trace(go.Scatter(x=frequencies, y=magnitude_db(values), mode="lines", name=name, line={"color": color, "width": width}, opacity=opacity))
    figure.update_layout(
        title={"text": "Spektrum frekuensi", "x": .02, "xanchor": "left"},
        height=400, margin={"l": 54, "r": 20, "t": 62, "b": 48},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.35)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": .98},
        hovermode="x unified",
    )
    figure.update_xaxes(title="Frekuensi (Hz)", range=[0, min(8000, sample_rate / 2)], gridcolor="rgba(148,163,184,.35)", zeroline=False)
    figure.update_yaxes(title="Magnitudo (dB)", range=[-20, 75], gridcolor="rgba(148,163,184,.35)", zeroline=False)
    return figure


def plot_quality_comparison(before: float, after: float) -> go.Figure:
    """Compact chart that makes the filter's measurable improvement easy to explain."""
    values = [before, after]
    improvement = after - before
    figure = go.Figure(go.Bar(x=["Sebelum filter", "Sesudah filter"], y=values, marker_color=["#fb7185", "#14b8a6"], text=[f"{before:.2f} dB", f"{after:.2f} dB"], textposition="outside", textfont={"size": 14, "color": "#0f172a"}, hovertemplate="%{x}<br>SNR: %{y:.2f} dB<extra></extra>"))
    figure.update_layout(
        title={"text": f"Kualitas sinyal · perubahan {improvement:+.2f} dB", "x": .02, "xanchor": "left"},
        height=360, margin={"l": 26, "r": 20, "t": 62, "b": 46},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.35)", showlegend=False,
    )
    figure.update_yaxes(title="SNR (dB)", range=[min(-5, min(values) - 3), max(values) + 5], gridcolor="rgba(148,163,184,.35)", zeroline=False)
    figure.update_xaxes(fixedrange=True)
    return figure


inject_css()
st.markdown(
    """<div class="topbar"><div class="brand"><div class="brand-mark">NC</div>NoiseClean</div><div class="topbar-right">PSD Lab · Kelompok 6</div></div>
    <div class="hero"><div class="eyebrow">Digital Signal Processing Lab</div><h1>Eksperimen reduksi noise</h1><p>Bandingkan kondisi sinyal sebelum dan sesudah proses filtering secara langsung.</p><span class="hero-tag">Filtering / Noise Reduction</span></div>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Control Lab")
    st.caption("Atur sumber sinyal, noise, dan metode penyaringan.")
    st.markdown('<div class="control-title">Sinyal input</div>', unsafe_allow_html=True)
    source = st.radio("Sumber sinyal", ["Sinyal sintetis", "Unggah WAV"])
    uploaded = None
    if source == "Sinyal sintetis":
        sample_rate = st.selectbox("Sampling rate", [8000, 16000, 44100], index=1)
        duration = st.slider("Durasi (detik)", 1.0, 10.0, 3.0, 0.5)
        frequency = st.slider("Frekuensi fundamental (Hz)", 100, 2000, 440, 10)
    else:
        uploaded = st.file_uploader("File audio WAV", type=["wav"])
        sample_rate, duration, frequency = 16000, 3.0, 440

    st.markdown('<div class="control-title">Noise dan filter</div>', unsafe_allow_html=True)
    noise_percent = st.slider("Level noise (% RMS sinyal)", 0, 100, 30, 1)
    random_seed = st.number_input("Seed noise (hasil konsisten)", min_value=0, value=6, step=1)
    st.markdown('<div class="control-title">Wiener Filter</div>', unsafe_allow_html=True)
    window = st.slider("Ukuran jendela Wiener (ganjil)", 3, 101, 11, 2)
    st.caption("Wiener Filter mengestimasi noise secara lokal. Jendela lebih besar memberi penghalusan lebih kuat.")
    
try:
    if source == "Unggah WAV":
        if uploaded is None:
            st.info("Unggah file WAV dari panel kiri, atau pilih sinyal sintetis untuk memulai.")
            st.stop()
        sample_rate, raw = wavfile.read(uploaded)
        clean_signal = to_float_audio(raw)
        maximum_samples = sample_rate * 30
        if clean_signal.size > maximum_samples:
            clean_signal = clean_signal[:maximum_samples]
            st.warning("Audio dibatasi hingga 30 detik agar visualisasi tetap responsif.")
    else:
        clean_signal = make_test_signal(sample_rate, duration, frequency)

    if clean_signal.size < 50:
        st.error("Sinyal terlalu pendek untuk diproses.")
        st.stop()
    noisy_signal = add_awgn(clean_signal, noise_percent, int(random_seed))
    filtered_signal = apply_wiener_filter(noisy_signal, window)

    before = snr_db(clean_signal, noisy_signal)
    after = snr_db(clean_signal, filtered_signal)
    rmse = np.sqrt(np.mean((clean_signal - filtered_signal) ** 2))
    gain = after - before
    st.markdown(
        f"""<div class="metric-grid">
          <div class="metric-card"><div class="metric-label">SAMPLING RATE</div><div class="metric-value">{sample_rate:,} Hz</div><div class="metric-note">Jumlah sampel audio yang dibaca setiap detik.</div></div>
          <div class="metric-card"><div class="metric-label">SNR SEBELUM FILTER</div><div class="metric-value">{before:.2f} dB</div><div class="metric-note">Kualitas sinyal setelah noise ditambahkan.</div></div>
          <div class="metric-card"><div class="metric-label">SNR SETELAH FILTER</div><div class="metric-value">{after:.2f} dB</div><div class="metric-positive">↑ {gain:+.2f} dB</div><div class="metric-note" style="margin-top:.38rem">Semakin tinggi nilainya, semakin bersih sinyal.</div></div>
          <div class="metric-card"><div class="metric-label">RMSE HASIL FILTER</div><div class="metric-value">{rmse:.4f}</div><div class="metric-note">Rata-rata kesalahan terhadap sinyal awal; lebih kecil lebih baik.</div></div>
        </div>""",
        unsafe_allow_html=True,
    )

    assessment = "meningkatkan" if gain >= 0 else "menurunkan"
    accent = "baik" if gain >= 3 else "masih dapat dioptimalkan"
    st.markdown(
        f"""<div class="insight"><strong>Ringkasan hasil Wiener Filter.</strong> Filter {assessment} SNR sebesar <strong>{abs(gain):.2f} dB</strong>. """
        f"Nilai SNR naik berarti keluaran filter lebih mendekati sinyal awal; nilai RMSE membantu memastikan detail sinyal tidak terlalu berubah.</div>""",
        unsafe_allow_html=True,
    )

    source_name = "Sinyal sintetis" if source == "Sinyal sintetis" else "Audio WAV"
    window_text = f"Jendela {window} sampel"
    st.markdown(
        f"""<div class="experiment-strip"><strong>Eksperimen aktif</strong> &nbsp; {source_name} · {sample_rate:,} Hz · {clean_signal.size / sample_rate:.1f} detik
        &nbsp; | &nbsp; Noise putih {noise_percent}% RMS &nbsp; | &nbsp; Wiener Filter ({window_text})</div>""",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["Dashboard Analisis", "Audio & Ekspor", "Metode dan Interpretasi"])
    with tab1:
        st.markdown('<div class="section-title">Visualisasi hasil pengolahan</div><div class="section-subtitle">Biru = sinyal awal · Merah = sinyal ber-noise · Hijau = keluaran filter</div>', unsafe_allow_html=True)
        chart_left, chart_right = st.columns([1.25, 1])
        with chart_left:
            with st.container(border=True):
                st.plotly_chart(plot_waveforms(clean_signal, noisy_signal, filtered_signal, sample_rate), use_container_width=True, config={"displaylogo": False})
        with chart_right:
            with st.container(border=True):
                st.plotly_chart(plot_quality_comparison(before, after), use_container_width=True, config={"displaylogo": False})
        st.markdown(
            f"""<div class="info-grid">
              <div class="info-card"><div class="info-label">Objek pengujian</div><div class="info-value">{source_name}</div><div class="info-note">Referensi sinyal bersih sebelum penambahan noise.</div></div>
              <div class="info-card"><div class="info-label">Parameter aktif</div><div class="info-value">{window_text}</div><div class="info-note">Dapat diubah dari panel Control Lab.</div></div>
              <div class="info-card"><div class="info-label">Kesimpulan eksperimen</div><div class="info-value">SNR {gain:+.2f} dB</div><div class="info-note">{assessment.capitalize()} kualitas dibanding sinyal ber-noise.</div></div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.caption("SNR dihitung terhadap sinyal awal: 10 log10(P_sinyal / P_error). Nilai lebih tinggi berarti kesalahan terhadap sinyal referensi lebih kecil.")
        with st.expander("Lihat analisis spektrum frekuensi"):
            st.plotly_chart(plot_spectrum(clean_signal, noisy_signal, filtered_signal, sample_rate), use_container_width=True, config={"displaylogo": False})
            st.caption("Spektrum dihitung memakai FFT untuk visualisasi saja; metode reduksi noise dipilih dari panel Control Lab.")
    with tab2:
        audio1, audio2 = st.columns(2)
        with audio1:
            with st.container(border=True):
                st.subheader("Sinyal ber-noise")
                st.audio(wav_bytes(noisy_signal, sample_rate), format="audio/wav")
        with audio2:
            with st.container(border=True):
                st.subheader("Hasil filter")
                st.audio(wav_bytes(filtered_signal, sample_rate), format="audio/wav")
        st.download_button(
            "⬇️ Unduh hasil WAV", wav_bytes(filtered_signal, sample_rate),
            file_name="noiseclean_filtered.wav", mime="audio/wav",
        )
    with tab3:
        st.subheader("Wiener Filter")
        st.latex(r"\hat{X}(f) = \frac{S_{xx}(f)}{S_{xx}(f) + S_{nn}(f)}Y(f)")
        st.write("Wiener Filter mengestimasi sinyal bersih dari sinyal ber-noise dengan meminimalkan mean square error (MSE). Pada aplikasi ini, estimasi dilakukan secara lokal menggunakan ukuran jendela yang dipilih.")
        st.info("Untuk eksperimen yang adil, pertahankan seed noise yang sama saat mengganti level noise atau ukuran jendela.")
        st.markdown("#### Cara membaca hasil")
        st.markdown("- **Domain waktu:** lihat apakah garis hijau mengikuti garis biru dan menjauh dari garis merah.\n- **Spektrum frekuensi:** noise biasanya menaikkan energi di banyak frekuensi; Wiener Filter menekan komponen noise berdasarkan estimasi lokal.\n- **SNR dan RMSE:** ubah level noise atau ukuran jendela, lalu bandingkan hasil antar-pengujian.")
except Exception as error:
    st.error(f"Tidak dapat memproses sinyal: {error}")
    st.exception(error)

st.markdown('<div class="footer">NoiseClean · Aplikasi Pengolahan Sinyal Digital · Kelompok 6</div>', unsafe_allow_html=True)
