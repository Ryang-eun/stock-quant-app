"""
=============================================================
  국내주식 매수 후보 TOP 10 자동 추출기
  작성자: Ryangeun (quant MVP v1.2)
  데이터: FinanceDataReader (Python 3.14 호환, API Key 불필요)
  변경: 전체 컬럼명 영문 통일 → KeyError/AttributeError 완전 제거
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import math
from datetime import datetime, timedelta
from io import BytesIO

# FinanceDataReader
try:
    import FinanceDataReader as fdr
    FDR_AVAILABLE = True
except ImportError:
    FDR_AVAILABLE = False

# ─────────────────────────────────────────
#  페이지 설정
# ─────────────────────────────────────────
st.set_page_config(
    page_title="국내주식 매수 후보 TOP 10",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not FDR_AVAILABLE:
    st.error("FinanceDataReader가 없습니다. requirements.txt를 확인하세요.")
    st.stop()

# ─────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
.main-title {
    font-size:2rem; font-weight:800; color:#1a1a2e;
    text-align:center; padding:10px 0;
}
.sub-title {
    font-size:1rem; color:#555; text-align:center; margin-bottom:20px;
}
.score-badge {
    background:linear-gradient(135deg,#667eea,#764ba2);
    color:white; padding:4px 10px; border-radius:12px;
    font-weight:bold; font-size:0.85rem;
}
.info-box {
    background:#d1ecf1; border-left:5px solid #17a2b8;
    padding:12px 16px; border-radius:6px; margin:10px 0;
    font-size:0.88rem; color:#0c5460;
}
.metric-card {
    background:#f8f9fa; border-radius:10px; padding:12px;
    text-align:center; border:1px solid #e0e0e0;
}
.disclaimer {
    background:#f8d7da; border:1px solid #f5c6cb;
    border-radius:8px; padding:14px; margin-top:30px;
    font-size:0.82rem; color:#721c24; text-align:center;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  유틸리티 함수
# ─────────────────────────────────────────

def get_business_date(offset_days: int = 0) -> str:
    """N일 전 영업일을 YYYYMMDD로 반환"""
    d = datetime.today() - timedelta(days=offset_days)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime("%Y%m%d")


def safe_float(val, default=np.nan):
    """안전한 float 변환"""
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """RSI 계산 (Wilder's Smoothing)"""
    if len(prices) < period + 1:
        return np.nan
    delta = prices.diff().dropna()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    ag = gain.iloc[:period].mean()
    al = loss.iloc[:period].mean()
    for i in range(period, len(gain)):
        ag = (ag * (period - 1) + gain.iloc[i]) / period
        al = (al * (period - 1) + loss.iloc[i]) / period
    if al == 0:
        return 100.0
    return round(100 - 100 / (1 + ag / al), 2)


def is_etf_etn_spac(name: str) -> bool:
    """ETF/ETN/SPAC 여부 (종목명 키워드)"""
    if not isinstance(name, str):
        return False
    keywords = ["KODEX","TIGER","KINDEX","KOSEF","ARIRANG","KBSTAR",
                "HANARO","TREX","SOL","ACE","ETN","스팩","SPAC",
                "FOCUS","SMART","TIMEFOLIO","PLUS","WOORI","SAMSUNG"]
    return any(k in name.upper() for k in keywords)


def is_preferred(name: str) -> bool:
    """우선주 여부"""
    if not isinstance(name, str):
        return False
    return any(name.endswith(s) for s in ["우","우B","우C","우D","1우","2우","3우"])


# ─────────────────────────────────────────
#  데이터 수집 (모두 영문 컬럼)
# ─────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_ticker_list(test_mode: bool, test_limit: int) -> tuple:
    """
    FDR StockListing → 영문 컬럼 DataFrame 반환
    컬럼: ticker, name, market, per, pbr, roe
    """
    records = []
    errors  = []

    for mkt in ["KOSPI", "KOSDAQ"]:
        try:
            raw = fdr.StockListing(mkt)

            # ── 코드 컬럼 찾기 ──
            code_col = next(
                (c for c in raw.columns if c.upper() in ["CODE","SYMBOL","TICKER"]),
                raw.columns[0]
            )
            # ── 이름 컬럼 찾기 ──
            name_col = next(
                (c for c in raw.columns
                 if c.upper() in ["NAME","CORP","COMPANY"] or c in ["종목명","Name"]),
                raw.columns[1]
            )
            # ── PER/PBR 컬럼 찾기 ──
            per_col = next(
                (c for c in raw.columns if c.upper() == "PER"), None
            )
            pbr_col = next(
                (c for c in raw.columns if c.upper() == "PBR"), None
            )

            for _, r in raw.iterrows():
                code = str(r[code_col]).strip().zfill(6)
                nm   = str(r[name_col]).strip()
                if not code or code == "000000" or nm in ["nan","None",""]:
                    continue

                rec = {
                    "ticker": code,
                    "name":   nm,
                    "market": mkt,
                    "per":    safe_float(r[per_col]) if per_col else np.nan,
                    "pbr":    safe_float(r[pbr_col]) if pbr_col else np.nan,
                }
                # ROE = PBR/PER * 100
                if pd.notna(rec["per"]) and rec["per"] > 0 and pd.notna(rec["pbr"]):
                    rec["roe"] = round(rec["pbr"] / rec["per"] * 100, 2)
                else:
                    rec["roe"] = np.nan

                records.append(rec)

        except Exception as e:
            errors.append(f"{mkt} 조회 실패: {str(e)[:80]}")

    df = pd.DataFrame(records)
    if df.empty:
        return df, errors

    if test_mode:
        lim = test_limit // 2
        df = (df.groupby("market", group_keys=False)
                .apply(lambda x: x.head(lim))
                .reset_index(drop=True))

    return df, errors


@st.cache_data(ttl=3600, show_spinner=False)
def get_ohlcv(ticker: str, fromdate: str, todate: str) -> pd.DataFrame:
    """
    FDR DataReader → 영문 컬럼 DataFrame 반환
    컬럼: open, high, low, close, volume, amount(원)
    """
    try:
        raw = fdr.DataReader(ticker, fromdate, todate)
        if raw is None or raw.empty:
            return pd.DataFrame()

        # FDR 컬럼 → 소문자 영문 통일
        col_map = {
            "Open":"open", "High":"high", "Low":"low",
            "Close":"close", "Volume":"volume",
            "Adj Close":"adj_close", "Change":"change"
        }
        raw = raw.rename(columns=col_map)

        # 필수 컬럼 확인
        if "close" not in raw.columns or "volume" not in raw.columns:
            return pd.DataFrame()

        # 거래대금(원) = 종가 × 거래량
        raw["amount"] = raw["close"] * raw["volume"]
        raw = raw.dropna(subset=["close", "volume"])

        return raw

    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────
#  점수 계산
# ─────────────────────────────────────────

def score_financial(per, pbr, roe, per_max, pbr_max, roe_min) -> float:
    """재무점수 50점 만점"""
    s = 0.0
    if pd.notna(per) and per > 0:
        s += min(20, max(0, 20 * (1 - (per - 1) / max(per_max - 1, 1))))
    if pd.notna(pbr) and pbr > 0:
        s += min(15, max(0, 15 * (1 - (pbr - 0.5) / max(pbr_max - 0.5, 0.1))))
    if pd.notna(roe) and roe >= roe_min:
        s += min(15, 15 * (roe - roe_min) / max(30 - roe_min, 1))
    return round(s, 1)


def score_chart(close_s: pd.Series, vol_s: pd.Series) -> dict:
    """차트점수 40점 만점 (각 8점 × 5개 조건)"""
    res = {
        "score": 0,
        "above_ma20": False, "golden_cross": False,
        "ma60_rising": False, "new_high": False, "vol_surge": False,
        "rsi": np.nan, "ma20": np.nan, "ma60": np.nan,
    }
    if len(close_s) < 60:
        return res

    c  = close_s.values.astype(float)
    v  = vol_s.values.astype(float)
    ma20 = float(np.mean(c[-20:]))
    ma60 = float(np.mean(c[-60:]))
    cur  = float(c[-1])

    res["ma20"] = round(ma20, 0)
    res["ma60"] = round(ma60, 0)

    if cur > ma20:
        res["above_ma20"] = True;  res["score"] += 8
    if ma20 > ma60:
        res["golden_cross"] = True; res["score"] += 8
    if len(c) >= 65 and np.mean(c[-60:]) > np.mean(c[-65:-5]):
        res["ma60_rising"] = True;  res["score"] += 8
    if cur >= float(np.max(c[-20:])) * 0.99:
        res["new_high"] = True;     res["score"] += 8
    if len(v) >= 20:
        avg_v = float(np.mean(v[-20:]))
        if avg_v > 0 and float(v[-1]) >= avg_v * 1.5:
            res["vol_surge"] = True; res["score"] += 8

    res["rsi"] = calculate_rsi(pd.Series(c), 14)
    return res


def score_liquidity(avg_eok: float, min_eok: float = 30) -> float:
    """유동성점수 10점 만점 (로그스케일)"""
    if pd.isna(avg_eok) or avg_eok < min_eok:
        return 0.0
    s = 10 * (math.log10(avg_eok) - math.log10(min_eok)) / \
             (math.log10(500)     - math.log10(min_eok))
    return round(min(10.0, max(0.0, s)), 1)


# ─────────────────────────────────────────
#  메인 스크리닝
# ─────────────────────────────────────────

def run_screener(params, progress_bar, status_text, log_box) -> pd.DataFrame:
    logs    = []
    results = []
    skipped = 0

    base_date = get_business_date(1)
    from_date = get_business_date(90)

    # 1. 종목 리스트
    status_text.text("📋 종목 리스트 수집 중...")
    df, errs = get_ticker_list(params["test_mode"], params["test_limit"])
    logs.extend(errs)

    if df.empty:
        st.error("종목 리스트를 가져올 수 없습니다.")
        return pd.DataFrame()

    logs.append(f"수집: {len(df)}개")

    # 2. ETF/SPAC 제외
    if params["exclude_etf"]:
        before = len(df)
        df = df[~df["name"].apply(is_etf_etn_spac)].reset_index(drop=True)
        logs.append(f"ETF/SPAC 제외: {before}→{len(df)}")

    # 3. 우선주 제외
    if params["exclude_preferred"]:
        before = len(df)
        df = df[~df["name"].apply(is_preferred)].reset_index(drop=True)
        logs.append(f"우선주 제외: {before}→{len(df)}")

    # 4. 재무 필터
    has_fin = df["per"].notna().sum() > 0
    if has_fin:
        before = len(df)
        df = df.dropna(subset=["per","pbr","roe"])
        df = df[
            (df["per"] > 0) & (df["per"] <= params["per_max"]) &
            (df["pbr"] > 0) & (df["pbr"] <= params["pbr_max"]) &
            (df["roe"] >= params["roe_min"])
        ].reset_index(drop=True)
        logs.append(f"재무 필터: {before}→{len(df)}")
    else:
        logs.append("⚠️ 재무 데이터 없음 - 재무 필터 미적용")

    total = len(df)
    if total == 0:
        st.warning("재무 필터 통과 종목 없음. 조건을 완화해보세요.")
        return pd.DataFrame()

    status_text.text(f"📈 기술적 분석 시작... ({total}개)")

    # 5. 종목별 분석 (모두 영문 컬럼 사용)
    for i in range(total):
        row = df.iloc[i]          # Series — 영문 키로만 접근
        ticker = str(row["ticker"])
        name   = str(row["name"])
        market = str(row["market"])
        per    = safe_float(row["per"])
        pbr    = safe_float(row["pbr"])
        roe    = safe_float(row["roe"])

        progress_bar.progress(min((i + 1) / total, 1.0))
        if i % 5 == 0:
            status_text.text(f"📈 분석 중... ({i+1}/{total}) {name}")

        try:
            ohlcv = get_ohlcv(ticker, from_date, base_date)

            if ohlcv.empty or len(ohlcv) < 60:
                logs.append(f"[SKIP] {ticker}({name}): 데이터 부족")
                skipped += 1
                continue

            close_s  = ohlcv["close"].astype(float)
            vol_s    = ohlcv["volume"].astype(float)
            amount_s = ohlcv["amount"].astype(float)

            # 거래대금 필터 (억원)
            avg_eok = float(amount_s.tail(20).mean()) / 1e8
            if avg_eok < params["min_trade_eok"]:
                skipped += 1
                continue

            # 차트 점수
            chart = score_chart(close_s, vol_s)

            # RSI 과열 제외
            if pd.notna(chart["rsi"]) and chart["rsi"] >= params["rsi_max"]:
                logs.append(f"[SKIP] {ticker}({name}): RSI {chart['rsi']:.1f}")
                skipped += 1
                continue

            # 점수 계산
            fin_s  = score_financial(per, pbr, roe,
                                     params["per_max"], params["pbr_max"],
                                     params["roe_min"])
            liq_s  = score_liquidity(avg_eok, params["min_trade_eok"])
            total_s = fin_s + chart["score"] + liq_s

            cur_price = float(close_s.iloc[-1])
            stop_loss = round(cur_price * (1 - params["stop_loss_pct"] / 100))
            target    = round(cur_price * (1 + params["target_pct"]    / 100))

            # 리밸런싱 예정일
            today = datetime.today()
            qmap  = {1:3, 2:6, 3:9, 4:12}
            q     = (today.month - 1) // 3 + 1
            rd    = datetime(today.year, qmap[q], 30)
            if rd < today:
                nq   = (q % 4) + 1
                ny   = today.year + (1 if q == 4 else 0)
                rd   = datetime(ny, qmap[nq], 30)

            results.append({
                "rank":        0,           # 나중에 채움
                "ticker":      ticker,
                "name":        name,
                "market":      market,
                "price":       int(cur_price),
                "per":         round(per, 2) if pd.notna(per) else None,
                "pbr":         round(pbr, 2) if pd.notna(pbr) else None,
                "roe":         round(roe, 1) if pd.notna(roe) else None,
                "avg_eok":     round(avg_eok, 1),
                "rsi":         chart["rsi"],
                "ma20":        chart["ma20"],
                "ma60":        chart["ma60"],
                "above_ma20":  "✅" if chart["above_ma20"]  else "❌",
                "golden_cross":"✅" if chart["golden_cross"] else "❌",
                "ma60_rising": "✅" if chart["ma60_rising"]  else "❌",
                "new_high":    "✅" if chart["new_high"]     else "❌",
                "vol_surge":   "✅" if chart["vol_surge"]    else "❌",
                "fin_score":   fin_s,
                "chart_score": chart["score"],
                "liq_score":   liq_s,
                "total_score": round(total_s, 1),
                "stop_loss":   stop_loss,
                "target":      target,
                "rebal_date":  rd.strftime("%Y-%m-%d"),
            })

        except Exception as e:
            logs.append(f"[ERR] {ticker}({name}): {str(e)[:60]}")
            skipped += 1

    progress_bar.progress(1.0)
    status_text.text(f"✅ 완료! 후보 {len(results)}개 / 제외 {skipped}개")

    if logs:
        with log_box.expander(f"📋 처리 로그 ({len(logs)}건)", expanded=False):
            for lg in logs[-50:]:
                st.text(lg)

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    out = out.sort_values("total_score", ascending=False).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    return out


# ─────────────────────────────────────────
#  엑셀 다운로드
# ─────────────────────────────────────────

# 출력용 한글 컬럼 매핑 (내부 영문 → 표시용 한글)
COL_KO = {
    "rank":         "순위",
    "ticker":       "종목코드",
    "name":         "종목명",
    "market":       "시장",
    "price":        "현재가",
    "per":          "PER",
    "pbr":          "PBR",
    "roe":          "ROE(%)",
    "avg_eok":      "20일평균거래대금(억)",
    "rsi":          "RSI",
    "ma20":         "MA20",
    "ma60":         "MA60",
    "above_ma20":   "20일선위",
    "golden_cross": "정배열",
    "ma60_rising":  "60일선상승",
    "new_high":     "20일신고가",
    "vol_surge":    "거래량급증",
    "fin_score":    "재무점수(50)",
    "chart_score":  "차트점수(40)",
    "liq_score":    "유동성점수(10)",
    "total_score":  "종합점수(100)",
    "stop_loss":    "손절가",
    "target":       "목표가",
    "rebal_date":   "리밸런싱예정일",
}

DISPLAY_COLS = list(COL_KO.keys())


def to_excel(df: pd.DataFrame) -> bytes:
    renamed = df[DISPLAY_COLS].rename(columns=COL_KO)
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        renamed.to_excel(writer, index=False, sheet_name="매수후보")
        ws = writer.sheets["매수후보"]
        for col in ws.columns:
            w = max(len(str(cell.value or "")) for cell in col)
            ws.column_dimensions[col[0].column_letter].width = w + 4
    return buf.getvalue()


# ─────────────────────────────────────────
#  사이드바
# ─────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ 필터 설정")

    st.markdown("### 🔧 분석 모드")
    test_mode = st.toggle("테스트 모드 (빠른 실행)", value=True,
        help="ON: 시장별 상위 종목만 | OFF: 전종목 (수십 분 소요)")
    if test_mode:
        test_limit = st.slider("테스트 종목 수", 50, 300, 100, step=50)
        st.info(f"시장별 {test_limit//2}개씩 분석")
    else:
        test_limit = 9999
        st.warning("⏳ 전종목 분석은 수십 분 소요됩니다.")

    st.divider()
    st.markdown("### 📊 재무 필터")
    per_max = st.number_input("PER 상한", 1.0, 50.0, 15.0, 0.5,
        help="주가수익비율 (기본 15배)")
    pbr_max = st.number_input("PBR 상한", 0.1, 10.0, 1.5, 0.1,
        help="주가순자산비율 (기본 1.5배)")
    roe_min = st.number_input("ROE 하한 (%)", 0.0, 50.0, 10.0, 1.0,
        help="자기자본이익률 (기본 10%)")

    st.divider()
    st.markdown("### 📈 기술적 필터")
    min_trade_eok = st.number_input("최소 거래대금 (억원)", 1.0, 500.0, 30.0, 5.0,
        help="20일 평균 거래대금 하한 (기본 30억원)")
    rsi_max = st.number_input("RSI 상한 (과열 제외)", 60.0, 90.0, 70.0, 1.0,
        help="RSI ≥ 이 값이면 과열로 제외")

    st.divider()
    st.markdown("### 🎯 매도 기준")
    stop_loss_pct = st.number_input("손절 기준 (%)", 1.0, 20.0, 10.0, 0.5)
    target_pct    = st.number_input("목표 수익률 (%)", 5.0, 50.0, 20.0, 1.0)

    st.divider()
    st.markdown("### 🚫 제외 옵션")
    exclude_etf       = st.checkbox("ETF/ETN/SPAC 제외", value=True)
    exclude_preferred = st.checkbox("우선주 제외",        value=True)

    st.divider()
    st.markdown("### 🏆 출력 설정")
    top_n = st.slider("TOP N 출력 수", 5, 30, 10, 1)

    run_button = st.button("🚀 스크리닝 시작",
                           use_container_width=True, type="primary")


# ─────────────────────────────────────────
#  메인 화면
# ─────────────────────────────────────────

st.markdown('<div class="main-title">📈 국내주식 매수 후보 TOP 10 추출기</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">코스피 · 코스닥 전종목 | PER/PBR/ROE + 차트 + 거래대금 복합 스크리닝</div>',
    unsafe_allow_html=True)

with st.expander("📋 분석 기준 보기", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**📊 재무 필터 (50점)**")
        st.markdown(f"- PER ≤ **{per_max}배**\n- PBR ≤ **{pbr_max}배**\n- ROE ≥ **{roe_min}%**")
    with c2:
        st.markdown("**📈 차트 조건 (40점, 각 8점)**")
        st.markdown("1. 현재가 > 20일선\n2. 20일선 > 60일선\n3. 60일선 상승\n4. 20일 신고가\n5. 거래량 1.5배↑")
    with c3:
        st.markdown("**💧 유동성 (10점)**")
        st.markdown(
            f"- 거래대금 ≥ **{min_trade_eok}억원**\n"
            f"- RSI < **{rsi_max}**\n"
            f"- 손절: ×**{1-stop_loss_pct/100:.2f}**\n"
            f"- 목표: ×**{1+target_pct/100:.2f}**"
        )

st.markdown("""
<div class="info-box">
💡 <b>데이터:</b> FinanceDataReader (Python 3.14 호환, API Key 불필요)
&nbsp;|&nbsp; <b>ROE</b>: PBR÷PER×100 역산
&nbsp;|&nbsp; <b>거래대금</b>: 종가×거래량(원)→억원
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  실행
# ─────────────────────────────────────────

if run_button:
    params = dict(
        test_mode=test_mode, test_limit=test_limit,
        per_max=per_max, pbr_max=pbr_max, roe_min=roe_min,
        min_trade_eok=min_trade_eok, rsi_max=rsi_max,
        stop_loss_pct=stop_loss_pct, target_pct=target_pct,
        exclude_etf=exclude_etf, exclude_preferred=exclude_preferred,
        top_n=top_n,
    )

    st.divider()
    pbar      = st.progress(0)
    stat_text = st.empty()
    log_box   = st.empty()
    t0        = time.time()

    with st.spinner("스크리닝 진행 중..."):
        result_df = run_screener(params, pbar, stat_text, log_box)

    elapsed = time.time() - t0

    if result_df is None or result_df.empty:
        st.warning("조건을 충족하는 종목이 없습니다. 필터를 완화해보세요.")
    else:
        top_df = result_df.head(top_n).copy()

        st.markdown(
            f"### 🏆 매수 후보 TOP {min(top_n, len(top_df))}"
            f"<span style='font-size:0.8rem;color:#888;'> ({elapsed:.0f}초)</span>",
            unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("분석 종목 수", f"{len(result_df)}개")
        m2.metric("1위 종목",     top_df.iloc[0]["name"])
        m3.metric("1위 점수",     f"{top_df.iloc[0]['total_score']}점")
        m4.metric(f"TOP{top_n} 평균", f"{top_df['total_score'].mean():.1f}점")

        st.divider()

        # 상위 5개 카드
        st.markdown("#### 📌 상위 종목 요약")
        card_cols = st.columns(min(5, len(top_df)))
        for idx in range(min(5, len(top_df))):
            r  = top_df.iloc[idx]
            sc = r["total_score"]
            color = "#28a745" if sc>=70 else "#ffc107" if sc>=50 else "#dc3545"
            with card_cols[idx]:
                st.markdown(f"""
<div class="metric-card">
  <div style="font-size:1.3rem;font-weight:bold;color:{color};">{int(r['rank'])}위</div>
  <div style="font-weight:600;margin:4px 0;">{r['name']}</div>
  <div style="color:#666;font-size:0.8rem;">{r['market']}</div>
  <div style="font-size:1.1rem;font-weight:bold;color:#1a1a2e;">{r['price']:,}원</div>
  <div style="margin-top:6px;"><span class="score-badge">{sc}점</span></div>
  <div style="font-size:0.78rem;margin-top:6px;color:#555;">
    목표: <b>{r['target']:,}</b>원<br>
    손절: <b style="color:#dc3545;">{r['stop_loss']:,}</b>원
  </div>
</div>""", unsafe_allow_html=True)

        st.divider()

        # 결과 테이블 (한글 컬럼으로 rename해서 표시)
        st.markdown("#### 📊 전체 결과 테이블")
        show_df = top_df[DISPLAY_COLS].rename(columns=COL_KO)
        fmt = {"현재가":"{:,}", "손절가":"{:,}", "목표가":"{:,}"}
        st.dataframe(
            show_df.style.format(fmt, na_rep="-"),
            use_container_width=True, height=400
        )

        # 엑셀 다운로드
        st.divider()
        dl1, dl2 = st.columns([1, 3])
        with dl1:
            st.download_button(
                "📥 엑셀 다운로드",
                data=to_excel(top_df),
                file_name=f"매수후보TOP{top_n}_{datetime.today().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with dl2:
            st.caption(
                f"📅 기준일: {get_business_date(1)} | "
                f"{'🧪 테스트' if test_mode else '🔥 실전'} 모드 | "
                f"후보 {len(result_df)}개 중 TOP{top_n}"
            )

        # 매도 전략
        st.divider()
        st.markdown("#### 🎯 공통 매도 전략")
        st.markdown(f"""
| 구분 | 기준 | 비고 |
|------|------|------|
| 📉 손절가 | 매수가 × **{1-stop_loss_pct/100:.2f}** (-{stop_loss_pct:.0f}%) | 무조건 손절 |
| 📈 익절가 | 매수가 × **{1+target_pct/100:.2f}** (+{target_pct:.0f}%) | 분할 매도 권장 |
| ⚠️ 추가 검토 | 현재가 **20일선 이탈** 시 | 매도 검토 |
| 🔄 리밸런싱 | **분기 말** (3/6/9/12월) | 조건 탈락 종목 제외 |
""")

# ─────────────────────────────────────────
#  면책 조항
# ─────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
⚠️ <b>투자 위험 고지</b><br>
이 도구는 <b>투자 권유가 아닌 후보 추출 도구</b>입니다.
제공되는 정보는 참고자료이며, 실제 투자 손실에 대한 책임은 투자자 본인에게 있습니다.<br>
<b>데이터:</b> FinanceDataReader | <b>분석 기간:</b> 최근 3개월(60영업일)
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center;font-size:0.75rem;color:#aaa;margin-top:8px;'>"
    "Made by Ryangeun · 국내주식 퀀트 MVP v1.2 · FinanceDataReader 기반</p>",
    unsafe_allow_html=True)
