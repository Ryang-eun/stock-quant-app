"""
=============================================================
  국내주식 매수 후보 TOP 10 자동 추출기
  작성자: Ryangeun (quant MVP v1.5 - KRX API 직접 호출)
  
  [수정 이력 v1.5]
  - fdr.StockListing() → KRX 공식 REST API 직접 호출로 교체
    이유: Streamlit Cloud IP에서 네이버/FDR 서버 차단 문제
  - fdr.DataReader() → yfinance fallback 추가
  - 네트워크 오류 상세 로깅 추가
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import math
import requests
from datetime import datetime, timedelta
from io import BytesIO

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
.warn-box {
    background:#fff3cd; border-left:5px solid #ffc107;
    padding:12px 16px; border-radius:6px; margin:10px 0;
    font-size:0.88rem; color:#856404;
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
    d = datetime.today() - timedelta(days=offset_days)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime("%Y%m%d")


def safe_float(val, default=np.nan):
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    prices = prices.dropna()
    if len(prices) < period + 1:
        return np.nan
    delta = prices.diff().dropna()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    ag = float(gain.iloc[:period].mean())
    al = float(loss.iloc[:period].mean())
    for i in range(period, len(gain)):
        ag = (ag * (period - 1) + float(gain.iloc[i])) / period
        al = (al * (period - 1) + float(loss.iloc[i])) / period
    if al == 0:
        return 100.0
    return round(100 - 100 / (1 + ag / al), 2)


def is_etf_etn_spac(name: str) -> bool:
    if not isinstance(name, str):
        return False
    keywords = [
        "KODEX","TIGER","KINDEX","KOSEF","ARIRANG","KBSTAR",
        "HANARO","TREX","SOL","ACE","ETN","스팩","SPAC",
        "FOCUS","SMART","TIMEFOLIO","PLUS"
    ]
    return any(k in name.upper() for k in keywords)


def is_preferred(name: str) -> bool:
    if not isinstance(name, str):
        return False
    return any(name.endswith(s) for s in
               ["우", "우B", "우C", "우D", "1우", "2우", "3우"])


# ─────────────────────────────────────────
#  [v1.5 핵심] KRX 공식 API로 종목 리스트 수집
#  - Streamlit Cloud 해외 IP에서도 안정적으로 동작
#  - fdr.StockListing() 완전 대체
# ─────────────────────────────────────────

def get_krx_stock_list(market: str) -> pd.DataFrame:
    """
    KRX 공식 REST API (data.krx.co.kr) 직접 호출
    market: 'KOSPI' 또는 'KOSDAQ'
    반환: code, name, mkt 컬럼 포함 DataFrame
    """
    # KRX API 마켓 코드
    mkt_code = "STK" if market == "KOSPI" else "KSQ"

    url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
    payload = {
        "bld":       "dbms/MDC/STAT/standard/MDCSTAT01901",
        "mktId":     mkt_code,
        "segTpCd":   "ALL",
        "kindStckIndTPDd": "",
        "conditonUner": "0",
    }
    headers = {
        "User-Agent":   "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer":      "http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201020101",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    try:
        resp = requests.post(url, data=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        rows = data.get("OutBlock_1", [])
        if not rows:
            return pd.DataFrame()

        records = []
        for r in rows:
            # KRX API 컬럼명: ISU_SRT_CD(단축코드), ISU_NM(종목명)
            code = str(r.get("ISU_SRT_CD", "")).strip().zfill(6)
            name = str(r.get("ISU_ABBRV", r.get("ISU_NM", ""))).strip()
            if not code or code == "000000":
                continue
            if name in ["nan", "None", "", "NaN"]:
                continue
            records.append({
                "code": code,
                "name": name,
                "mkt":  market,
            })

        return pd.DataFrame(records)

    except Exception as e:
        return pd.DataFrame()


def get_krx_stock_list_v2(market: str) -> pd.DataFrame:
    """
    KRX API 백업 - 다른 엔드포인트 시도
    """
    mkt_code = "1" if market == "KOSPI" else "2"
    url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
    payload = {
        "bld":    "dbms/MDC/STAT/standard/MDCSTAT01501",
        "mktId":  mkt_code,
        "trdDd":  get_business_date(1),
        "money":  "1",
        "csvxls_isNo": "false",
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer":    "http://data.krx.co.kr/",
    }
    try:
        resp = requests.post(url, data=payload, headers=headers, timeout=30)
        data = resp.json()
        rows = data.get("OutBlock_1", [])
        if not rows:
            return pd.DataFrame()

        records = []
        for r in rows:
            code = str(r.get("ISU_SRT_CD", "")).strip().zfill(6)
            name = str(r.get("ISU_ABBRV", r.get("ISU_NM", ""))).strip()
            if not code or code == "000000":
                continue
            if name in ["nan", "None", "", "NaN"]:
                continue
            records.append({"code": code, "name": name, "mkt": market})

        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame()


def get_fdr_stock_list(market: str) -> pd.DataFrame:
    """
    FDR fallback (네트워크 허용 시)
    """
    if not FDR_AVAILABLE:
        return pd.DataFrame()
    try:
        raw = fdr.StockListing(market)
        if raw is None or raw.empty:
            return pd.DataFrame()

        cols = raw.columns.tolist()
        code_col = next((c for c in ["Code","code","Symbol"] if c in cols), cols[0])
        name_col = next((c for c in ["Name","name","종목명"] if c in cols), cols[1])

        records = []
        for idx in range(len(raw)):
            row  = raw.iloc[idx]
            code = str(row[code_col]).strip().zfill(6)
            name = str(row[name_col]).strip()
            if not code or code == "000000":
                continue
            if name in ["nan","None","","NaN"]:
                continue
            records.append({"code": code, "name": name, "mkt": market})

        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def get_ticker_list(test_mode: bool, test_limit: int) -> tuple:
    """
    종목 리스트 수집 - 3단계 fallback
    1순위: KRX API v1
    2순위: KRX API v2
    3순위: FDR StockListing
    """
    records_kospi  = []
    records_kosdaq = []
    errors         = []

    for market in ["KOSPI", "KOSDAQ"]:
        df = pd.DataFrame()

        # 1순위: KRX API v1
        df = get_krx_stock_list(market)
        if not df.empty:
            errors.append(f"{market}: KRX API v1 성공 ({len(df)}개)")
        else:
            errors.append(f"{market}: KRX API v1 실패 → v2 시도")
            # 2순위: KRX API v2
            df = get_krx_stock_list_v2(market)
            if not df.empty:
                errors.append(f"{market}: KRX API v2 성공 ({len(df)}개)")
            else:
                errors.append(f"{market}: KRX API v2 실패 → FDR 시도")
                # 3순위: FDR
                df = get_fdr_stock_list(market)
                if not df.empty:
                    errors.append(f"{market}: FDR 성공 ({len(df)}개)")
                else:
                    errors.append(f"{market}: 모든 소스 실패")
                    continue

        if market == "KOSPI":
            records_kospi  = df.to_dict("records")
        else:
            records_kosdaq = df.to_dict("records")

    # 테스트 모드 슬라이싱
    lim = (test_limit // 2) if test_mode else None
    if lim:
        records_kospi  = records_kospi[:lim]
        records_kosdaq = records_kosdaq[:lim]

    all_records = records_kospi + records_kosdaq

    if not all_records:
        return pd.DataFrame(), errors

    result = pd.DataFrame(all_records)

    # 필수 컬럼 검증
    for col in ["code", "name", "mkt"]:
        if col not in result.columns:
            errors.append(f"필수 컬럼 누락: {col}")
            return pd.DataFrame(), errors

    return result, errors


# ─────────────────────────────────────────
#  OHLCV 수집 (FDR → yfinance fallback)
# ─────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_ohlcv(code: str, fromdate: str, todate: str) -> pd.DataFrame:
    """
    1순위: FDR DataReader
    2순위: yfinance (KRX 티커: {code}.KS / .KQ)
    """
    # 1순위: FDR
    if FDR_AVAILABLE:
        try:
            raw = fdr.DataReader(code, fromdate, todate)
            if raw is not None and not raw.empty:
                rename_map = {
                    "Open": "open", "High": "high", "Low": "low",
                    "Close": "close", "Volume": "volume",
                    "Change": "change", "Adj Close": "adj_close",
                }
                raw = raw.rename(columns=rename_map)
                if "close" in raw.columns and "volume" in raw.columns:
                    raw["amount"] = raw["close"].astype(float) * raw["volume"].astype(float)
                    raw = raw.dropna(subset=["close", "volume"])
                    raw = raw[raw["volume"] > 0]
                    if not raw.empty:
                        return raw
        except Exception:
            pass

    # 2순위: yfinance
    try:
        import yfinance as yf
        for suffix in [".KS", ".KQ"]:
            ticker = code + suffix
            df = yf.download(ticker, start=fromdate[:4]+"-"+fromdate[4:6]+"-"+fromdate[6:],
                             end=todate[:4]+"-"+todate[4:6]+"-"+todate[6:],
                             progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                df.columns = [c.lower() for c in df.columns]
                df = df.rename(columns={"adj close": "adj_close"})
                if "close" in df.columns and "volume" in df.columns:
                    df["amount"] = df["close"].astype(float) * df["volume"].astype(float)
                    df = df.dropna(subset=["close", "volume"])
                    df = df[df["volume"] > 0]
                    if not df.empty:
                        return df
    except Exception:
        pass

    return pd.DataFrame()


# ─────────────────────────────────────────
#  점수 계산
# ─────────────────────────────────────────

def score_chart(close_arr: np.ndarray, vol_arr: np.ndarray) -> dict:
    res = {
        "score": 0, "above_ma20": False, "golden_cross": False,
        "ma60_rising": False, "new_high_20": False, "vol_surge": False,
        "rsi": np.nan, "ma20": np.nan, "ma60": np.nan,
    }
    n = len(close_arr)
    if n < 60:
        return res

    c    = close_arr.astype(float)
    v    = vol_arr.astype(float)
    cur  = float(c[-1])
    ma20 = float(np.mean(c[-20:]))
    ma60 = float(np.mean(c[-60:]))
    res["ma20"] = round(ma20, 0)
    res["ma60"] = round(ma60, 0)

    if cur > ma20:
        res["above_ma20"] = True; res["score"] += 8
    if ma20 > ma60:
        res["golden_cross"] = True; res["score"] += 8
    if n >= 65:
        if float(np.mean(c[-60:])) > float(np.mean(c[-65:-5])):
            res["ma60_rising"] = True; res["score"] += 8
    if cur >= float(np.max(c[-20:])) * 0.99:
        res["new_high_20"] = True; res["score"] += 8
    avg_vol = float(np.mean(v[-20:]))
    if avg_vol > 0 and float(v[-1]) >= avg_vol * 1.5:
        res["vol_surge"] = True; res["score"] += 8

    res["rsi"] = calculate_rsi(pd.Series(c), 14)
    return res


def score_liquidity(avg_eok: float, min_eok: float = 30.0) -> float:
    if not np.isfinite(avg_eok) or avg_eok < min_eok:
        return 0.0
    s = 10.0 * (math.log10(avg_eok) - math.log10(min_eok)) / \
               (math.log10(500.0)   - math.log10(min_eok))
    return round(min(10.0, max(0.0, s)), 1)


# ─────────────────────────────────────────
#  메인 스크리닝
# ─────────────────────────────────────────

def run_screener(params: dict, pbar, stat, logbox) -> pd.DataFrame:
    logs    = []
    results = []
    skipped = 0

    base_date = get_business_date(1)
    from_date = get_business_date(90)

    stat.text("📋 종목 리스트 수집 중...")
    df, errs = get_ticker_list(params["test_mode"], params["test_limit"])
    logs.extend(errs)

    if df is None or df.empty:
        st.error("종목 리스트를 가져올 수 없습니다. 잠시 후 다시 시도해주세요.")
        return pd.DataFrame()

    for col in ["code", "name", "mkt"]:
        if col not in df.columns:
            st.error(f"컬럼 오류: '{col}' 없음. 실제 컬럼: {df.columns.tolist()}")
            return pd.DataFrame()

    logs.append(f"수집 완료: {len(df)}개")

    if params["excl_etf"]:
        before = len(df)
        df = df[~df["name"].apply(is_etf_etn_spac)].reset_index(drop=True)
        logs.append(f"ETF/SPAC 제외: {before} → {len(df)}개")

    if params["excl_pref"]:
        before = len(df)
        df = df[~df["name"].apply(is_preferred)].reset_index(drop=True)
        logs.append(f"우선주 제외: {before} → {len(df)}개")

    total = len(df)
    if total == 0:
        st.warning("필터 후 종목이 없습니다.")
        return pd.DataFrame()

    stat.text(f"📈 기술적 분석 시작... (총 {total}개)")

    for i in range(total):
        row = df.iloc[i]
        try:
            code = str(row["code"])
            name = str(row["name"])
            mkt  = str(row["mkt"])
        except KeyError as ke:
            logs.append(f"[ERR] row{i}: {ke}")
            skipped += 1
            continue

        if not code or code in ["nan", "None"]:
            skipped += 1
            continue
        if not mkt or mkt in ["nan", "None"]:
            mkt = "UNKNOWN"

        pbar.progress(min((i + 1) / total, 1.0))
        if i % 5 == 0:
            stat.text(f"📈 분석 중... ({i+1}/{total}) {name}")

        try:
            ohlcv = get_ohlcv(code, from_date, base_date)
            if ohlcv.empty or len(ohlcv) < 60:
                logs.append(f"[SKIP] {code}({name}): 데이터 부족({len(ohlcv)}일)")
                skipped += 1
                continue

            close_arr = ohlcv["close"].values.astype(float)
            vol_arr   = ohlcv["volume"].values.astype(float)
            amt_arr   = ohlcv["amount"].values.astype(float)

            avg_eok = float(np.mean(amt_arr[-20:])) / 1e8
            if avg_eok < params["min_eok"]:
                skipped += 1
                continue

            chart = score_chart(close_arr, vol_arr)

            if np.isfinite(chart["rsi"]) and chart["rsi"] >= params["rsi_max"]:
                logs.append(f"[SKIP] {code}({name}): RSI {chart['rsi']:.1f}")
                skipped += 1
                continue

            liq_s   = score_liquidity(avg_eok, params["min_eok"])
            total_s = float(chart["score"]) + liq_s
            cur_price = float(close_arr[-1])
            stop_loss = round(cur_price * (1 - params["stop_pct"] / 100))
            target    = round(cur_price * (1 + params["tgt_pct"]  / 100))

            today = datetime.today()
            qmap  = {1: 3, 2: 6, 3: 9, 4: 12}
            q     = (today.month - 1) // 3 + 1
            rd    = datetime(today.year, qmap[q], 30)
            if rd < today:
                nq = (q % 4) + 1
                ny = today.year + (1 if q == 4 else 0)
                rd = datetime(ny, qmap[nq], 30)

            results.append({
                "rank": 0, "code": code, "name": name, "mkt": mkt,
                "price":        int(cur_price),
                "avg_eok":      round(avg_eok, 1),
                "rsi":          round(chart["rsi"], 1) if np.isfinite(chart["rsi"]) else None,
                "ma20":         int(chart["ma20"]) if np.isfinite(chart["ma20"]) else None,
                "ma60":         int(chart["ma60"]) if np.isfinite(chart["ma60"]) else None,
                "above_ma20":   "✅" if chart["above_ma20"]   else "❌",
                "golden_cross": "✅" if chart["golden_cross"]  else "❌",
                "ma60_rising":  "✅" if chart["ma60_rising"]   else "❌",
                "new_high_20":  "✅" if chart["new_high_20"]   else "❌",
                "vol_surge":    "✅" if chart["vol_surge"]     else "❌",
                "chart_score":  chart["score"],
                "liq_score":    liq_s,
                "total_score":  round(total_s, 1),
                "stop_loss":    stop_loss,
                "target":       target,
                "rebal":        rd.strftime("%Y-%m-%d"),
            })

        except Exception as e:
            logs.append(f"[ERR] {code}({name}): {str(e)[:80]}")
            skipped += 1

    pbar.progress(1.0)
    stat.text(f"✅ 완료! 후보 {len(results)}개 / 제외 {skipped}개")

    if logs:
        with logbox.expander(f"📋 처리 로그 ({len(logs)}건)", expanded=False):
            for lg in logs[-50:]:
                st.text(lg)

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    out = out.sort_values("total_score", ascending=False).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    return out


# ─────────────────────────────────────────
#  컬럼 매핑
# ─────────────────────────────────────────

COL_MAP = {
    "rank": "순위", "code": "종목코드", "name": "종목명", "mkt": "시장",
    "price": "현재가", "avg_eok": "20일평균거래대금(억)", "rsi": "RSI",
    "ma20": "MA20", "ma60": "MA60", "above_ma20": "20일선위",
    "golden_cross": "정배열", "ma60_rising": "60일선상승",
    "new_high_20": "20일신고가", "vol_surge": "거래량급증",
    "chart_score": "차트점수(40)", "liq_score": "유동성점수(10)",
    "total_score": "종합점수(50)", "stop_loss": "손절가",
    "target": "목표가", "rebal": "리밸런싱예정일",
}
SHOW_COLS = list(COL_MAP.keys())


def make_excel(df: pd.DataFrame) -> bytes:
    show = df[SHOW_COLS].rename(columns=COL_MAP)
    buf  = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        show.to_excel(w, index=False, sheet_name="매수후보")
        ws = w.sheets["매수후보"]
        for col in ws.columns:
            width = max(len(str(cell.value or "")) for cell in col)
            ws.column_dimensions[col[0].column_letter].width = width + 4
    return buf.getvalue()


# ─────────────────────────────────────────
#  사이드바
# ─────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ 필터 설정")
    st.markdown("### 🔧 분석 모드")
    test_mode = st.toggle("테스트 모드 (빠른 실행)", value=True,
        help="ON: 시장별 상위 종목만 / OFF: 전종목")
    if test_mode:
        test_limit = st.slider("테스트 종목 수", 50, 300, 100, step=50)
        st.info(f"시장별 {test_limit//2}개씩 분석")
    else:
        test_limit = 9999
        st.warning("⏳ 전종목은 수십 분 소요됩니다.")

    st.divider()
    st.markdown("### 📈 기술적 필터")
    min_eok = st.number_input("최소 거래대금 (억원)", 1.0, 500.0, 30.0, 5.0)
    rsi_max = st.number_input("RSI 상한 (과열 제외)", 60.0, 90.0, 70.0, 1.0)

    st.divider()
    st.markdown("### 🎯 매도 기준")
    stop_pct = st.number_input("손절 기준 (%)", 1.0, 20.0, 10.0, 0.5)
    tgt_pct  = st.number_input("목표 수익률 (%)", 5.0, 50.0, 20.0, 1.0)

    st.divider()
    st.markdown("### 🚫 제외 옵션")
    excl_etf  = st.checkbox("ETF/ETN/SPAC 제외", value=True)
    excl_pref = st.checkbox("우선주 제외", value=True)

    st.divider()
    st.markdown("### 🏆 출력 설정")
    top_n = st.slider("TOP N 출력 수", 5, 30, 10, 1)

    run_btn = st.button("🚀 스크리닝 시작", use_container_width=True, type="primary")


# ─────────────────────────────────────────
#  메인 화면
# ─────────────────────────────────────────

st.markdown('<div class="main-title">📈 국내주식 매수 후보 TOP 10 추출기</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-title">코스피 · 코스닥 전종목 | 차트 조건 + 거래대금 복합 스크리닝</div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="warn-box">
⚠️ <b>FDR 데이터 안내:</b> FinanceDataReader는 PER/PBR/ROE를 제공하지 않습니다.<br>
이번 버전은 <b>차트점수(40점) + 유동성점수(10점) = 종합 50점</b> 기준으로 스크리닝합니다.<br>
데이터 소스: KRX 공식 API → FDR → yfinance (순서대로 fallback)
</div>
""", unsafe_allow_html=True)

with st.expander("📋 분석 기준 보기", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📈 차트 조건 (40점, 각 8점)**")
        st.markdown("1. 현재가 > 20일 이동평균\n2. 20일선 > 60일선 (정배열)\n"
                    "3. 60일선 상승 추세\n4. 20일 신고가 돌파 (±1%)\n"
                    "5. 거래량 ≥ 20일 평균 × 1.5")
    with c2:
        st.markdown("**💧 유동성 (10점) + 필터**")
        st.markdown(f"- 20일 평균 거래대금 ≥ **{min_eok}억원**\n"
                    f"- RSI < **{rsi_max}** (과열 제외)\n"
                    f"- 손절가: 매수가 × **{1-stop_pct/100:.2f}**\n"
                    f"- 목표가: 매수가 × **{1+tgt_pct/100:.2f}**")

st.markdown("""
<div class="info-box">
💡 <b>데이터 소스:</b> KRX 공식 API (1순위) → FinanceDataReader (2순위) → yfinance (3순위)
&nbsp;|&nbsp; <b>OHLCV:</b> 최근 3개월 (60영업일 이상 필요)
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  실행
# ─────────────────────────────────────────

if run_btn:
    params = dict(
        test_mode=test_mode, test_limit=test_limit,
        min_eok=min_eok, rsi_max=rsi_max,
        stop_pct=stop_pct, tgt_pct=tgt_pct,
        excl_etf=excl_etf, excl_pref=excl_pref, top_n=top_n,
    )

    st.divider()
    pbar   = st.progress(0)
    stat   = st.empty()
    logbox = st.empty()
    t0     = time.time()

    with st.spinner("스크리닝 진행 중..."):
        result_df = run_screener(params, pbar, stat, logbox)

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
        m1.metric("분석 종목 수",    f"{len(result_df)}개")
        m2.metric("1위 종목",         top_df.iloc[0]["name"])
        m3.metric("1위 점수",         f"{top_df.iloc[0]['total_score']}점")
        m4.metric(f"TOP{top_n} 평균", f"{top_df['total_score'].mean():.1f}점")

        st.divider()
        st.markdown("#### 📌 상위 종목 요약")
        ncards    = min(5, len(top_df))
        card_cols = st.columns(ncards)
        for i in range(ncards):
            r  = top_df.iloc[i]
            sc = r["total_score"]
            color = "#28a745" if sc >= 35 else "#ffc107" if sc >= 20 else "#dc3545"
            with card_cols[i]:
                st.markdown(f"""
<div class="metric-card">
  <div style="font-size:1.3rem;font-weight:bold;color:{color};">{int(r['rank'])}위</div>
  <div style="font-weight:600;margin:4px 0;">{r['name']}</div>
  <div style="color:#666;font-size:0.8rem;">{r['mkt']}</div>
  <div style="font-size:1.1rem;font-weight:bold;color:#1a1a2e;">{int(r['price']):,}원</div>
  <div style="margin-top:6px;"><span class="score-badge">{sc}점</span></div>
  <div style="font-size:0.78rem;margin-top:6px;color:#555;">
    목표: <b>{int(r['target']):,}</b>원<br>
    손절: <b style="color:#dc3545;">{int(r['stop_loss']):,}</b>원
  </div>
</div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("#### 📊 전체 결과 테이블")
        show_df = top_df[SHOW_COLS].rename(columns=COL_MAP).copy()
        st.dataframe(
            show_df.style.format({"현재가": "{:,}", "손절가": "{:,}", "목표가": "{:,}"}, na_rep="-"),
            use_container_width=True, height=420)

        st.divider()
        d1, d2 = st.columns([1, 3])
        with d1:
            st.download_button(
                "📥 엑셀 다운로드", data=make_excel(top_df),
                file_name=f"매수후보TOP{top_n}_{datetime.today().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True)
        with d2:
            st.caption(f"📅 기준일: {get_business_date(1)} | "
                       f"{'🧪 테스트' if test_mode else '🔥 실전'} | "
                       f"후보 {len(result_df)}개 중 TOP{top_n}")

        st.divider()
        st.markdown("#### 🎯 공통 매도 전략")
        st.markdown(f"""
| 구분 | 기준 | 비고 |
|------|------|------|
| 📉 손절가 | 매수가 × **{1-stop_pct/100:.2f}** (-{stop_pct:.0f}%) | 무조건 손절 |
| 📈 익절가 | 매수가 × **{1+tgt_pct/100:.2f}** (+{tgt_pct:.0f}%) | 분할 매도 권장 |
| ⚠️ 추가 검토 | 현재가 **20일선 이탈** 시 | 매도 검토 |
| 🔄 리밸런싱 | **분기 말** (3/6/9/12월) | 조건 탈락 종목 제외 |
""")

st.markdown("""
<div class="disclaimer">
⚠️ <b>투자 위험 고지</b><br>
이 도구는 <b>투자 권유가 아닌 후보 추출 도구</b>입니다.
제공되는 정보는 참고자료이며, 실제 투자 손실에 대한 책임은 투자자 본인에게 있습니다.
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center;font-size:0.75rem;color:#aaa;margin-top:8px;'>"
    "Made by Ryangeun · 국내주식 퀀트 MVP v1.5 · KRX API + FDR + yfinance</p>",
    unsafe_allow_html=True)
