"""
=============================================================
  🇰🇷 국내주식 매수 후보 TOP 10 자동 추출기
  작성자: Ryangeun (quant MVP v1.1)
  분석 대상: 코스피 / 코스닥 전종목
  데이터 출처: FinanceDataReader (Python 3.14 완전 호환)
  변경: pykrx 완전 제거 → FinanceDataReader 단독 사용
=============================================================

⚠️ 이 도구는 투자 권유가 아닌 후보 추출 도구입니다.
   모든 투자 결정과 손실에 대한 책임은 투자자 본인에게 있습니다.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import math
from datetime import datetime, timedelta
from io import BytesIO

# FinanceDataReader - Python 3.14 완전 호환 (API Key 불필요)
try:
    import FinanceDataReader as fdr
    FDR_AVAILABLE = True
except ImportError:
    FDR_AVAILABLE = False

# ─────────────────────────────────────────
#  Streamlit 페이지 기본 설정
# ─────────────────────────────────────────
st.set_page_config(
    page_title="국내주식 매수 후보 TOP 10",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not FDR_AVAILABLE:
    st.error("FinanceDataReader가 설치되지 않았습니다. requirements.txt를 확인하세요.")
    st.stop()

# ─────────────────────────────────────────
#  CSS 스타일
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 800; color: #1a1a2e;
        text-align: center; padding: 10px 0;
    }
    .sub-title {
        font-size: 1rem; color: #555; text-align: center; margin-bottom: 20px;
    }
    .score-badge {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; padding: 4px 10px; border-radius: 12px;
        font-weight: bold; font-size: 0.85rem;
    }
    .info-box {
        background: #d1ecf1; border-left: 5px solid #17a2b8;
        padding: 12px 16px; border-radius: 6px; margin: 10px 0;
        font-size: 0.88rem; color: #0c5460;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px; padding: 12px;
        text-align: center; border: 1px solid #e0e0e0;
    }
    .disclaimer {
        background: #f8d7da; border: 1px solid #f5c6cb;
        border-radius: 8px; padding: 14px; margin-top: 30px;
        font-size: 0.82rem; color: #721c24; text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  유틸리티 함수
# ─────────────────────────────────────────

def get_business_date(offset_days: int = 0) -> str:
    """오늘 또는 N일 전 영업일 날짜를 YYYYMMDD 형식으로 반환"""
    date = datetime.today() - timedelta(days=offset_days)
    while date.weekday() >= 5:
        date -= timedelta(days=1)
    return date.strftime("%Y%m%d")


def safe_float(val, default=np.nan):
    """안전하게 float 변환 (실패 시 default 반환)"""
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def calculate_rsi(close_series: pd.Series, period: int = 14) -> float:
    """
    RSI 직접 계산 (Wilder's Smoothing Method)
    데이터가 period+1개 미만이면 NaN 반환
    """
    if len(close_series) < period + 1:
        return np.nan

    delta = close_series.diff().dropna()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.iloc[:period].mean()
    avg_loss = loss.iloc[:period].mean()

    for i in range(period, len(gain)):
        avg_gain = (avg_gain * (period - 1) + gain.iloc[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss.iloc[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def is_etf_etn_spac(name: str) -> bool:
    """ETF, ETN, SPAC 여부 판단 (종목명 키워드 기반)"""
    if not isinstance(name, str):
        return False
    keywords = ["KODEX", "TIGER", "KINDEX", "KOSEF", "ARIRANG", "KBSTAR",
                "HANARO", "TREX", "SOL", "ACE", "ETN", "스팩", "SPAC",
                "FOCUS", "SMART", "TIMEFOLIO", "PLUS", "KB스타", "미래에셋"]
    name_upper = name.upper()
    return any(kw in name_upper for kw in keywords)


def is_preferred_stock(name: str) -> bool:
    """우선주 여부 판단 (종목명 끝 suffix 기반)"""
    if not isinstance(name, str):
        return False
    suffixes = ["우", "우B", "우C", "우D", "1우", "2우", "3우"]
    return any(name.endswith(s) for s in suffixes)


# ─────────────────────────────────────────
#  데이터 수집 함수 (FinanceDataReader 기반)
# ─────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_ticker_list(test_mode: bool, test_limit: int) -> tuple:
    """
    FinanceDataReader로 코스피 + 코스닥 종목 리스트 수집
    FDR 컬럼: Code, Name, Sector, Industry, ListingDate 등
    반환: (DataFrame[티커, 종목명, 시장, PER, PBR, EPS, BPS], errors)
    """
    records = []
    errors = []

    for market in ["KOSPI", "KOSDAQ"]:
        try:
            df = fdr.StockListing(market)

            # FDR 컬럼명이 버전마다 다를 수 있어서 유연하게 처리
            # 코드 컬럼 찾기
            code_col = None
            for c in df.columns:
                if c.upper() in ["CODE", "SYMBOL", "TICKER"]:
                    code_col = c
                    break
            if code_col is None and len(df.columns) > 0:
                code_col = df.columns[0]

            # 이름 컬럼 찾기
            name_col = None
            for c in df.columns:
                if c.upper() in ["NAME", "CORP", "COMPANY", "종목명"]:
                    name_col = c
                    break
            if name_col is None and len(df.columns) > 1:
                name_col = df.columns[1]

            for _, row in df.iterrows():
                code = str(row[code_col]).zfill(6) if code_col else ""
                name = str(row[name_col]) if name_col else ""

                if not code or code == "000000":
                    continue

                rec = {"티커": code, "종목명": name, "시장": market}

                # PER, PBR, EPS, BPS가 있으면 같이 수집
                for fin_col in ["PER", "PBR", "EPS", "BPS"]:
                    matched = [c for c in df.columns if c.upper() == fin_col]
                    if matched:
                        rec[fin_col] = safe_float(row[matched[0]])

                records.append(rec)

        except Exception as e:
            errors.append(f"{market} 종목 리스트 조회 실패: {str(e)[:80]}")

    df_all = pd.DataFrame(records)

    if len(df_all) == 0:
        return df_all, errors

    # 테스트 모드: 시장별 상위 N개만
    if test_mode:
        limit_per_market = test_limit // 2
        df_all = (df_all.groupby("시장", group_keys=False)
                        .apply(lambda x: x.head(limit_per_market))
                        .reset_index(drop=True))

    return df_all, errors


@st.cache_data(ttl=3600, show_spinner=False)
def get_ohlcv_data(ticker: str, fromdate: str, todate: str) -> pd.DataFrame:
    """
    FinanceDataReader로 단일 종목 OHLCV 수집
    FDR 반환 컬럼: Open, High, Low, Close, Volume, Change
    → 한글 컬럼으로 변환 후 거래대금(원) 추가
    """
    try:
        df = fdr.DataReader(ticker, fromdate, todate)

        if df is None or len(df) == 0:
            return pd.DataFrame()

        # 컬럼명 표준화 (영문 → 한글)
        rename_map = {
            "Open": "시가", "High": "고가", "Low": "저가",
            "Close": "종가", "Volume": "거래량", "Adj Close": "수정종가"
        }
        df = df.rename(columns=rename_map)

        # 필수 컬럼 확인
        if "종가" not in df.columns or "거래량" not in df.columns:
            return pd.DataFrame()

        # 거래대금 계산: 종가(원) × 거래량(주) = 원 단위
        df["거래대금"] = df["종가"] * df["거래량"]

        # 결측치 제거
        df = df.dropna(subset=["종가", "거래량"])

        return df

    except Exception:
        return pd.DataFrame()


def calculate_roe(per_val, pbr_val) -> float:
    """
    ROE 역산: ROE = PBR / PER * 100
    FDR에서 EPS/BPS가 없을 때 PER, PBR로 ROE 추정
    ROE = (순이익/자본) = (EPS/BPS) = (P/PER) / (P/PBR) = PBR/PER
    """
    try:
        per = float(per_val)
        pbr = float(pbr_val)
        if per > 0 and pbr > 0:
            return round((pbr / per) * 100, 2)
    except (TypeError, ValueError):
        pass
    return np.nan


# ─────────────────────────────────────────
#  점수 계산 함수
# ─────────────────────────────────────────

def calculate_financial_score(per, pbr, roe,
                               per_max=15, pbr_max=1.5, roe_min=10) -> float:
    """
    재무점수 (50점 만점)
    - PER 점수 (20점): 낮을수록 가점
    - PBR 점수 (15점): 낮을수록 가점
    - ROE 점수 (15점): 높을수록 가점
    """
    score = 0

    # PER 점수 (20점)
    if pd.notna(per) and per > 0:
        per_score = max(0, 20 * (1 - (per - 1) / max(per_max - 1, 1)))
        score += min(20, per_score)

    # PBR 점수 (15점)
    if pd.notna(pbr) and pbr > 0:
        pbr_score = max(0, 15 * (1 - (pbr - 0.5) / max(pbr_max - 0.5, 0.1)))
        score += min(15, pbr_score)

    # ROE 점수 (15점)
    if pd.notna(roe) and roe >= roe_min:
        roe_score = min(15, 15 * (roe - roe_min) / max(30 - roe_min, 1))
        score += roe_score

    return round(score, 1)


def calculate_chart_score(close_series: pd.Series,
                           volume_series: pd.Series) -> dict:
    """
    차트점수 (40점 만점) - 조건별 8점씩
    1. 현재가 > 20일 이동평균
    2. 20일선 > 60일선 (정배열)
    3. 60일선 상승 추세
    4. 20일 신고가 돌파 (±1% 허용)
    5. 거래량 ≥ 20일 평균 × 1.5
    """
    result = {
        "차트점수": 0,
        "20일선위": False,
        "정배열": False,
        "60일선상승": False,
        "신고가": False,
        "거래량급증": False,
        "RSI": np.nan,
        "MA20": np.nan,
        "MA60": np.nan,
    }

    if len(close_series) < 60:
        return result

    close = close_series.values
    volume = volume_series.values

    ma20 = np.mean(close[-20:])
    ma60 = np.mean(close[-60:])
    current_price = close[-1]

    result["MA20"] = round(ma20, 0)
    result["MA60"] = round(ma60, 0)

    # 조건 1: 현재가 > 20일선
    if current_price > ma20:
        result["20일선위"] = True
        result["차트점수"] += 8

    # 조건 2: 20일선 > 60일선 (정배열)
    if ma20 > ma60:
        result["정배열"] = True
        result["차트점수"] += 8

    # 조건 3: 60일선 상승 (과거 대비 최근 60일선이 더 높으면 상승)
    if len(close) >= 65:
        ma60_recent = np.mean(close[-60:])
        ma60_prev = np.mean(close[-65:-5])
        if ma60_recent > ma60_prev:
            result["60일선상승"] = True
            result["차트점수"] += 8

    # 조건 4: 20일 신고가 돌파 (1% 이내 근접도 인정)
    high_20d = np.max(close[-20:])
    if current_price >= high_20d * 0.99:
        result["신고가"] = True
        result["차트점수"] += 8

    # 조건 5: 거래량 급증
    if len(volume) >= 20:
        vol_avg_20 = np.mean(volume[-20:])
        vol_current = volume[-1]
        if vol_avg_20 > 0 and vol_current >= vol_avg_20 * 1.5:
            result["거래량급증"] = True
            result["차트점수"] += 8

    # RSI (14일)
    result["RSI"] = calculate_rsi(pd.Series(close), period=14)

    return result


def calculate_liquidity_score(avg_trade_amount_eok: float,
                               min_amount: float = 30) -> float:
    """
    유동성점수 (10점 만점) - 로그 스케일
    30억 미만: 0점 / 100억: ~4점 / 500억 이상: 10점
    """
    if pd.isna(avg_trade_amount_eok) or avg_trade_amount_eok < min_amount:
        return 0.0

    score = 10 * (math.log10(avg_trade_amount_eok) - math.log10(min_amount)) / \
            (math.log10(500) - math.log10(min_amount))
    return round(min(10.0, max(0.0, score)), 1)


# ─────────────────────────────────────────
#  메인 스크리닝 함수
# ─────────────────────────────────────────

def run_screener(params: dict, progress_bar, status_text,
                 log_container) -> pd.DataFrame:
    """
    전체 스크리닝 파이프라인
    1. 종목 리스트 수집 (FDR)
    2. ETF/우선주 제외
    3. 재무 필터 (PER/PBR/ROE)
    4. OHLCV + 기술적 분석
    5. 거래대금 필터
    6. 점수 계산 → TOP N 추출
    """
    logs = []
    results = []
    skipped = 0

    base_date = get_business_date(1)   # 가장 최근 영업일
    from_date = get_business_date(90)  # 약 3개월 전

    # ── 1. 종목 리스트 수집 ──
    status_text.text("📋 종목 리스트 수집 중... (FDR)")
    ticker_df, list_errors = get_ticker_list(
        params["test_mode"], params["test_limit"]
    )
    logs.extend(list_errors)

    if len(ticker_df) == 0:
        st.error("종목 리스트를 가져올 수 없습니다. 네트워크 연결을 확인하세요.")
        return pd.DataFrame()

    logs.append(f"수집 완료: 총 {len(ticker_df)}개 종목")

    # ── 2. ETF/ETN/SPAC 제외 ──
    if params["exclude_etf"]:
        before = len(ticker_df)
        ticker_df = ticker_df[~ticker_df["종목명"].apply(is_etf_etn_spac)]
        logs.append(f"ETF/SPAC 제외: {before} → {len(ticker_df)}개")

    # ── 3. 우선주 제외 ──
    if params["exclude_preferred"]:
        before = len(ticker_df)
        ticker_df = ticker_df[~ticker_df["종목명"].apply(is_preferred_stock)]
        logs.append(f"우선주 제외: {before} → {len(ticker_df)}개")

    # ── 4. 재무 필터 (FDR에서 수집된 PER/PBR 활용) ──
    status_text.text(f"📊 재무 필터 적용 중... ({len(ticker_df)}개)")

    # PER 컬럼 있으면 재무 필터 적용
    has_fundamental = all(c in ticker_df.columns for c in ["PER", "PBR"])

    if has_fundamental:
        # ROE 계산 (PBR/PER * 100)
        if "ROE" not in ticker_df.columns:
            ticker_df["ROE"] = ticker_df.apply(
                lambda r: calculate_roe(r.get("PER"), r.get("PBR")), axis=1
            )

        # 숫자 변환
        for col in ["PER", "PBR", "ROE"]:
            if col in ticker_df.columns:
                ticker_df[col] = pd.to_numeric(ticker_df[col], errors="coerce")

        pre = len(ticker_df)
        ticker_df = ticker_df.dropna(subset=["PER", "PBR", "ROE"])
        ticker_df = ticker_df[
            (ticker_df["PER"] > 0) & (ticker_df["PER"] <= params["per_max"]) &
            (ticker_df["PBR"] > 0) & (ticker_df["PBR"] <= params["pbr_max"]) &
            (ticker_df["ROE"] >= params["roe_min"])
        ]
        logs.append(f"재무 필터: {pre}개 → {len(ticker_df)}개 통과")
    else:
        # FDR에 재무 데이터 없으면 PER/PBR/ROE NaN으로 유지
        ticker_df["PER"] = np.nan
        ticker_df["PBR"] = np.nan
        ticker_df["ROE"] = np.nan
        logs.append("⚠️ 재무 데이터 없음 - 재무 필터 미적용 (차트/유동성 기준만 적용)")

    total_filtered = len(ticker_df)
    if total_filtered == 0:
        st.warning("재무 필터 통과 종목이 없습니다. PER/PBR/ROE 조건을 완화해보세요.")
        return pd.DataFrame()

    # ── 5. 종목별 OHLCV + 기술적 분석 ──
    status_text.text(f"📈 기술적 분석 시작... (재무 통과 {total_filtered}개)")

    for i, row in enumerate(ticker_df.itertuples(index=False)):
        ticker = row.티커
        name   = row.종목명
        market = row.시장

        progress = (i + 1) / total_filtered
        progress_bar.progress(min(progress, 1.0))
        if i % 5 == 0:
            status_text.text(
                f"📈 분석 중... ({i+1}/{total_filtered}) {name}"
            )

        try:
            # OHLCV 수집
            ohlcv = get_ohlcv_data(ticker, from_date, base_date)

            if len(ohlcv) < 60:
                logs.append(f"[SKIP] {ticker}({name}): 데이터 부족 ({len(ohlcv)}일)")
                skipped += 1
                continue

            close        = ohlcv["종가"]
            volume       = ohlcv["거래량"]
            trade_amount = ohlcv["거래대금"]  # 원 단위

            # 거래대금 필터 (원 → 억원 변환)
            avg_trade_eok = trade_amount.tail(20).mean() / 1e8
            if avg_trade_eok < params["min_trade_eok"]:
                skipped += 1
                continue

            # 차트 점수 계산
            chart = calculate_chart_score(close, volume)

            # RSI 과열 필터
            if pd.notna(chart["RSI"]) and chart["RSI"] >= params["rsi_max"]:
                logs.append(
                    f"[SKIP] {ticker}({name}): RSI 과열 {chart['RSI']:.1f}"
                )
                skipped += 1
                continue

            # 재무 지표
            per = safe_float(getattr(row, "PER", np.nan))
            pbr = safe_float(getattr(row, "PBR", np.nan))
            roe = safe_float(getattr(row, "ROE", np.nan))

            # 점수 계산
            fin_score = calculate_financial_score(
                per, pbr, roe,
                params["per_max"], params["pbr_max"], params["roe_min"]
            )
            liq_score   = calculate_liquidity_score(
                avg_trade_eok, params["min_trade_eok"]
            )
            total_score = fin_score + chart["차트점수"] + liq_score

            # 손절가 / 목표가
            current_price = float(close.iloc[-1])
            stop_loss = round(current_price * (1 - params["stop_loss_pct"] / 100))
            target    = round(current_price * (1 + params["target_pct"] / 100))

            # 리밸런싱 예정일 (분기 말)
            today = datetime.today()
            quarter_end = {1: 3, 2: 6, 3: 9, 4: 12}
            q = (today.month - 1) // 3 + 1
            rebal_month = quarter_end[q]
            rebal_date  = datetime(today.year, rebal_month, 30)
            if rebal_date < today:
                next_q = (q % 4) + 1
                next_year = today.year + (1 if q == 4 else 0)
                rebal_date = datetime(next_year, quarter_end[next_q], 30)

            results.append({
                "종목코드"            : ticker,
                "종목명"              : name,
                "시장"                : market,
                "현재가"              : int(current_price),
                "PER"                 : round(per, 2) if pd.notna(per) else "-",
                "PBR"                 : round(pbr, 2) if pd.notna(pbr) else "-",
                "ROE(%)"              : round(roe, 1) if pd.notna(roe) else "-",
                "20일평균거래대금(억)" : round(avg_trade_eok, 1),
                "RSI"                 : chart["RSI"],
                "MA20"                : chart["MA20"],
                "MA60"                : chart["MA60"],
                "20일선위"            : "✅" if chart["20일선위"]   else "❌",
                "정배열"              : "✅" if chart["정배열"]     else "❌",
                "60일선상승"          : "✅" if chart["60일선상승"]  else "❌",
                "20일신고가"          : "✅" if chart["신고가"]     else "❌",
                "거래량급증"          : "✅" if chart["거래량급증"]  else "❌",
                "재무점수(50)"        : fin_score,
                "차트점수(40)"        : chart["차트점수"],
                "유동성점수(10)"      : liq_score,
                "종합점수(100)"       : round(total_score, 1),
                "손절가"              : stop_loss,
                "목표가"              : target,
                "리밸런싱예정일"      : rebal_date.strftime("%Y-%m-%d"),
            })

        except Exception as e:
            logs.append(f"[ERR] {ticker}({name}): {str(e)[:60]}")
            skipped += 1
            continue

    progress_bar.progress(1.0)
    status_text.text(
        f"✅ 완료! 후보 {len(results)}개 / 제외 {skipped}개"
    )

    # 로그 출력
    if logs:
        with log_container.expander(
            f"📋 처리 로그 ({len(logs)}건)", expanded=False
        ):
            for log in logs[-50:]:
                st.text(log)

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("종합점수(100)", ascending=False)
    result_df.insert(0, "순위", range(1, len(result_df) + 1))

    return result_df


# ─────────────────────────────────────────
#  엑셀 다운로드 헬퍼
# ─────────────────────────────────────────

def to_excel(df: pd.DataFrame) -> bytes:
    """DataFrame → 엑셀 바이트 변환"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="매수후보")
        ws = writer.sheets["매수후보"]
        for col in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            ws.column_dimensions[col[0].column_letter].width = max_len + 4
    return output.getvalue()


# ─────────────────────────────────────────
#  Streamlit 사이드바 (필터 설정)
# ─────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ 필터 설정")

    # ── 모드 선택 ──
    st.markdown("### 🔧 분석 모드")
    test_mode = st.toggle(
        "테스트 모드 (빠른 실행)", value=True,
        help="OFF: 전종목 분석 (시간 많이 소요) | ON: 상위 종목만 분석"
    )
    if test_mode:
        test_limit = st.slider(
            "테스트 종목 수 (코스피+코스닥 합계)", 50, 300, 100, step=50
        )
        st.info(f"시장별 {test_limit//2}개씩 분석합니다.")
    else:
        test_limit = 9999
        st.warning("⏳ 전종목 분석은 수십 분 소요됩니다.")

    st.divider()

    # ── 재무 필터 ──
    st.markdown("### 📊 재무 필터")
    per_max = st.number_input(
        "PER 상한", min_value=1.0, max_value=50.0, value=15.0, step=0.5,
        help="주가수익비율. 낮을수록 저평가 (기본 15배)"
    )
    pbr_max = st.number_input(
        "PBR 상한", min_value=0.1, max_value=10.0, value=1.5, step=0.1,
        help="주가순자산비율. 낮을수록 저평가 (기본 1.5배)"
    )
    roe_min = st.number_input(
        "ROE 하한 (%)", min_value=0.0, max_value=50.0, value=10.0, step=1.0,
        help="자기자본이익률 (기본 10%)"
    )

    st.divider()

    # ── 기술적 필터 ──
    st.markdown("### 📈 기술적 필터")
    min_trade_eok = st.number_input(
        "최소 평균 거래대금 (억원)", min_value=1.0, max_value=500.0,
        value=30.0, step=5.0,
        help="최근 20일 평균 거래대금 하한 (기본 30억원)"
    )
    rsi_max = st.number_input(
        "RSI 상한 (과열 제외)", min_value=60.0, max_value=90.0,
        value=70.0, step=1.0,
        help="RSI가 이 값 이상이면 과열로 제외 (기본 70)"
    )

    st.divider()

    # ── 매도 기준 ──
    st.markdown("### 🎯 매도 기준")
    stop_loss_pct = st.number_input(
        "손절 기준 (%)", min_value=1.0, max_value=20.0, value=10.0, step=0.5,
        help="매수가 대비 하락 시 손절 (기본 -10%)"
    )
    target_pct = st.number_input(
        "목표 수익률 (%)", min_value=5.0, max_value=50.0, value=20.0, step=1.0,
        help="매수가 대비 목표 수익률 (기본 +20%)"
    )

    st.divider()

    # ── 제외 옵션 ──
    st.markdown("### 🚫 제외 옵션")
    exclude_etf       = st.checkbox("ETF/ETN/SPAC 제외", value=True)
    exclude_preferred = st.checkbox("우선주 제외", value=True)

    st.divider()

    # ── 출력 설정 ──
    st.markdown("### 🏆 출력 설정")
    top_n = st.slider("TOP N 출력 수", min_value=5, max_value=30, value=10, step=1)

    run_button = st.button(
        "🚀 스크리닝 시작", use_container_width=True, type="primary"
    )


# ─────────────────────────────────────────
#  메인 화면
# ─────────────────────────────────────────

st.markdown(
    '<div class="main-title">📈 국내주식 매수 후보 TOP 10 추출기</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-title">코스피 · 코스닥 전종목 | PER/PBR/ROE + 차트 조건 + 거래대금 복합 스크리닝</div>',
    unsafe_allow_html=True
)

# 분석 기준 요약
with st.expander("📋 분석 기준 보기", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📊 재무 필터 (50점)**")
        st.markdown(f"""
- PER ≤ **{per_max}배** (낮을수록 ↑)
- PBR ≤ **{pbr_max}배** (낮을수록 ↑)
- ROE ≥ **{roe_min}%** (높을수록 ↑)
""")
    with col2:
        st.markdown("**📈 차트 조건 (40점, 각 8점)**")
        st.markdown("""
1. 현재가 > 20일 이동평균
2. 20일선 > 60일선 (정배열)
3. 60일선 상승 추세
4. 20일 신고가 돌파 (±1%)
5. 거래량 ≥ 20일 평균 × 1.5
""")
    with col3:
        st.markdown("**💧 유동성 (10점) + 기타**")
        st.markdown(f"""
- 20일 평균 거래대금 ≥ **{min_trade_eok}억원**
- RSI < **{rsi_max}** (과열 제외)
- 손절가: 매수가 × **{1 - stop_loss_pct/100:.2f}**
- 목표가: 매수가 × **{1 + target_pct/100:.2f}**
""")

# 데이터 출처 안내
st.markdown("""
<div class="info-box">
💡 <b>데이터 출처:</b> FinanceDataReader (Python 3.14 완전 호환, API Key 불필요)<br>
&nbsp;&nbsp;&nbsp;&nbsp;<b>PER/PBR</b>: FDR StockListing 제공값 | <b>ROE</b>: PBR ÷ PER × 100 역산<br>
&nbsp;&nbsp;&nbsp;&nbsp;<b>OHLCV</b>: FDR DataReader | <b>거래대금</b>: 종가 × 거래량 (원 단위) → 억원 변환
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  스크리닝 실행
# ─────────────────────────────────────────

if run_button:
    params = {
        "test_mode"        : test_mode,
        "test_limit"       : test_limit,
        "per_max"          : per_max,
        "pbr_max"          : pbr_max,
        "roe_min"          : roe_min,
        "min_trade_eok"    : min_trade_eok,
        "rsi_max"          : rsi_max,
        "stop_loss_pct"    : stop_loss_pct,
        "target_pct"       : target_pct,
        "exclude_etf"      : exclude_etf,
        "exclude_preferred": exclude_preferred,
        "top_n"            : top_n,
    }

    st.divider()

    progress_bar  = st.progress(0)
    status_text   = st.empty()
    log_container = st.empty()
    start_time    = time.time()

    with st.spinner("스크리닝 진행 중..."):
        result_df = run_screener(
            params, progress_bar, status_text, log_container
        )

    elapsed = time.time() - start_time

    if result_df is None or len(result_df) == 0:
        st.warning("조건을 충족하는 종목이 없습니다. 필터 기준을 완화해보세요.")
    else:
        top_df = result_df.head(top_n).copy()

        # 요약 메트릭
        st.markdown(
            f"### 🏆 매수 후보 TOP {min(top_n, len(top_df))}  "
            f"<span style='font-size:0.8rem; color:#888;'>({elapsed:.0f}초 소요)</span>",
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("분석 종목 수", f"{len(result_df)}개")
        with c2: st.metric("1위 종목", top_df.iloc[0]["종목명"])
        with c3: st.metric("1위 종합점수", f"{top_df.iloc[0]['종합점수(100)']}점")
        with c4:
            st.metric(
                f"TOP{top_n} 평균점수",
                f"{top_df['종합점수(100)'].mean():.1f}점"
            )

        st.divider()

        # 상위 5개 카드
        st.markdown("#### 📌 상위 종목 요약")
        card_cols = st.columns(min(5, len(top_df)))
        for idx, (_, r) in enumerate(top_df.head(5).iterrows()):
            with card_cols[idx]:
                sc = r["종합점수(100)"]
                color = "#28a745" if sc >= 70 else "#ffc107" if sc >= 50 else "#dc3545"
                st.markdown(f"""
<div class="metric-card">
  <div style="font-size:1.3rem;font-weight:bold;color:{color};">{int(r['순위'])}위</div>
  <div style="font-weight:600;margin:4px 0;">{r['종목명']}</div>
  <div style="color:#666;font-size:0.8rem;">{r['시장']}</div>
  <div style="font-size:1.1rem;font-weight:bold;color:#1a1a2e;">{r['현재가']:,}원</div>
  <div style="margin-top:6px;"><span class="score-badge">{sc}점</span></div>
  <div style="font-size:0.78rem;margin-top:6px;color:#555;">
    목표가: <b>{r['목표가']:,}</b>원<br>
    손절가: <b style="color:#dc3545;">{r['손절가']:,}</b>원
  </div>
</div>
""", unsafe_allow_html=True)

        st.divider()

        # 전체 결과 테이블
        st.markdown("#### 📊 전체 결과 테이블")
        display_cols = [
            "순위", "종목코드", "종목명", "시장", "현재가",
            "PER", "PBR", "ROE(%)", "20일평균거래대금(억)",
            "RSI", "20일선위", "정배열", "60일선상승", "20일신고가", "거래량급증",
            "재무점수(50)", "차트점수(40)", "유동성점수(10)", "종합점수(100)",
            "손절가", "목표가", "리밸런싱예정일",
        ]
        display_cols = [c for c in display_cols if c in top_df.columns]

        st.dataframe(
            top_df[display_cols].style.format(
                {"현재가": "{:,}", "손절가": "{:,}", "목표가": "{:,}"},
                na_rep="-"
            ),
            use_container_width=True,
            height=400,
        )

        # 엑셀 다운로드
        st.divider()
        dl1, dl2 = st.columns([1, 3])
        with dl1:
            st.download_button(
                label="📥 엑셀 다운로드",
                data=to_excel(top_df[display_cols]),
                file_name=f"매수후보TOP{top_n}_{datetime.today().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with dl2:
            st.caption(
                f"📅 기준일: {get_business_date(1)} | "
                f"{'🧪 테스트 모드' if test_mode else '🔥 실전 모드'} | "
                f"총 {len(result_df)}개 후보 중 TOP{top_n}"
            )

        # 매도 전략
        st.divider()
        st.markdown("#### 🎯 공통 매도 전략")
        st.markdown(f"""
| 구분 | 기준 | 비고 |
|------|------|------|
| 📉 손절가 | 매수가 × **{(1 - stop_loss_pct/100):.2f}** (-{stop_loss_pct:.0f}%) | 무조건 손절 |
| 📈 익절가 | 매수가 × **{(1 + target_pct/100):.2f}** (+{target_pct:.0f}%) | 분할 매도 권장 |
| ⚠️ 추가 검토 | 현재가가 **20일 이동평균 이탈** 시 | 매도 검토 |
| 🔄 리밸런싱 | **분기 말** (3/6/9/12월 30일) | 조건 탈락 종목 제외 |
""")

# ─────────────────────────────────────────
#  면책 조항
# ─────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
⚠️ <b>투자 위험 고지</b><br>
이 도구는 <b>투자 권유가 아닌 후보 추출 도구</b>입니다.
제공되는 정보는 투자 의사결정의 참고자료일 뿐이며,
실제 투자에 따른 손실에 대해 어떠한 책임도 지지 않습니다.<br>
주식 투자는 원금 손실이 발생할 수 있으며,
모든 투자 결정과 책임은 투자자 본인에게 있습니다.<br>
<b>데이터 출처:</b> FinanceDataReader | <b>분석 기간:</b> 최근 3개월(60영업일)
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center;font-size:0.75rem;color:#aaa;margin-top:8px;'>"
    "Made by Ryangeun · 국내주식 퀀트 MVP v1.1 · FinanceDataReader 기반</p>",
    unsafe_allow_html=True,
)
