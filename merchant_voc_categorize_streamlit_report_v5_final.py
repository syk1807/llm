import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import calendar
import os
import time

# 설정 변수
OUTPUT_FILE = 'voc_raw_classified_fixed.csv'

# 페이지 설정
st.set_page_config(
    page_title="VOC 분석 리포트",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding: 0.8rem;
        background-color: #f8f9fa;
        border-left: 5px solid #1f77b4;
        border-radius: 5px;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .chart-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .comment-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .period-selector {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file_path):
    """데이터 로드 및 전처리"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')

        # 날짜 컬럼 찾기
        date_columns = [
            'voc_start_dt', 'created_at', 'date', 'timestamp', 'created_date',
            'voc_date', 'analysis_date', 'processed_date'
        ]
        date_col = None

        for col in date_columns:
            if col in df.columns:
                date_col = col
                break

        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])

            # 날짜 관련 컬럼 생성
            df['year_month'] = df[date_col].dt.to_period('M')
            df['year'] = df[date_col].dt.year
            df['month'] = df[date_col].dt.month
            df['date_only'] = df[date_col].dt.date

            st.success(f"✅ 날짜 컬럼 '{date_col}' 처리 완료")
        else:
            st.error("❌ 날짜 컬럼을 찾을 수 없습니다.")
            return None, None

        # 감정 컬럼 정규화
        emotion_mappings = {
            'positive': '긍정', 'negative': '부정', 'neutral': '중립',
            'pos': '긍정', 'neg': '부정', 'neu': '중립',
            '1': '긍정', '0': '중립', '-1': '부정'
        }

        if '감정' in df.columns:
            df['감정'] = df['감정'].map(emotion_mappings).fillna(df['감정'])
        elif 'emotion' in df.columns:
            df['감정'] = df['emotion'].map(emotion_mappings).fillna(df['emotion'])
        elif 'sentiment' in df.columns:
            df['감정'] = df['sentiment'].map(emotion_mappings).fillna(df['sentiment'])

        # 영어 컬럼명을 한글로 매핑
        column_mappings = {
            'major_category': '대분류',
            'minor_category': '중분류', 
            'category_major': '대분류',
            'category_minor': '중분류',
            'main_category': '대분류',
            'sub_category': '중분류'
        }

        for eng_col, kor_col in column_mappings.items():
            if eng_col in df.columns and kor_col not in df.columns:
                df[kor_col] = df[eng_col]

        # 25년 데이터만 필터링
        df_2025 = df[df['year'] == 2025]

        # 데이터 정보 표시
        st.info(f"""
        📋 **{OUTPUT_FILE} 분석 데이터:**
        - 전체 데이터: {len(df):,}건
        - 2025년 데이터: {len(df_2025):,}건
        - 분석 기간: {df_2025[date_col].min().strftime('%Y-%m-%d')} ~ {df_2025[date_col].max().strftime('%Y-%m-%d')}
        - 월별 범위: {df_2025['year_month'].min()} ~ {df_2025['year_month'].max()}
        - 감정 분류: {df_2025['감정'].nunique() if '감정' in df_2025.columns else 0}개 유형
        - 대분류: {df_2025['대분류'].nunique() if '대분류' in df_2025.columns else 0}개 카테고리
        """)

        return df_2025, date_col

    except Exception as e:
        st.error(f"❌ 데이터 로드 중 오류 발생: {e}")
        return None, None

def get_filtered_data(df, period_type):
    """기간별 데이터 필터링"""
    if period_type == "현월":
        latest_month = df['year_month'].max()
        return df[df['year_month'] == latest_month], latest_month
    else:  # "25년 누적"
        return df, "2025년 누적"

def create_period_selector():
    """기간 선택기"""
    st.markdown('<div class="period-selector">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        period_type = st.radio(
            "📅 분석 기간 선택",
            ["현월", "25년 누적"],
            horizontal=True,
            help="현월: 가장 최근 월 데이터만 분석 | 25년 누적: 2025년 전체 데이터 분석"
        )

    st.markdown('</div>', unsafe_allow_html=True)
    return period_type

def create_trend_analysis(df):
    """트렌드 분석"""
    st.markdown('<div class="section-header">📈 트렌드 알아보기 (2025년 월별)</div>', unsafe_allow_html=True)

    # 월별 통계 계산
    monthly_stats = []
    for month in sorted(df['year_month'].unique()):
        month_df = df[df['year_month'] == month]

        # VOC 개수 (고유 ID 기준)
        voc_count = month_df['voc_id'].nunique() if 'voc_id' in df.columns else len(month_df)

        # 분류 수
        classification_count = len(month_df)

        # 감정별 비중
        if '감정' in df.columns:
            emotion_counts = month_df['감정'].value_counts()
            total_emotions = emotion_counts.sum()
            positive_ratio = (emotion_counts.get('긍정', 0) / total_emotions * 100) if total_emotions > 0 else 0
            neutral_ratio = (emotion_counts.get('중립', 0) / total_emotions * 100) if total_emotions > 0 else 0
            negative_ratio = (emotion_counts.get('부정', 0) / total_emotions * 100) if total_emotions > 0 else 0
        else:
            positive_ratio = neutral_ratio = negative_ratio = 0

        monthly_stats.append({
            'year_month': month,
            'voc_count': voc_count,
            'classification_count': classification_count,
            'positive_ratio': positive_ratio,
            'neutral_ratio': neutral_ratio,
            'negative_ratio': negative_ratio
        })

    monthly_df = pd.DataFrame(monthly_stats).sort_values('year_month')
    monthly_df['month_str'] = monthly_df['year_month'].astype(str)

    # 서브플롯 생성
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['월별 VOC 수', '월별 분류 수', '감정분류 추이 (%)', '카테고리 분류 추이'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # VOC 수 추이
    fig.add_trace(
        go.Scatter(x=monthly_df['month_str'], y=monthly_df['voc_count'],
                  mode='lines+markers', name='VOC 수', line=dict(color='#1f77b4', width=3)),
        row=1, col=1
    )

    # 분류 수 추이
    fig.add_trace(
        go.Scatter(x=monthly_df['month_str'], y=monthly_df['classification_count'],
                  mode='lines+markers', name='분류 수', line=dict(color='#ff7f0e', width=3)),
        row=1, col=2
    )

    # 감정분류 추이
    colors = {'긍정': '#2ca02c', '중립': '#ffbb78', '부정': '#d62728'}
    fig.add_trace(
        go.Scatter(x=monthly_df['month_str'], y=monthly_df['positive_ratio'],
                  mode='lines+markers', name='긍정(%)', line=dict(color=colors['긍정'], width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=monthly_df['month_str'], y=monthly_df['neutral_ratio'],
                  mode='lines+markers', name='중립(%)', line=dict(color=colors['중립'], width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=monthly_df['month_str'], y=monthly_df['negative_ratio'],
                  mode='lines+markers', name='부정(%)', line=dict(color=colors['부정'], width=2)),
        row=2, col=1
    )

    # 카테고리 분류 추이 (대분류별)
    if '대분류' in df.columns:
        category_trend = df.groupby(['year_month', '대분류']).size().unstack(fill_value=0)
        top_categories = df['대분류'].value_counts().head(3).index

        for i, category in enumerate(top_categories):
            if category in category_trend.columns:
                fig.add_trace(
                    go.Scatter(x=[str(x) for x in category_trend.index], 
                             y=category_trend[category],
                             mode='lines+markers', name=category),
                    row=2, col=2
                )

    fig.update_layout(height=800, showlegend=True, title_text="VOC 트렌드 분석")
    st.plotly_chart(fig, use_container_width=True)

def create_overview(filtered_df, period_label):
    """개요"""
    st.markdown(f'<div class="section-header">📊 {period_label} 개요</div>', unsafe_allow_html=True)

    # 메트릭 계산
    voc_count = filtered_df['voc_id'].nunique() if 'voc_id' in filtered_df.columns else len(filtered_df)
    classification_count = len(filtered_df)

    # 부정 비중
    negative_ratio = 0
    if '감정' in filtered_df.columns:
        emotion_counts = filtered_df['감정'].value_counts()
        negative_ratio = (emotion_counts.get('부정', 0) / emotion_counts.sum() * 100) if emotion_counts.sum() > 0 else 0

    # 가장 많은 대-중분류
    top_category = ""
    if '대분류' in filtered_df.columns and '중분류' in filtered_df.columns:
        category_counts = filtered_df.groupby(['대분류', '중분류']).size()
        if not category_counts.empty:
            top_idx = category_counts.idxmax()
            top_category = f"{top_idx[0]} > {top_idx[1]}"

    # 메트릭 표시
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f'''
        <div class="metric-container">
            <div class="metric-label">VOC 수</div>
            <div class="metric-value">{voc_count:,}</div>
        </div>
        ''', unsafe_allow_html=True)

    with col2:
        st.markdown(f'''
        <div class="metric-container">
            <div class="metric-label">분류 수</div>
            <div class="metric-value">{classification_count:,}</div>
        </div>
        ''', unsafe_allow_html=True)

    with col3:
        st.markdown(f'''
        <div class="metric-container">
            <div class="metric-label">부정 비중</div>
            <div class="metric-value">{negative_ratio:.1f}%</div>
        </div>
        ''', unsafe_allow_html=True)

    with col4:
        st.markdown(f'''
        <div class="metric-container">
            <div class="metric-label">TOP 분류</div>
            <div class="metric-value" style="font-size: 1rem;">{top_category}</div>
        </div>
        ''', unsafe_allow_html=True)

def create_distribution_overview(filtered_df, period_label):
    """분포 상세개요"""
    st.markdown(f'<div class="section-header">🔍 {period_label} 분포개요</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # 감정 분포 파이차트
        if '감정' in filtered_df.columns:
            emotion_counts = filtered_df['감정'].value_counts()
            fig_emotion = px.pie(
                values=emotion_counts.values,
                names=emotion_counts.index,
                title="감정 분포",
                color_discrete_map={'긍정': '#2ca02c', '중립': '#ffbb78', '부정': '#d62728'}
            )
            fig_emotion.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_emotion, use_container_width=True)

    with col2:
        # 카테고리 분포 막대그래프
        if '대분류' in filtered_df.columns:
            category_counts = filtered_df['대분류'].value_counts()
            fig_category = px.bar(
                x=category_counts.values,
                y=category_counts.index,
                orientation='h',
                title="대분류별 분포",
                color=category_counts.values,
                color_continuous_scale='viridis'
            )
            fig_category.update_layout(showlegend=False, xaxis_title="건수", yaxis_title="대분류")
            st.plotly_chart(fig_category, use_container_width=True)

    # 대분류별 중분류 분포 - 선버스트
    if '대분류' in filtered_df.columns and '중분류' in filtered_df.columns:
        st.markdown("### 대분류별 중분류 분포")

        col1, col2 = st.columns([2, 1])

        with col1:
            category_subcategory = filtered_df.groupby(['대분류', '중분류']).size().reset_index(name='건수')

            # TOP 10 대분류만 선택 (25년 누적일 때 너무 복잡해지는 것 방지)
            top_major_categories = filtered_df['대분류'].value_counts().head(10).index
            category_subcategory_filtered = category_subcategory[
                category_subcategory['대분류'].isin(top_major_categories)
            ]

            fig_sunburst = px.sunburst(
                category_subcategory_filtered,
                path=['대분류', '중분류'],
                values='건수',
                title=f'대분류별 중분류 분포 (TOP 10 대분류)',
                color='건수',
                color_continuous_scale='viridis'
            )
            fig_sunburst.update_layout(height=500)
            st.plotly_chart(fig_sunburst, use_container_width=True)

        with col2:
            # 대분류별 중분류 비중 상세 수치
            st.markdown("### 📊 중분류 비중 상세")

            top_categories = filtered_df['대분류'].value_counts().head(5).index

            for category in top_categories:
                cat_data = filtered_df[filtered_df['대분류'] == category]
                subcat_counts = cat_data['중분류'].value_counts()

                st.markdown(f"**{category}**")

                # 수치 표시
                total = subcat_counts.sum()
                for subcat, count in subcat_counts.head(3).items():  # TOP 3만 표시
                    ratio = (count / total * 100) if total > 0 else 0
                    st.write(f"• {subcat}: {ratio:.1f}% ({count:,}건)")

                if len(subcat_counts) > 3:
                    st.write(f"• 기타 {len(subcat_counts)-3}개: {((subcat_counts.iloc[3:].sum() / total * 100) if total > 0 else 0):.1f}%")

                st.markdown("---")

def create_detailed_analysis_table(filtered_df, period_label):
    """상세분석 테이블"""
    st.markdown(f'<div class="section-header">📋 {period_label} 상세분석테이블</div>', unsafe_allow_html=True)

    # 필터 설정
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        emotion_filter = st.selectbox(
            "감정 필터",
            ['전체'] + list(filtered_df['감정'].unique()) if '감정' in filtered_df.columns else ['전체']
        )

    with col2:
        category_filter = st.selectbox(
            "대분류 필터",
            ['전체'] + list(filtered_df['대분류'].unique()) if '대분류' in filtered_df.columns else ['전체']
        )

    with col3:
        if '대분류' in filtered_df.columns and '중분류' in filtered_df.columns:
            # 대-중분류 조합 필터
            category_combinations = filtered_df['대분류'].astype(str) + ' > ' + filtered_df['중분류'].astype(str)
            combo_filter = st.selectbox(
                "대-중분류 필터",
                ['전체'] + list(category_combinations.unique())
            )
        else:
            combo_filter = '전체'

    with col4:
        sort_order = st.selectbox(
            "정렬",
            ['분류건수 내림차순', '분류건수 오름차순']
        )

    # 필터 적용
    table_df = filtered_df.copy()

    if emotion_filter != '전체' and '감정' in table_df.columns:
        table_df = table_df[table_df['감정'] == emotion_filter]

    if category_filter != '전체' and '대분류' in table_df.columns:
        table_df = table_df[table_df['대분류'] == category_filter]

    if combo_filter != '전체':
        parts = combo_filter.split(' > ')
        if len(parts) == 2:
            table_df = table_df[
                (table_df['대분류'] == parts[0]) & 
                (table_df['중분류'] == parts[1])
            ]

    # 집계 및 정렬
    if len(table_df) > 0:
        required_cols = []
        if '감정' in table_df.columns:
            required_cols.append('감정')
        if '대분류' in table_df.columns:
            required_cols.append('대분류')
        if '중분류' in table_df.columns:
            required_cols.append('중분류')

        if required_cols:
            summary_table = table_df.groupby(required_cols).agg({
                'voc_id': 'nunique' if 'voc_id' in table_df.columns else 'count',
                table_df.columns[0]: 'count'
            }).rename(columns={
                'voc_id': 'VOC수' if 'voc_id' in table_df.columns else '분류건수',
                table_df.columns[0]: '분류건수'
            })

            if 'voc_id' in table_df.columns:
                summary_table.columns = ['VOC수', '분류건수']
            else:
                summary_table = summary_table[['분류건수']]

            summary_table = summary_table.reset_index()

            # 정렬
            sort_col = '분류건수'
            ascending = sort_order == '분류건수 오름차순'
            summary_table = summary_table.sort_values(sort_col, ascending=ascending)

            # 비율 추가
            total_count = summary_table['분류건수'].sum()
            summary_table['비율(%)'] = (summary_table['분류건수'] / total_count * 100).round(1)

            st.dataframe(
                summary_table,
                use_container_width=True,
                height=400
            )

            # 요약 정보
            st.info(f"📊 필터 결과: {len(summary_table)}개 조합, 총 {total_count:,}건")
        else:
            st.warning("분류할 수 있는 컬럼이 없습니다.")
    else:
        st.warning("필터 조건에 맞는 데이터가 없습니다.")

def create_summary_statistics(filtered_df, period_label):
    """요약통계"""
    st.markdown(f'<div class="section-header">📊 {period_label} 요약통계</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏆 TOP 10 대-중분류")

        if '대분류' in filtered_df.columns and '중분류' in filtered_df.columns:
            top_combinations = filtered_df.groupby(['대분류', '중분류']).size().sort_values(ascending=False).head(10)

            for i, ((major, minor), count) in enumerate(top_combinations.items(), 1):
                ratio = (count / len(filtered_df) * 100)
                st.write(f"{i}. **{major} > {minor}**: {count:,}건 ({ratio:.1f}%)")
        else:
            st.info("대분류 또는 중분류 데이터가 없습니다.")

    with col2:
        st.markdown(f"### 💡 {period_label} 종합 코멘트")

        # 자동 코멘트 생성
        comments = []

        # VOC 규모 평가
        total_voc = len(filtered_df)
        voc_count = filtered_df['voc_id'].nunique() if 'voc_id' in filtered_df.columns else total_voc

        if period_label == "2025년 누적":
            comments.append(f"📈 2025년 총 {voc_count:,}건의 VOC가 {total_voc:,}건의 분류로 처리되었습니다.")
        else:
            if total_voc > 1000:
                comments.append(f"📈 이번 달 VOC가 {total_voc:,}건으로 상당히 많은 편입니다.")
            elif total_voc > 500:
                comments.append(f"📊 이번 달 VOC가 {total_voc:,}건으로 보통 수준입니다.")
            else:
                comments.append(f"📉 이번 달 VOC가 {total_voc:,}건으로 비교적 적은 편입니다.")

        # 감정 분석
        if '감정' in filtered_df.columns:
            emotion_counts = filtered_df['감정'].value_counts()
            negative_ratio = (emotion_counts.get('부정', 0) / emotion_counts.sum() * 100) if emotion_counts.sum() > 0 else 0

            if negative_ratio > 60:
                comments.append(f"⚠️ 부정 감정 비율이 {negative_ratio:.1f}%로 높아 개선이 시급합니다.")
            elif negative_ratio > 40:
                comments.append(f"📍 부정 감정 비율이 {negative_ratio:.1f}%로 주의가 필요합니다.")
            else:
                comments.append(f"✅ 부정 감정 비율이 {negative_ratio:.1f}%로 양호한 편입니다.")

        # 주요 이슈 분야
        if '대분류' in filtered_df.columns:
            top_category = filtered_df['대분류'].value_counts().index[0]
            top_count = filtered_df['대분류'].value_counts().iloc[0]
            category_ratio = (top_count / len(filtered_df) * 100)

            comments.append(f"🎯 '{top_category}' 분야가 {category_ratio:.1f}%로 가장 높은 비중을 차지합니다.")

        # 개선 제안
        if '감정' in filtered_df.columns and '대분류' in filtered_df.columns:
            negative_by_category = filtered_df[filtered_df['감정'] == '부정']['대분류'].value_counts()
            if not negative_by_category.empty:
                top_negative_category = negative_by_category.index[0]
                comments.append(f"💡 '{top_negative_category}' 분야의 부정 피드백 개선을 우선 검토하시기 바랍니다.")

        # 코멘트 표시
        for comment in comments:
            st.markdown(f'<div class="comment-box">{comment}</div>', unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">📊 VOC 분석 리포트</h1>', unsafe_allow_html=True)

    # 사이드바
    st.sidebar.header("📋 분석 설정")

    # OUTPUT_FILE 자동 로딩 상태 표시
    with st.sidebar:
        st.markdown(f"### 🤖 {OUTPUT_FILE} 로딩")

        with st.spinner(f"{OUTPUT_FILE}을 읽어오는 중..."):
            data_file = OUTPUT_FILE if os.path.exists(OUTPUT_FILE) else None

        if data_file:
            st.success(f"✅ 파일 로드 완료: `{OUTPUT_FILE}`")
            if os.path.exists(data_file):
                st.info(f"📅 최종 업데이트: {datetime.fromtimestamp(os.path.getmtime(data_file)).strftime('%Y-%m-%d %H:%M')}")
                file_size = os.path.getsize(data_file) / 1024  # KB
                st.info(f"📊 파일 크기: {file_size:.1f} KB")
        else:
            st.error(f"❌ {OUTPUT_FILE}을 찾을 수 없습니다.")
            st.markdown(f"""
            **확인 사항:**
            - 현재 디렉토리에 `{OUTPUT_FILE}` 파일이 있는지 확인
            - 파일에 VOC 분석 결과가 올바르게 저장되어 있는지 확인

            **필수 컬럼:**
            - 날짜 컬럼 (voc_start_dt, created_at 등)
            - 감정 (긍정/중립/부정)
            - 대분류, 중분류
            """)
            st.stop()

        # 자동 새로고침 옵션
        st.markdown("### 🔄 자동 업데이트")
        auto_refresh = st.checkbox("5분마다 자동 새로고침", value=False)

        if auto_refresh:
            time.sleep(300)  # 5분 대기
            st.rerun()

        manual_refresh = st.button("🔄 수동 새로고침")
        if manual_refresh:
            st.cache_data.clear()
            st.rerun()

    # 데이터 로드
    result = load_data(data_file)
    if result[0] is not None:
        df, date_col = result
    else:
        st.error(f"{OUTPUT_FILE} 로딩 중 오류가 발생했습니다.")
        st.stop()

    # 기간 선택기
    period_type = create_period_selector()

    # 선택된 기간에 따라 데이터 필터링
    filtered_df, period_label = get_filtered_data(df, period_type)

    # 분석 섹션들
    create_trend_analysis(df)  # 트렌드는 항상 전체 데이터로
    create_overview(filtered_df, period_label)
    create_distribution_overview(filtered_df, period_label)
    create_detailed_analysis_table(filtered_df, period_label)
    create_summary_statistics(filtered_df, period_label)

    # 추가 인사이트 섹션
    st.markdown("---")
    st.markdown('<div class="section-header">📈 [6] 추가 인사이트</div>', unsafe_allow_html=True)

    # 월별 비교 (25년 누적일 때만 표시)
    if period_type == "25년 누적":
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 월별 VOC 수 비교")
            monthly_voc_counts = df.groupby('year_month')['voc_id'].nunique().reset_index()
            monthly_voc_counts['month_str'] = monthly_voc_counts['year_month'].astype(str)

            fig_monthly = px.bar(
                monthly_voc_counts,
                x='month_str',
                y='voc_id',
                title='월별 VOC 수',
                color='voc_id',
                color_continuous_scale='blues'
            )
            fig_monthly.update_layout(showlegend=False, xaxis_title="월", yaxis_title="VOC 수")
            st.plotly_chart(fig_monthly, use_container_width=True)

        with col2:
            st.markdown("### 📈 월별 부정 감정 비율")
            monthly_negative = df.groupby('year_month').apply(
                lambda x: (x['감정'] == '부정').sum() / len(x) * 100 if '감정' in x.columns else 0
            ).reset_index(name='negative_ratio')
            monthly_negative['month_str'] = monthly_negative['year_month'].astype(str)

            fig_negative = px.line(
                monthly_negative,
                x='month_str',
                y='negative_ratio',
                title='월별 부정 감정 비율 (%)',
                markers=True,
                line_shape='spline'
            )
            fig_negative.update_traces(line_color='#d62728', line_width=3)
            fig_negative.update_layout(xaxis_title="월", yaxis_title="부정 비율 (%)")
            st.plotly_chart(fig_negative, use_container_width=True)

    # 데이터 다운로드 섹션
    with st.expander("📥 데이터 다운로드"):
        st.markdown("### 필터링된 데이터 다운로드")

        # 다운로드할 데이터 준비
        download_cols = ['voc_id', 'year_month', '대분류', '중분류', '감정']
        if 'voc_answer_re' in filtered_df.columns:
            download_cols.append('voc_answer_re')

        available_cols = [col for col in download_cols if col in filtered_df.columns]
        download_df = filtered_df[available_cols]

        # CSV 다운로드
        csv = download_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label=f"📥 {period_label} 데이터 다운로드 (CSV)",
            data=csv,
            file_name=f'voc_analysis_{period_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv'
        )

        # 미리보기
        st.markdown("### 📋 다운로드 데이터 미리보기")
        st.dataframe(download_df.head(10), use_container_width=True)
        st.info(f"총 {len(download_df):,}행 × {len(available_cols)}열")

    # 푸터
    st.markdown("---")
    st.markdown("### 📌 분석 완료")
    if period_type == "25년 누적":
        st.success(f"총 {len(df):,}건의 VOC 데이터를 분석했습니다. ({df['year_month'].min()} ~ {df['year_month'].max()})")
    else:
        latest_month = filtered_df['year_month'].iloc[0] if len(filtered_df) > 0 else "N/A"
        st.success(f"{latest_month} 총 {len(filtered_df):,}건의 VOC 데이터를 분석했습니다.")

    # 분석 시간 표시
    st.markdown(
        f"<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
        f"분석 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        f"</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
