import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go
import pickle
import joblib

# 페이지 기본 설정
st.set_page_config(
    page_title="헬스장 이탈률 분석 대시보드",
    page_icon="💪",
    layout="wide"
)

# 데이터 로드 함수
@st.cache_data
def load_data():
    data = pd.read_csv('data/raw/gym_churn_us.csv')
    return data

# 상관관계 히트맵 생성 함수
def create_correlation_heatmap(data, selected_features):
    corr_matrix = data[selected_features].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='변수 간 상관관계 히트맵',
        height=600,
        width=800
    )
    
    return fig

# 산점도 생성 함수
def create_scatter_plot(data, x_col, y_col, color_by='Churn'):
    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        color=color_by,
        title=f'{x_col} vs {y_col}',
        labels={x_col: x_col, y_col: y_col},
        color_discrete_map={0: 'blue', 1: 'red'}
    )
    fig.update_layout(height=500)
    return fig

# 회원 프로필 분석 함수
def create_profile_plots(data):
    # 연령대별 이탈률
    age_bins = [0, 20, 30, 40, 50, 60, 100]
    age_labels = ['20세 미만', '20-30세', '30-40세', '40-50세', '50-60세', '60세 이상']
    data['AgeGroup'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels)
    age_churn = data.groupby('AgeGroup')['Churn'].mean().reset_index()
    
    fig_age = px.bar(
        age_churn,
        x='AgeGroup',
        y='Churn',
        title='연령대별 이탈률',
        labels={'Churn': '이탈률', 'AgeGroup': '연령대'},
        color='Churn',
        color_continuous_scale='Reds'
    )
    
    # 계약 기간별 이탈률
    contract_churn = data.groupby('Contract_period')['Churn'].mean().reset_index()
    fig_contract = px.bar(
        contract_churn,
        x='Contract_period',
        y='Churn',
        title='계약 기간별 이탈률',
        labels={'Churn': '이탈률', 'Contract_period': '계약 기간'},
        color='Churn',
        color_continuous_scale='Blues'
    )
    
    return fig_age, fig_contract

# 예측 모델 로드 함수
@st.cache_resource
def load_model():
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # 모델 학습
        data = load_data()
        X = data.drop('Churn', axis=1)
        y = data['Churn']
        model.fit(X, y)
        return model
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")
        return None

def main():
    # 제목
    st.title("💪 헬스장 회원 이탈률 분석 대시보드")
    
    # 데이터 로드
    try:
        data = load_data()
        model = load_model()
        st.success("데이터 및 모델 로드 완료!")
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        return
    
    # 탭 생성
    tab1, tab2, tab3 = st.tabs(["📊 상관관계 분석", "🎯 이탈률 예측", "👥 회원 프로필 분석"])
    
    # 탭 1: 상관관계 분석
    with tab1:
        st.header("📊 변수 간 상관관계 분석")
        
        # 수치형 변수 목록
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # 변수 선택 (다중 선택)
        selected_features = st.multiselect(
            "상관관계 분석을 위한 변수 선택",
            options=numeric_columns,
            default=numeric_columns[:5]
        )
        
        if len(selected_features) > 1:
            # 상관관계 히트맵
            correlation_fig = create_correlation_heatmap(data, selected_features)
            st.plotly_chart(correlation_fig, use_container_width=True)
            
            # 산점도 분석
            st.subheader("🎯 산점도 분석")
            col1, col2 = st.columns(2)
            
            with col1:
                x_var = st.selectbox("X축 변수 선택", selected_features)
            with col2:
                y_var = st.selectbox("Y축 변수 선택", 
                                   [col for col in selected_features if col != x_var],
                                   index=min(1, len(selected_features)-1))
            
            scatter_fig = create_scatter_plot(data, x_var, y_var)
            st.plotly_chart(scatter_fig, use_container_width=True)
            
            # 기초 통계량
            st.subheader("📊 선택된 변수들의 기초 통계량")
            st.write(data[selected_features].describe())
        else:
            st.warning("최소 2개 이상의 변수를 선택해주세요!")
    
    # 탭 2: 이탈률 예측
    with tab2:
        st.header("🎯 회원 이탈 예측")
        st.write("회원의 정보를 입력하여 이탈 가능성을 예측해보세요.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("나이", min_value=0, max_value=100, value=30)
            lifetime = st.number_input("회원 기간 (개월)", min_value=0, max_value=120, value=12)
            contract_period = st.selectbox("계약 기간", options=[1, 3, 6, 12])
            avg_class_frequency = st.number_input("주간 평균 방문 횟수", min_value=0.0, max_value=7.0, value=3.0)
        
        with col2:
            avg_additional_charges = st.number_input("월 평균 추가 지출 (USD)", min_value=0.0, value=50.0)
            group_visits = st.checkbox("그룹 수업 참여")
            near_location = st.checkbox("집/직장이 근처에 있음")
            partner = st.checkbox("제휴사 직원")
        
        if st.button("이탈 가능성 예측"):
            # 예측을 위한 데이터 준비
            input_data = pd.DataFrame({
                'Age': [age],
                'Lifetime': [lifetime],
                'Contract_period': [contract_period],
                'Avg_class_frequency_total': [avg_class_frequency],
                'Avg_additional_charges_total': [avg_additional_charges],
                'Group_visits': [int(group_visits)],
                'Near_Location': [int(near_location)],
                'Partner': [int(partner)]
            })
            
            # 예측
            prediction = model.predict_proba(input_data)[0]
            
            # 결과 표시
            st.subheader("예측 결과")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("이탈 가능성", f"{prediction[1]:.1%}")
            with col2:
                risk_level = "높음 🔴" if prediction[1] > 0.7 else "중간 🟡" if prediction[1] > 0.3 else "낮음 🟢"
                st.metric("위험 수준", risk_level)
            
            # 위험 수준에 따른 제안
            st.subheader("💡 추천 사항")
            if prediction[1] > 0.7:
                st.error("⚠️ 이탈 위험이 매우 높습니다!")
                st.write("- 1:1 상담을 통한 회원 만족도 조사 실시")
                st.write("- 맞춤형 프로모션 또는 할인 혜택 제공")
                st.write("- PT 무료 체험 세션 제공")
            elif prediction[1] > 0.3:
                st.warning("⚠️ 이탈 위험이 있습니다.")
                st.write("- 그룹 수업 참여 권장")
                st.write("- 운동 목표 재설정 및 동기부여 프로그램 제공")
                st.write("- 회원 전용 이벤트 초대")
            else:
                st.success("✅ 이탈 위험이 낮습니다.")
                st.write("- 현재 운동 루틴 유지 권장")
                st.write("- 장기 회원 보상 프로그램 안내")
                st.write("- 추천인 프로그램 참여 제안")
    
    # 탭 3: 회원 프로필 분석
    with tab3:
        st.header("👥 회원 프로필 분석")
        
        # 연령대별 이탈률
        fig_age, fig_contract = create_profile_plots(data)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_age, use_container_width=True)
        with col2:
            st.plotly_chart(fig_contract, use_container_width=True)
        
        # 추가 통계
        st.subheader("📊 주요 통계")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("전체 이탈률", f"{data['Churn'].mean():.1%}")
        with col2:
            st.metric("평균 회원 기간", f"{data['Lifetime'].mean():.1f}개월")
        with col3:
            st.metric("그룹 수업 참여율", f"{data['Group_visits'].mean():.1%}")

if __name__ == "__main__":
    main()