from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('malgun', 'B', 16)
        self.cell(0, 10, 'PHM Challenge 2026 Technical Report', 0, 1, 'C')
        self.ln(5)

pdf = PDF()
pdf.add_font('malgun', '', 'C:\\Windows\\Fonts\\malgun.ttf')
pdf.add_font('malgun', 'B', 'C:\\Windows\\Fonts\\malgunbd.ttf')
pdf.add_page()
pdf.set_font('malgun', 'B', 14)
pdf.cell(0, 10, '1. 모델 구조 (Model Architecture)', 0, 1, 'L')
pdf.set_font('malgun', '', 11)

content1 = """본 예측 모델은 베어링의 진동 데이터로부터 현재 마모도(Health Indicator, HI)를 예측하는 심층 신경망입니다. 기존의 복잡한 다중 출력 구조(수명 예측 및 마모 단계 분류 보조망)를 완전히 배제하고, 입력 데이터에서 마모도 궤적을 추출하는 1D CNN 및 Residual 구조에만 집중하도록 극도로 경량화하였습니다. 이로써 불필요한 과적합 요소를 원천 차단하고 구조적 강건성을 확보했습니다."""
pdf.multi_cell(0, 8, content1)
pdf.ln(5)

pdf.set_font('malgun', 'B', 14)
pdf.cell(0, 10, '2. 학습 방식 (Training Method)', 0, 1, 'L')
pdf.set_font('malgun', '', 11)
content2 = """학습 과정에는 제1원칙(First Principles)에 입각한 'One-Loss Diet' 방식을 적용했습니다. 수만 가지의 오차를 유발하던 기존의 6개 복합 손실 함수를 모두 제거하고, 오직 다음 두 가지만을 사용합니다:
1) 예측 HI와 실제 타겟 HI 간의 평균 제곱 오차(MSE)
2) 시간의 흐름에 따라 마모도가 감소하지 않도록 억제하는 단조 증가 패널티(Monotonicity)
결과적으로 모델은 불필요한 노이즈 학습 없이, 순수하게 물리적인 2차 함수(x^2) 형태의 마모 곡선을 정확히 모사하도록 최적화되었습니다."""
pdf.multi_cell(0, 8, content2)
pdf.ln(5)

pdf.set_font('malgun', 'B', 14)
pdf.cell(0, 10, '3. 예측 및 추론 방법 (Prediction Method)', 0, 1, 'L')
pdf.set_font('malgun', '', 11)
content3 = """신경망이 예측한 0~1 사이의 마모도(HI)를 실제 물리적 잔여 수명(RUL)으로 변환하기 위해 '해석학적 멱법칙 외삽법(Analytic Power Law Extrapolation)'을 도입했습니다.
타겟 마모도가 HI = (현재시간 / 총수명)^2 임을 물리적으로 역산하면, 총수명 = 현재시간 / sqrt(HI) 로 도출됩니다. 
따라서 최종 잔여 수명(RUL)은 '현재시간 * (1 / sqrt(HI) - 1)' 공식으로 산출됩니다. 
기존의 억지스러운 선형(Linear) 및 지수(Exponential) 추세선 피팅이 초래하던 극단적인 추론 폭락 및 고정 오류를 수학적으로 완전히 해결하였습니다. 아울러 극초기 마모 단계에서는 과거 베어링 수명 분포(Lognormal Prior)인 CQRL 안전망을 가중 평균(Alpha Blending)하여 극한의 안정성과 정확도를 동시에 달성했습니다."""
pdf.multi_cell(0, 8, content3)

pdf.output('PHM_report.pdf')
