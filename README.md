# 2026-PHM-challenge
KSPHM-KIMM 기계 데이터 챌린지 2026 코드 저장소

데이터 샘플링 레이트 (Sampling Rate)
진동 신호: 25.6 kHz
이외 신호: 0.1 Hz
 데이터 수집 주기
10분 주기로 1분씩 취득 (테스트베드는 연속적으로 운전)
 시험 중단 조건과 데이터 특성
베어링이 중단 조건에 도달하면 실험이 종료됨
데이터 측정 중 고장이 발생할 경우, 고장 시점의 데이터가 포함
데이터 미측정 중 고장이 발생할 경우, 고장 시점의 데이터가 불포함
실제 고장 시점과 마지막 데이터 측정 시간은 일치하지 않을 수 있으며,  이는 예측 모델에서 고려되어야 할 중요한 변수임

실험에 활용된 베어링의 상세 규격은 아래와 같습니다.
                           항목
             단위
             값
Bearing Model
n/a
30306
Inner Diameter
mm
30
Outer Diameter
mm
72
Width
mm
20.75
Static Loading Rate
kN
60
Dynamic Loading Rate
kN
59.5
Grease Maximum Temperature
℃
200
베어링 고장 주파수(Bearing Fault Frequency) (1000 RPM 기준)
                                    항목
                     주파수
Ball Passing Frequency Inner (BPFI)
140 Hz
Ball Passing Frequency Outer (BPFO)
93 Hz
Ball Spinning Frequency (BSF)
78 Hz
Cage Rotational Speed
6.7 Hz
3.2 열화시험 조건
                       항목
                               값
축방향 하중
15 kN
반경 방향 하중
10 kN
회전 속도
약 700 - 950rpm (1시간 간격)
3.3 열화시험 중단 조건
아래 2개의 조건 중 1개가 먼저 만족되는 조건
                       항목
                               값
베어링 하우징 온도
200 ℃ 이상
회전 토크
-20 Nm 이하
