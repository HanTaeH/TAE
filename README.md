# AD 분자 아형 교차 코호트 재현성 연구 (D1)

알츠하이머병(AD) 뇌 전사체 분자 아형(Neff et al. 2021 등)이 AMP-AD 밖의 독립 GEO 코호트에서도 재현되는지 검증하는 분석 코드입니다.

## 데이터 (저장소에 포함하지 않음)

GEO와 논문 보충자료를 직접 내려받아 아래 구조로 둡니다.

```
<data>/GSE33000/GSE33000_series_matrix.txt.gz
<data>/GSE33000/GSE33000_raw_data.txt.gz        # reporterID -> Gene 대응에만 사용
<data>/neff2021/abb5398_data_file_s1.xlsx       # Neff 2021 Data file S1
<data>/neff2021/abb5398_data_file_s2.xlsx       # Neff 2021 Data file S2 (271개 분류 유전자)
```

## 실행

```
pip install -r requirements.txt
cd src
python pilot_gse33000.py --data <data> --out ../results/pilot
```

## 구성

- `src/load_gse33000.py`: GSE33000 발현 행렬과 메타데이터 로드 (AD, 정상만; 헌팅턴병 제외)
- `src/neff_subtypes.py`: Neff 2021 아형 기준(중심값, logFC) 로드
- `src/pilot_gse33000.py`: 실험 0 파일럿. Neff 아형 투영, 순열 검정, go/no-go 판정, 세포 구성 보정 미리보기
- `results/pilot/`: 파일럿 결과 (요약 JSON, 샘플별 배정, 그림)
