<h2 align="center"> Curation_MI2RL </h2>

## Requirement 설치

```bash
pip install -r requirements.txt
```

## Weight File 다운로드

Weight File [link](https://drive.google.com/drive/folders/1ijTA32hN52WjRORh7KmdaQ_jrxyDc4qw?usp=sharing).

## 코드의 경로 설정

DCM_DIR: 원본 DICOM image 폴더 경로

NII_DIR: 변환된 NII image 폴더 경로

LBL_DIR:  원본 DICOM label 폴더 경로

GT_DIR: 변환된 NII label 폴더 경로

EXCEL_DIR: Center point가 담긴 엑셀 파일 저장 경로

CKPT_DIR : checkpoint 폴더의 경로

(CKPT_DIR 이후에 /ckpt_ep##.pt 등의 이름까지 작성 필요)

OUTPUT_DIR: 예측된 label 저장 폴더 경로

POST_DIR: 예측된 label의 postprocessing 저장 폴더 경로

CSV_DIR: Metric 저장 폴더 경로

(CSV_DIR 이후에 metric_ep##.csv 등의 이름까지 작성 필요)

## Center Point -> Excel로 저장
```bash
python preprocessing.py --dcm_dir GT_DIR --excel_path EXCEL_DIR
```

## NII 변환 진행

DICOM2NII
```bash
python dcm2nii.py --dcm_dir DCM_DIR --nii_dir NII_DIR
```
LBL2NII
```python
python dcm2nii.py --dcm_dir LBL_DIR --nii_dir GT_DIR
```

## Test 진행

Test의 경우에는 총 5개의 checkpoint에 대하여 진행하고 싶습니다.

ckpt_ep299, ckpt_ep49, ckpt_ep99, ckpt_ep149, ckpt_ep199 순서대로 진행해주시면 감사하겠습니다.

```bash
python test.py --ckpt CKPT_DIR/ckpt_ep##.pt --input_dir NII_DIR --out_dir OUTPUT_DIR
```

## Postprocessing 진행
Excel 파일이 있는 경우
```bash
python postprocessing.py --input_dir NII_DIR --out_dir OUTPUT_DIR --post_dir POST_DIR --excel_path EXCEL_DIR --sheet_name SHEET_NAME
```
Excel 파일이 없는 경우
```bash
python postprocessing.py --input_dir NII_DIR --out_dir OUTPUT_DIR --post_dir POST_DIR --select_surface --z_strict
```

## Metric 진행

```bash
python metric.py --gt_dir GT_DIR --post_dir POST_DIR --out_csv CSV_DIR
```
