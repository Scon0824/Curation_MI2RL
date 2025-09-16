<h2 align="center"> Curation_MI2RL </h2>

</div>
## Requirement 설치

Install the other packages in `requirements.txt` following:
```bash
pip install -r requirements.txt
```

## 코드의 경로 설정
CKPT_DIR : ckpt를 다운받은 폴더의 경로
INPUT_DIR: test를 진행할 image의 폴더 경로
GT_DIR: test를 진행할 image의 레이블 폴더 경로
OUTPUT_DIR: test의 예측된 레이블 저장 폴더 경로
POST_DIR: 예측된 레이블의 후처리 이후 저장 폴더 경로
CSV_DIR: 예측된 레이블의 결과 저장 파일 이름
(CSV_DIR의 경우에는 파일 이름까지 작성해야함)


## Test 진행
```bash
python test.py --ckpt CKPT_DIR/ckpt_ep299.pt --input_dir INPUT_DIR --out_dir OUTPUT_DIR
```

## Postprocessing 진행
```bash
python test.py --input_dir INPUT_DIR --out_dir OUTPUT_DIR --post_dir POST_DIR
```

## Metric 진행
```bash
python metric.py --gt_dir GT_DIR --post_dir POST_DIR --out_csv CSV_DIR
```
