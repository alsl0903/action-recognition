# action-recognition


### 깃허브에서 코드 다운로드 후 폴더 이동 
https://github.com/felixchenfy/Realtime-Action-Recognition

### cmu 폴더로 이동하여 openpose를 사용하기 위한 모듈 다운로드
```
cd $MyRoot/src/githubs/tf-pose-estimation/models/graph/cmu 
``` 

```
bash download.sh
```

# 1. 가상환경 설정 후 필요한 라이브러리 다운로드(버전 맞춰서 다운로드)
```
conda create -n tf tensorflow-gpu
```
```
conda activate tf
```
```
cd $MyRoot/src/githubs/tf-pose-estimation
```
```
pip3 install -r requirements.txt
```
```
pip3 install jupyter tqdm
```
```
pip3 install tensorflow-gpu==1.13.1
```
```
sudo apt install swig
```
```
pip3 install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```
```
cd $MyRoot/src/githubs/tf-pose-estimation/tf_pose/pafprocess
```
```
swig -python -c++ pafprocess.i && python3 setup.py build_ext —inplace
```
```
cd $MyRoot
```
```
pip3 install –r requirements.txt
```
# 2. 예제를 실행하여 설치 확인

cd $MyRoot/src/githubs/tf-pose-estimation

python run.py --model=mobilenet_thin —resize=432x368 --image=./images/p1.jpg

## ※ 파라미터 수정
config/config.yaml 파일에서 기호에 맞는 이미지 분류를 위해 파라미터를 수정할 수 있음 (ex. kick -> wave )

또한 image_folder를 입력하여 본인 pc의 경로에 따라 훈련시킬 데이터셋 경로 지정가능.   

# 3. 훈련 스크립트

#### src/s1_get_skeletons_from_training_imgs.py 

제작한 이미지 데이터셋에서 각각의 이미지마다 skeleton(골격)데이터 감지 및 출력

#### src/s2_put_skeleton_txts_to_a_single_txt.py

출력된 골격 데이터(텍스트)들을 하나의 텍스트로 취합  

#### src/s3_preprocess_features.py

골격 데이터를 (x,y) 좌표에 따라 노이즈를 추가하여 데이터를 증가시키는 전처리 작업 진행 및 신체 속도, 정규화된 관절 위치, 관절 속도의 특징을 추출

#### src/s4_train.py 

전처리된 데이터셋을 기반으로 100*100*100 3개 레이어의 DNN 분류기로 훈련 및 trained_classifier.pickle에 훈련시킨 모델 저장


# 4. 테스트

### 비디오 파일에서 테스트
python src/s5_test.py \
    --model_path model/trained_classifier.pickle \
    --data_type video \
    --data_path data_test/exercise.avi \
    --output_folder output

### 이미지 폴더에서 테스트
python src/s5_test.py \
    --model_path model/trained_classifier.pickle \
    --data_type folder \
    --data_path data_test/apple/ \
    --output_folder output

### 웹 카메라에서 테스트
python src/s5_test.py \
    --model_path model/trained_classifier.pickle \
    --data_type webcam \
    --data_path 0 \
    --output_folder output

