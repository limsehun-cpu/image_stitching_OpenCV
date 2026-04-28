# image_stitching_OpenCV
Simple image stitching using OpenCV
## 추가 기능
```python
def blending(copy, before_copy, size):
    alpha = np.linspace(0, 1, 30).reshape(1, -1, 1)
    copy[:,size-30:size] = (alpha * before_copy[:,size-30:size] + (1 - alpha) * copy[:,size-30:size]).astype(np.uint8)
    
    return copy
```
Image blending을 통한 이미지간 부드러운 연결
## 결과
![](image_stitching.png)
## 참고 자료
* image_stitching.py