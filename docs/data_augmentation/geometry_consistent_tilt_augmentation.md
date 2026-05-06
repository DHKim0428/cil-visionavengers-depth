# Geometry-Consistent Camera Tilt Augmentation for RGB-Depth Pairs

## 0. 목적과 한 줄 요약

이 문서는 monocular depth estimation 프로젝트에서 제안된 **camera tilt 기반 data augmentation** 아이디어를 정리한다. 핵심 질문은 다음과 같다.

> RGB image와 depth map pair가 있을 때, 이미지를 vertical/horizontal 방향으로 기울인 것처럼 만들고, 그에 맞는 depth map도 기하학적으로 일관되게 새로 계산할 수 있는가?

짧은 답은 다음과 같다.

> 가능하다. 단순히 RGB와 depth를 같은 2D warp로 변형하는 것은 충분하지 않고, camera intrinsic을 이용해 pixel을 3D point로 back-project한 뒤, 가상 camera rotation을 적용하고, 다시 image plane으로 project해야 한다. 이때 새 depth는 회전된 camera frame에서의 z-coordinate으로 계산한다.

이 방법은 새로운 기하학 자체라기보다는 **Depth-Image-Based Rendering (DIBR)** 또는 self-supervised monocular depth의 **view synthesis**에서 쓰는 projection geometry를 supervised RGB-depth augmentation에 적용한 것으로 볼 수 있다.

---

## 1. 문제 설정

하나의 training sample이 다음과 같이 주어진다고 하자.

- RGB image

$$
I \in \mathbb{R}^{H \times W \times 3}
$$

- Depth map

$$
D \in \mathbb{R}^{H \times W}
$$

각 pixel coordinate를 homogeneous coordinate로 쓰면:

$$
p = \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
$$

여기서 $u$는 horizontal pixel coordinate, $v$는 vertical pixel coordinate이다.

우리가 만들고 싶은 것은 가상으로 camera를 pitch/yaw 방향으로 기울인 augmented pair:

$$
(I, D) \rightarrow (I', D')
$$

이다.

여기서 주의해야 할 점은 `I'`만 그럴듯하게 만드는 것이 아니라, `D'`도 새 camera frame에 대해 올바른 depth 값을 가져야 한다는 것이다.

---

## 2. 왜 단순 2D warp는 부족한가?

가장 단순한 방법은 RGB와 depth에 같은 image warp를 적용하는 것이다.

$$
I' = \operatorname{warp}(I), \qquad D' = \operatorname{warp}(D)
$$

하지만 이것은 일반적으로 기하학적으로 정확하지 않다.

이유는 camera가 pitch/yaw 방향으로 회전하면, 같은 3D point라도 새 camera coordinate system에서의 z-coordinate이 달라지기 때문이다. Depth map이 보통 camera optical axis 방향의 z-depth를 나타낸다면, camera frame이 바뀌는 순간 depth value도 바뀌어야 한다.

즉 image 위치만 바뀌는 것이 아니라, 각 point의 depth value도 다음과 같이 바뀐다.

$$
D'(p') \neq D(p)
$$

일반적으로는:

$$
D'(p') = \text{rotated camera frame에서의 3D point의 z-coordinate}
$$

이다.

---

## 3. Camera intrinsic이 필요한 이유

Camera intrinsic matrix를 다음과 같이 둔다.

$$
K =
\begin{bmatrix}
 f_x & 0 & c_x \\
 0 & f_y & c_y \\
 0 & 0 & 1
\end{bmatrix}
$$

여기서:

- $f_x, f_y$: focal length in pixel units
- $c_x, c_y$: principal point

pixel $p = [u, v, 1]^\top$에 해당하는 normalized camera ray는:

$$
x = K^{-1} p
$$

이다. 구체적으로:

$$
K^{-1}p =
\begin{bmatrix}
(u-c_x)/f_x \\
(v-c_y)/f_y \\
1
\end{bmatrix}
$$

Depth가 z-depth라고 하면, 해당 pixel의 3D point는:

$$
X = D(u, v) K^{-1}p
$$

즉:

$$
X = D(u, v)
\begin{bmatrix}
(u-c_x)/f_x \\
(v-c_y)/f_y \\
1
\end{bmatrix}
$$

이다.

여기서 중요한 점은, intrinsic $K$가 없으면 pixel $(u,v)$가 3D 공간의 어떤 ray에 해당하는지 알 수 없다는 것이다. 따라서 camera tilt를 물리적으로 올바르게 시뮬레이션하려면 $K$가 필요하다.

---

## 4. Camera tilt를 rotation으로 표현하기

Vertical tilt는 보통 pitch rotation, horizontal tilt는 yaw rotation으로 볼 수 있다.

### 4.1 Yaw rotation

$y$-axis 기준 rotation을 yaw라고 하면, 한 convention에서는 다음과 같이 쓸 수 있다.

$$
R_y(\theta) =
\begin{bmatrix}
\cos\theta & 0 & \sin\theta \\
0 & 1 & 0 \\
-\sin\theta & 0 & \cos\theta
\end{bmatrix}
$$

### 4.2 Pitch rotation

$x$-axis 기준 rotation을 pitch라고 하면:

$$
R_x(\phi) =
\begin{bmatrix}
1 & 0 & 0 \\
0 & \cos\phi & -\sin\phi \\
0 & \sin\phi & \cos\phi
\end{bmatrix}
$$

### 4.3 Combined rotation

두 rotation을 같이 쓰면 예를 들어:

$$
R = R_x(\phi) R_y(\theta)
$$

처럼 둘 수 있다. 실제 구현에서는 coordinate convention에 따라 곱하는 순서나 부호가 달라질 수 있다. 중요한 것은 RGB warp와 depth recomputation에서 같은 convention을 일관되게 쓰는 것이다.

---

## 5. Geometry-consistent depth recomputation

원래 3D point는:

$$
X = D(p) K^{-1}p
$$

가상 camera rotation을 적용하면:

$$
X' = R X
$$

새 camera frame에서의 depth는 $X'$의 z-coordinate이다.

$$
D'(p') = e_3^\top X'
$$

where:

$$
e_3 = \begin{bmatrix}0 \\ 0 \\ 1\end{bmatrix}
$$

따라서:

$$
D'(p') = e_3^\top R D(p) K^{-1}p
$$

Depth $D(p)$는 scalar이므로 밖으로 뺄 수 있다.

$$
D'(p') = D(p) \cdot e_3^\top R K^{-1}p
$$

여기서:

$$
\alpha_R(p) = e_3^\top R K^{-1}p
$$

라고 정의하면:

$$
D'(p') = D(p) \alpha_R(p)
$$

이 식이 가장 핵심이다.

> Camera tilt는 depth에 단순한 constant scale을 곱하는 것이 아니라, pixel 위치 $p$에 의존하는 spatially varying scale factor $\alpha_R(p)$를 곱한다.

---

## 6. 새 pixel 위치 계산

회전된 3D point $X'$는 다시 image plane으로 project된다.

$$
p' \sim KX'
$$

즉:

$$
u' = f_x \frac{X'_x}{X'_z} + c_x
$$

$$
v' = f_y \frac{X'_y}{X'_z} + c_y
$$

따라서 원래 pixel $p$는 새 위치 $p'$로 이동하고, 그 위치의 depth는 $X'_z$가 된다.

---

## 7. 순수 rotation의 homography 표현

Translation 없이 camera rotation만 적용한다면, RGB image warp는 depth와 무관하게 homography로 표현된다.

원래 pixel $p$와 새 pixel $p'$ 사이에는:

$$
p' \sim H p
$$

where:

$$
H = K R K^{-1}
$$

이다.

따라서 RGB image는 homography $H$로 warp할 수 있다.

하지만 depth는 단순히 같은 homography로 warp하는 것만으로는 충분하지 않다. 위치는 homography로 이동하더라도, depth value는:

$$
D'(p') = D(p) \alpha_R(p)
$$

로 새로 계산해야 한다.

---

## 8. Inverse-depth 표현에서의 변화

Depth Anything 계열 모델은 보통 depth 자체보다 inverse depth 또는 relative inverse depth를 사용한다. Inverse depth를:

$$
q(p) = \frac{1}{D(p)}
$$

라고 하자.

위에서:

$$
D'(p') = D(p) \alpha_R(p)
$$

였으므로 inverse depth는:

$$
q'(p') = \frac{1}{D'(p')}
$$

$$
q'(p') = \frac{1}{D(p) \alpha_R(p)}
$$

따라서:

$$
q'(p') = \frac{q(p)}{\alpha_R(p)}
$$

즉 inverse depth 관점에서도 tilt augmentation은 pixel-dependent factor를 만든다.

---

## 9. Small-angle 근사로 보는 효과 크기

Normalized coordinate를:

$$
x_n = \frac{u-c_x}{f_x}, \qquad y_n = \frac{v-c_y}{f_y}
$$

라고 하자.

### 9.1 Yaw의 경우

위 convention의 yaw rotation을 쓰면:

$$
\alpha_R(p) = \cos\theta - x_n \sin\theta
$$

작은 각도에서는:

$$
\cos\theta \approx 1, \qquad \sin\theta \approx \theta
$$

이므로:

$$
\alpha_R(p) \approx 1 - \theta x_n
$$

따라서:

$$
D'(p') \approx D(p)(1 - \theta x_n)
$$

inverse depth는:

$$
q'(p') = \frac{q(p)}{1 - \theta x_n}
$$

작은 각도에서:

$$
q'(p') \approx q(p)(1 + \theta x_n)
$$

즉:

$$
q'(p') \approx q(p) + \theta x_n q(p)
$$

추가로 생기는 항은:

$$
\theta x_n q(p)
$$

이다. 이 항은 pixel coordinate $x_n$에 의존하므로 global scale이나 shift로 제거될 수 없다.

### 9.2 Pitch의 경우

마찬가지로 pitch에 대해서도 $y_n$에 의존하는 factor가 생긴다.

대략적으로:

$$
D'(p') \approx D(p)(1 + \phi y_n)
$$

또는 convention에 따라 부호가 반대가 될 수 있다. 중요한 점은 pitch는 vertical coordinate $y_n$에 따라 depth scale을 다르게 바꾼다는 것이다.

---

## 10. FOV를 가정한다는 것

실제 dataset에 camera intrinsic이 없을 수 있다. 이 경우 정확한 $K$는 알 수 없다. 하지만 approximate intrinsic을 만들 수는 있다.

이미지 크기가 $W \times H$이고, principal point를 image center로 가정하면:

$$
c_x = \frac{W}{2}, \qquad c_y = \frac{H}{2}
$$

그리고 horizontal field of view를 $\gamma$라고 가정하면 focal length는:

$$
f_x = \frac{W}{2\tan(\gamma/2)}
$$

정사각형 이미지이고 square pixel을 가정하면:

$$
f_y = f_x
$$

예를 들어 $W=H=560$일 때:

| Assumed FOV | Approx. focal length |
|---:|---:|
| 50° | $f \approx 600$ |
| 60° | $f \approx 485$ |
| 70° | $f \approx 400$ |

따라서 FOV 60°를 가정하면:

$$
K \approx
\begin{bmatrix}
485 & 0 & 280 \\
0 & 485 & 280 \\
0 & 0 & 1
\end{bmatrix}
$$

이렇게 approximate $K$를 만들어 augmentation을 수행할 수 있다.

---

## 11. FOV 가정이 왜 중요한가?

같은 pixel displacement라도 focal length가 다르면 normalized coordinate가 달라진다.

$$
x_n = \frac{u-c_x}{f_x}
$$

예를 들어 image center에서 오른쪽으로 100 pixel 떨어진 점이 있다고 하자.

만약:

$$
f_x = 300
$$

이면:

$$
x_n = \frac{100}{300} = 0.333
$$

반면:

$$
f_x = 1000
$$

이면:

$$
x_n = \frac{100}{1000} = 0.1
$$

즉 같은 pixel 위치라도 카메라 ray 방향은 완전히 다르다. 따라서 tilt 후 depth 변화량도 달라진다.

---

## 12. Depth Anything의 affine-invariant loss와의 관계

Depth Anything v1/v2 계열은 relative inverse depth를 다루며, 학습에 scale-and-shift invariant loss 또는 affine-invariant normalization을 사용한다. 이 점 때문에 다음 의문이 생긴다.

> 어차피 affine-invariant loss를 쓰면 depth scale이나 shift 변화는 사라지는데, geometry-consistent tilt augmentation이 의미가 있는가?

답은 다음과 같다.

> global affine 변화만 만든다면 의미가 약해진다. 하지만 camera pitch/yaw tilt는 일반적으로 global affine 변화가 아니라 pixel-dependent 변화이므로 affine-invariant loss가 완전히 제거하지 못한다.

### 12.1 Affine-invariant normalization

inverse depth vector를:

$$
q \in \mathbb{R}^N
$$

라고 하자. Affine normalization은 보통 다음 형태이다.

$$
\mathcal{N}(q)_i = \frac{q_i - t(q)}{s(q)}
$$

여기서 $t(q)$는 shift statistic, $s(q)$는 scale statistic이다.

임의의 global affine transform:

$$
q'_i = a q_i + b
$$

에 대해 $a>0$이면 대체로:

$$
\mathcal{N}(q') = \mathcal{N}(q)
$$

가 된다.

따라서 affine-invariant loss는 다음 변화에는 둔감하다.

$$
q \rightarrow aq + b
$$

### 12.2 Tilt augmentation은 affine transform이 아니다

하지만 tilt augmentation에서는:

$$
q'(p') = \frac{q(p)}{\alpha_R(p)}
$$

이다.

이것이 affine-invariant loss에서 사라지려면 어떤 상수 $a,b$가 있어서 모든 pixel에 대해:

$$
\frac{q(p)}{\alpha_R(p)} = a q(p) + b
$$

가 성립해야 한다.

하지만 $\alpha_R(p)$는 pixel 위치에 따라 달라진다. 일반적인 scene에서는 위 식을 만족하는 global constant $a,b$가 존재하지 않는다.

따라서 geometry-consistent tilt target과 naive warped target은 affine-invariant loss 아래에서도 구분 가능하다.

---

## 13. 효과 크기 예시

FOV가 60°라고 하자. 그러면 image half-width에서 normalized coordinate는 대략:

$$
|x_n| \leq \tan(30^\circ) \approx 0.577
$$

Yaw angle을 5°로 두면:

$$
\theta \approx 0.087 \text{ rad}
$$

depth factor 변화량은 대략:

$$
\theta |x_n| \approx 0.087 \times 0.577 \approx 0.05
$$

즉 좌우 끝에서 약 5% 정도의 depth variation이 생긴다.

이것은 수학적으로는 분명히 non-affine signal이지만, 크기가 아주 크지는 않다. 따라서 실제 성능 향상은 ablation으로 확인해야 한다.

---

## 14. 어떤 경우에는 의미가 약해지는가?

### 14.1 Pure global scale/shift만 바뀌는 경우

만약 어떤 augmentation이 inverse depth를 다음처럼만 바꾼다면:

$$
q' = aq + b
$$

affine-invariant loss에서는 거의 의미가 없다.

### 14.2 Roll rotation만 하는 경우

Optical axis 주변의 roll rotation은 z-coordinate을 바꾸지 않는다.

$$
R = R_z(\psi)
$$

이면:

$$
D'(p') \approx D(p)
$$

이다. 따라서 roll은 사실상 image rotation augmentation에 가깝고, depth value 자체의 새로운 geometric supervision은 거의 없다.

### 14.3 Tilt angle이 너무 작은 경우

작은 angle에서는:

$$
\alpha_R(p) \approx 1
$$

이므로 effect size가 작다. 예를 들어 1° 정도의 tilt는 depth target 차이가 너무 작아서 training에 큰 영향을 주지 않을 수 있다.

### 14.4 Intrinsic 추정이 너무 부정확한 경우

가정한 $K$가 실제 camera와 많이 다르면, geometry-consistent라고 생각한 target이 실제로는 noisy label이 될 수 있다. 특히 focal length가 크게 틀리면 $x_n,y_n$이 잘못 계산되어 $\alpha_R(p)$도 틀어진다.

---

## 15. 관련 아이디어와 prior work

이 아이디어와 유사한 계열은 여러 개가 있다.

### 15.1 Depth-Image-Based Rendering, DIBR

DIBR은 RGB image와 depth map으로 virtual view를 만드는 기술이다. 수학적으로는 거의 같은 식을 사용한다.

$$
X = D(p)K^{-1}p
$$

$$
X' = RX + t
$$

$$
p' \sim K'X'
$$

이 계열에서는 holes, disocclusion, occlusion artifact, depth boundary artifacts 등이 중요한 문제로 다뤄진다. 우리의 augmentation에서도 유사한 practical issue가 생긴다.

### 15.2 Self-supervised monocular depth estimation

SfMLearner나 Monodepth류 방법은 video frame 사이의 photometric reconstruction을 위해 depth, pose, intrinsic을 사용한 differentiable view synthesis를 수행한다.

Target view의 pixel을 3D로 올린 뒤 source camera로 변환하고 source image에서 sampling한다.

$$
p_s \sim K T_{t \rightarrow s} D_t(p_t)K^{-1}p_t
$$

우리 아이디어와의 차이는 목적이다.

| 항목 | Self-supervised view synthesis | Geometry-consistent tilt augmentation |
|---|---|---|
| 입력 | video frames | RGB-depth pair |
| depth | network prediction | ground-truth or pseudo-label depth |
| pose | learned or estimated relative pose | sampled virtual pitch/yaw |
| 목적 | self-supervised photometric loss | supervised data augmentation |

### 15.3 Camera-aware depth models

CAM-Convs 같은 방법은 single-view depth estimation에서 camera intrinsics가 중요하다는 점을 강조한다. Focal length나 principal point에서 camera-aware feature를 만들고 network에 넣는다.

이는 우리의 주장과 잘 맞는다.

> Camera intrinsics and FOV matter for monocular depth geometry.

### 15.4 Any-camera depth models

Depth Any Camera 같은 최근 방법은 다양한 FOV, camera pitch, fisheye, panorama setting에 robust한 depth estimation을 다룬다. 여기에서도 FOV alignment, pitch-aware conversion, camera-specific geometry가 중요하게 다뤄진다.

우리의 augmentation은 이 흐름과도 연결된다.

---

## 16. Practical implementation plan

Pure rotation만 고려한다면 inverse warping 방식이 구현하기 쉽다.

### 16.1 Inputs

- RGB image $I$
- Depth map $D$
- Approximate or estimated intrinsic $K$
- Rotation $R$

### 16.2 Compute homography

$$
H = KRK^{-1}
$$

### 16.3 For each target pixel

Target pixel을:

$$
p' = [u', v', 1]^\top
$$

라고 하자.

Source pixel은:

$$
p \sim H^{-1}p'
$$

이다.

Homogeneous normalization을 통해 source coordinate $(u,v)$를 얻는다.

### 16.4 Sample RGB and depth

Source coordinate가 image bounds 안에 있으면:

$$
I'(p') = I(p)
$$

$$
D_{\text{src}} = D(p)
$$

를 bilinear sampling으로 가져온다.

Depth boundary artifact를 줄이려면 depth에는 nearest-neighbor 또는 edge-aware sampling을 고려할 수 있다. 기본 구현에서는 bilinear sampling으로 시작해도 된다.

### 16.5 Recompute depth

Source pixel $p$에 대해:

$$
\alpha_R(p) = e_3^\top R K^{-1}p
$$

그리고:

$$
D'(p') = D_{\text{src}} \alpha_R(p)
$$

또는 inverse depth를 쓰면:

$$
q'(p') = \frac{q_{\text{src}}}{\alpha_R(p)}
$$

### 16.6 Valid mask

다음 조건을 만족하지 않으면 invalid로 둔다.

- source coordinate가 image bounds 안에 있어야 함
- $\alpha_R(p) > 0$이어야 함
- sampled depth가 valid range 안에 있어야 함

즉:

$$
M(p') = \mathbf{1}[p \in \Omega] \cdot \mathbf{1}[\alpha_R(p)>0] \cdot \mathbf{1}[D(p)>0]
$$

Training loss는 valid mask 안에서만 계산하는 것이 안전하다.

---

## 17. Forward splatting 방식과 z-buffer

다른 구현 방식은 source pixel을 target으로 forward project하는 것이다.

For each source pixel:

$$
X = D(p)K^{-1}p
$$

$$
X' = RX
$$

$$
p' \sim KX'
$$

$$
D'(p') = X'_z
$$

이 방식은 개념적으로 직관적이지만, discrete image에서는 holes와 collisions가 생긴다. 여러 source pixel이 같은 target pixel로 갈 경우 z-buffer를 사용할 수 있다.

$$
D'(p') = \min X'_z
$$

단, pure camera rotation만 있을 때 continuous geometry에서는 homography가 one-to-one이므로 진짜 physical occlusion change는 없다. Holes와 collisions는 주로 discretization과 resampling 때문에 생긴다. Translation까지 포함하면 실제 occlusion/disocclusion 문제가 생기므로 z-buffer가 더 중요해진다.

---

## 18. Hyperparameter 추천

초기 실험에서는 너무 aggressive하게 가지 않는 것이 좋다.

Recommended range:

```text
Yaw   θ ∈ [-3°, 3°] to [-8°, 8°]
Pitch φ ∈ [-3°, 3°] to [-8°, 8°]
FOV γ ∈ {50°, 60°, 70°}
```

첫 번째 기본값은 다음 정도가 좋다.

```text
FOV = 60°
max yaw = 5°
max pitch = 5°
```

이 설정에서는 image edge에서 대략 5% 정도의 depth factor variation이 생긴다. 너무 작지도 않고, 너무 큰 artifact를 만들 가능성도 낮다.

---

## 19. Ablation design

이 아이디어를 보고서에서 설득력 있게 보이려면 반드시 ablation이 필요하다.

추천 실험:

| Experiment | Description |
|---|---|
| A | Baseline fine-tuning |
| B | Baseline + standard augmentation |
| C | Baseline + naive 2D RGB/depth homography warp |
| D | Baseline + geometry-consistent tilt augmentation with fixed FOV |
| E | Baseline + geometry-consistent tilt augmentation with estimated focal length |

가장 중요한 비교는:

$$
D \quad \text{vs.} \quad C
$$

이다.

만약 geometry-consistent augmentation이 naive warp보다 좋다면, 다음 주장이 가능하다.

> 단순한 image-space augmentation이 아니라, camera-frame depth를 재계산하는 3D-consistent augmentation이 monocular depth fine-tuning에 도움이 된다.

---

## 20. Expected outcomes and interpretation

### 20.1 D > C인 경우

가장 좋은 결과다. 이 경우 geometry-consistent depth recomputation의 의미가 실험적으로 확인된다.

가능한 해석:

- naive depth warp는 label noise를 만든다.
- pitch/yaw에 따른 spatially varying depth factor가 학습에 유용하다.
- model이 camera tilt 변화에 더 robust해진다.

### 20.2 D ≈ C인 경우

이 경우 augmentation은 image-level robustness 정도만 제공하고, depth recomputation 자체의 효과는 작을 수 있다.

가능한 이유:

- affine-invariant loss가 일부 효과를 제거함
- tilt angle이 너무 작음
- model이 이미 large-scale pretraining으로 충분히 robust함
- approximate intrinsic이 부정확함

### 20.3 D < C인 경우

이 경우 geometry-consistent target이 오히려 noisy label이 되었을 가능성이 있다.

가능한 이유:

- assumed FOV가 실제 camera와 너무 다름
- depth map이 true z-depth가 아니라 다른 convention임
- interpolation artifact가 큼
- invalid mask 처리가 부정확함
- augmentation strength가 너무 큼

---

## 21. 보고서에서의 적절한 framing

이 아이디어를 과장해서 “완전히 새로운 depth augmentation”이라고 주장하기보다는, 다음처럼 정직하게 framing하는 것이 좋다.

> We adapt DIBR-style geometric reprojection to supervised monocular depth fine-tuning. Given an RGB-depth pair, we simulate small virtual camera pitch/yaw rotations. Instead of naively warping the depth map in image space, we back-project pixels to 3D using approximate intrinsics, rotate the points, and reproject them to obtain geometrically consistent depth labels.

한국어로 쓰면:

> 우리는 DIBR-style geometric reprojection을 supervised monocular depth fine-tuning에 적용한다. RGB-depth pair가 주어졌을 때 작은 가상 camera pitch/yaw rotation을 시뮬레이션하고, depth map을 단순히 image space에서 warp하는 대신 approximate intrinsic을 이용해 pixel을 3D로 역투영한 뒤 회전 및 재투영하여 기하학적으로 일관된 depth label을 생성한다.

핵심 contribution은 다음 정도로 잡는 것이 좋다.

```text
1. Strong pretrained monocular depth model fine-tuning
2. Scale/shift-invariant or scale-invariant loss
3. Geometry-consistent virtual camera tilt augmentation
4. Ablation against naive image-space depth warp
```

---

## 22. Claim strength calibration

### 강하게 주장해도 되는 부분

- Camera tilt augmentation은 수학적으로 3D reprojection으로 정의할 수 있다.
- 단순 2D depth warp는 일반적으로 camera-frame z-depth 변화를 반영하지 못한다.
- Pitch/yaw rotation은 depth에 pixel-dependent factor를 만든다.
- 이 변화는 일반적으로 global affine transform이 아니므로 affine-invariant loss가 완전히 제거하지 않는다.

### 조심해서 주장해야 하는 부분

- 실제 성능 향상은 보장되지 않는다.
- Approximate FOV assumption이 틀리면 noisy label이 될 수 있다.
- Depth Anything 계열 pretrained model에서는 gain이 작을 수 있다.
- Dataset depth convention이 z-depth가 아니면 식이 수정되어야 한다.

---

## 23. Depth convention caveat

위 식은 depth map이 camera optical axis 방향의 z-depth라고 가정한다.

즉:

$$
X = z K^{-1}p
$$

만약 depth가 Euclidean range, 즉 camera center에서 3D point까지의 거리라면 식이 달라진다.

Euclidean range를 $r$이라고 하면 normalized ray direction은:

$$
\hat{x} = \frac{K^{-1}p}{\lVert K^{-1}p \rVert}
$$

3D point는:

$$
X = r \hat{x}
$$

이다. 이후 rotation은 동일하게:

$$
X' = RX
$$

새 z-depth는:

$$
z' = e_3^\top X'
$$

이고, 새 Euclidean range를 depth로 저장해야 한다면:

$$
r' = \lVert X' \rVert
$$

이다. Pure rotation에서는 Euclidean range $r$ 자체는 변하지 않는다. 하지만 z-depth는 camera frame이 바뀌므로 변한다.

따라서 dataset의 depth convention을 확인하는 것이 중요하다.

---

## 24. Minimal pseudo-code

아래는 pure rotation + inverse warp 기준의 pseudo-code다.

```python
# Inputs:
# I: RGB image, shape [H, W, 3]
# D: depth map, shape [H, W]
# K: intrinsic matrix, shape [3, 3]
# R: sampled rotation matrix, shape [3, 3]

H_mat = K @ R @ inv(K)
H_inv = inv(H_mat)

for each target pixel p_t = [u_t, v_t, 1]:
    # Find corresponding source pixel
    p_s_h = H_inv @ p_t
    p_s = p_s_h[:2] / p_s_h[2]

    if p_s outside image:
        mask[p_t] = 0
        continue

    # Sample source RGB and depth
    I_src = bilinear_sample(I, p_s)
    D_src = sample_depth(D, p_s)

    # Recompute depth factor
    p_s_h_norm = [p_s.x, p_s.y, 1]
    ray = inv(K) @ p_s_h_norm
    alpha = e3.T @ R @ ray

    if alpha <= 0 or D_src invalid:
        mask[p_t] = 0
        continue

    I_aug[p_t] = I_src
    D_aug[p_t] = D_src * alpha
    mask[p_t] = 1
```

If using inverse depth:

```python
q_aug[p_t] = q_src / alpha
```

---

## 25. Final recommendation

이 augmentation은 시도할 가치가 있다. 다만 기대치를 다음처럼 잡는 것이 적절하다.

- 이 방법은 metric scale augmentation이라기보다는 **perspective geometry augmentation**이다.
- Affine-invariant loss를 쓰더라도 완전히 사라지지는 않는다.
- 하지만 effect size는 작은 tilt에서 제한적일 수 있다.
- 가장 설득력 있는 실험은 naive 2D warp와 geometry-consistent 3D reprojection을 직접 비교하는 것이다.

가장 추천하는 시작점:

```text
FOV = 60° fixed
max pitch/yaw = 5°
valid mask 사용
geometry-consistent D' 계산
naive warp와 ablation
```

보고서용 핵심 문장:

> We introduce a geometry-consistent virtual camera tilt augmentation for supervised monocular depth fine-tuning. Unlike naive 2D warping, our method recomputes the target depth using 3D back-projection and camera rotation, preserving the camera-frame depth geometry under virtual pitch/yaw perturbations.

한국어 버전:

> 우리는 supervised monocular depth fine-tuning을 위한 기하학적으로 일관된 virtual camera tilt augmentation을 제안한다. 단순 2D warp와 달리, 제안 방법은 3D back-projection과 camera rotation을 통해 target depth를 재계산함으로써 가상의 pitch/yaw 변화 아래에서도 camera-frame depth geometry를 보존한다.

---

## 26. References and useful links

- Depth Anything V2 paper: <https://papers.nips.cc/paper_files/paper/2024/file/26cfdcd8fe6fd75cc53e92963a656c58-Paper-Conference.pdf>
- MiDaS / dataset mixing paper: <https://vladlen.info/publications/towards-robust-monocular-depth-estimation-mixing-datasets-zero-shot-cross-dataset-transfer/>
- SfMLearner paper page: <https://huggingface.co/papers/1704.07813>
- CAM-Convs paper: <https://openaccess.thecvf.com/content_CVPR_2019/papers/Facil_CAM-Convs_Camera-Aware_Multi-Scale_Convolutions_for_Single-View_Depth_CVPR_2019_paper.pdf>
- Depth Any Camera CVPR 2025 page: <https://cvpr.thecvf.com/virtual/2025/poster/32925>
- DIBR artifact handling example: <https://www.mdpi.com/2076-3417/9/9/1834>
- Virtual Camera Augmentation example: <https://www.scitepress.org/Papers/2026/146268/146268.pdf>
