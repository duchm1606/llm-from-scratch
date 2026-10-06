# Chapter 3: Coding Attention Mechanisms

## Why this chapter matters

Trong Chapter 2, ta đã biến đổi chuỗi text thô thành các vector số học: **Input Embeddings** = Token Embeddings + Positional Embeddings.

Tuy nhiên, các embedding vector này vẫn mang tính **cục bộ và tĩnh**: một từ như `"bank"` trong *"river bank"* và *"bank account"* sẽ có token embedding ban đầu giống nhau (chỉ khác vị trí). Mô hình chưa thể nắm bắt được ngữ cảnh liên kết giữa các từ cách xa nhau.

**Attention mechanism** giải quyết triệt để vấn đề này bằng cách cho phép mỗi token "chú ý" (attend) đến các token khác trong câu, qua đó tổng hợp thông tin để tạo thành các **Context Vectors** giàu ngữ cảnh.

---

## The Attention Evolution

Chương 3 xây dựng cơ chế Attention qua 4 bước tiến hóa:

```
1. Simple Self-Attention (Không có trọng số trainable)
   ↓
2. Self-Attention with Trainable Weights (Query, Key, Value)
   ↓
3. Causal Attention (Thêm Causal Mask để không nhìn thấy tương lai)
   ↓
4. Multi-Head Attention (Chạy nhiều head song song + Weight Split)
```

---

## 3.1 & 3.2 Vấn đề của RNN và Sự ra đời của Attention

- **RNN Encoder-Decoder truyền thống**:
  - Toàn bộ chuỗi đầu vào phải nén vào một vector ẩn duy nhất $h$ (bottleneck).
  - Với câu dài, thông tin ở đầu câu bị suy giảm (vanishing context / catastrophic forgetting).
- **Attention**:
  - Cho phép decoder truy cập trực tiếp vào toàn bộ các trạng thái đầu vào.
  - Phân bổ trọng số (attention weights) linh hoạt: token nào quan trọng thì chú ý nhiều, không quan trọng thì bỏ qua.
- **Self-Attention**:
  - Không cần mô hình RNN; các token trong *cùng một câu* tự tính toán độ liên quan với nhau.

---

## 3.3 Simple Self-Attention (Không có trainable weights)

Cho chuỗi đầu vào gồm $T$ token embedding: $x^{(1)}, x^{(2)}, \dots, x^{(T)} \in \mathbb{R}^{d_{in}}$.

Mục tiêu: Tính toán context vector $z^{(i)}$ cho từng token $x^{(i)}$.

### Các bước tính toán:
1. **Attention Scores ($\omega$)**: Dùng tích vô hướng (dot product) đo độ tương đồng:
   $$\omega_{i, j} = x^{(i)} \cdot x^{(j)}$$
2. **Attention Weights ($\alpha$)**: Chuẩn hóa điểm số bằng hàm Softmax (tổng mỗi hàng = 1):
   $$\alpha_{i, j} = \frac{\exp(\omega_{i, j})}{\sum_{k=1}^T \exp(\omega_{i, k})}$$
3. **Context Vector ($z$)**: Tính tổng có trọng số của các vector đầu vào:
   $$z^{(i)} = \sum_{j=1}^T \alpha_{i, j} x^{(j)}$$

> **Nhược điểm**: Hoàn toàn cố định, không có tham số để mô hình huấn luyện và học hỏi (learnable parameters).

---

## 3.4 Self-Attention với Trainable Weights (Q, K, V)

Để mô hình có thể học các mối quan hệ phong phú, ta phân tách mỗi token thành 3 vai trò khác nhau thông qua 3 ma trận trọng số huấn luyện:

| Thuật ngữ | Ký hiệu | Vai trò | Tương tự trong Search Engine |
|-----------|---------|---------|------------------------------|
| **Query** | $Q$ | Token hiện tại đang tìm kiếm điều gì | Từ khóa bạn gõ vào ô tìm kiếm |
| **Key** | $K$ | Đặc trưng của từng token để so khớp | Tiêu đề/Tags của các video |
| **Value** | $V$ | Thông tin thực sự được trích xuất | Nội dung thực tế của video |

### Công thức:
1. **Chiếu tuyến tính (Linear Projections)**:
   $$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$
   - Với $X \in \mathbb{R}^{T \times d_{in}}$, $W_Q, W_K \in \mathbb{R}^{d_{in} \times d_k}$, $W_V \in \mathbb{R}^{d_{in} \times d_v}$.
2. **Scaled Dot-Product Attention**:
   $$\text{Scores} = Q K^T \in \mathbb{R}^{T \times T}$$
   $$\text{Attention Weights } A = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)$$
   > **Tại sao phải chia cho $\sqrt{d_k}$?**
   > Khi chiều $d_k$ lớn, tích vô hướng giữa 2 vector có phương sai tăng tuyến tính theo $d_k$. Giá trị điểm số quá lớn sẽ đẩy Softmax vào vùng bão hòa (gradient cực nhỏ $\approx 0$), khiến việc học bị đình trệ. Chia cho $\sqrt{d_k}$ đưa phương sai về lại 1.
3. **Context Vectors**:
   $$Z = A V \in \mathbb{R}^{T \times d_v}$$

---

## 3.5 Causal Attention (Masked Self-Attention)

### Tại sao cần Causal Attention?
Trong các mô hình sinh ngôn ngữ tự hồi quy (Autoregressive LLMs như GPT), tại bước thời gian $t$, mô hình **chỉ được phép thấy các token từ $1$ đến $t$** để dự đoán token $t+1$. Nó không được phép nhìn thấy tương lai.

### Causal Masking:
Sau khi tính $QK^T$, trước khi đưa qua Softmax, ta che các vị trí tương lai (nửa tam giác trên $j > i$) bằng $-\infty$:

$$\text{Mask}_{i, j} = \begin{cases} 0 & \text{nếu } j \le i \\ -\infty & \text{nếu } j > i \end{cases}$$

$$\text{Scores}_{\text{masked}} = \frac{Q K^T}{\sqrt{d_k}} + \text{Mask}$$

Vì $\exp(-\infty) = 0$, sau hàm Softmax:
- Toàn bộ trọng số attention đến tương lai triệt tiêu thành $0.0$.
- Tổng trọng số các token quá khứ và hiện tại vẫn chuẩn hóa về đúng $1.0$.

### Dropout trên Attention Weights:
Trong quá trình training, kỹ thuật **Dropout** được áp dụng lên ma trận attention weights $A$: ngẫu nhiên đưa một số trọng số về 0 (và scale các trọng số còn lại) để tránh việc mô hình phụ thuộc quá mức vào một vài token nhất định (chống overfitting).

---

## 3.6 Multi-Head Attention (MHA) — Trọng tâm cốt lõi

Thay vì chỉ có một bộ $(Q, K, V)$ duy nhất, Multi-Head Attention chia không gian biểu diễn thành $h$ "heads" hoạt động độc lập và song song.

```
                    Input X [B, T, C]
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
    W_Q [C, C]          W_K [C, C]          W_V [C, C]
       │                   │                   │
       ▼                   ▼                   ▼
    Q [B, T, C]         K [B, T, C]         V [B, T, C]
       │                   │                   │
 ┌─────┴───────────────────┴───────────────────┴─────┐
 │  Reshape & Transpose: [B, num_heads, T, head_dim]  │
 └─────┬───────────────────┬───────────────────┬─────┘
       ▼                   ▼                   ▼
    Head 1              Head 2              Head h
  (Syntax/Grammar)    (Semantics/Ref)     (Proximity)
       │                   │                   │
       └───────────────────┼───────────────────┘
                           ▼
                  Concat các Heads
                 [B, T, h * head_dim]
                           │
                           ▼
                 Linear Projection W_O
                        [C, C]
                           │
                           ▼
                    Output [B, T, C]
```

### 1. Tại sao cần nhiều Heads?
- Ngôn ngữ có tính đa diện: Một từ có thể vừa có quan hệ ngữ pháp (chủ ngữ - vị ngữ), vừa có quan hệ ngữ nghĩa (đại từ thay thế), vừa có quan hệ vị trí.
- Một attention head đơn lẻ chỉ học được một loại quan hệ trọng số.
- $h$ heads cho phép mô hình đồng thời trích xuất $h$ khía cạnh quan hệ khác nhau tại cùng một vị trí từ.

### 2. Kỹ thuật Weight Split (Triển khai hiệu quả)

Thay vì tạo $h$ ma trận riêng biệt cho mỗi head, thực tế ta tạo **1 ma trận duy nhất gộp chung**:
- Kích thước embedding gốc: $C = d_{\text{in}} = d_{\text{out}}$ (Ví dụ GPT-2: $C = 768$).
- Số heads: $h$ (Ví dụ GPT-2: $h = 12$).
- Kích thước mỗi head: $d_{\text{head}} = C / h$ (Ví dụ: $768 / 12 = 64$).

Toàn bộ $W_Q, W_K, W_V \in \mathbb{R}^{C \times C}$. Chi phí tính toán là hoàn toàn tương đương, nhưng tận dụng tối đa phép nhân ma trận lớn (BLAS / cuBLAS).

### 3. Biến đổi Tensor qua các chiều (Shape Transformations)

Hiểu rõ thứ tự các chiều là chìa khóa để implement MHA chính xác:

1. **Chiếu tuyến tính ban đầu**:
   $$X: [B, T, C] \xrightarrow{\times W_Q} Q: [B, T, C]$$
2. **View / Reshape**:
   $$[B, T, C] \rightarrow [B, T, h, d_{\text{head}}]$$
   *(Tách chiều $C$ thành $h$ cụm, mỗi cụm $d_{\text{head}}$ phần tử)*
3. **Transpose**:
   $$[B, T, h, d_{\text{head}}] \xrightarrow{\text{swap } 1 \leftrightarrow 2} [B, h, T, d_{\text{head}}]$$
   > **Tại sao cần swap $T$ và $h$?**
   > Khi chiều $h$ đứng trước, mỗi head trở thành một ma trận con $[T, d_{\text{head}}]$ độc lập. Phép nhân ma trận tiếp theo tự động diễn ra trên từng head:
4. **Batched Matrix Multiply**:
   $$Q \times K^T: [B, h, T, d_{\text{head}}] \times [B, h, d_{\text{head}}, T] \rightarrow [B, h, T, T]$$
5. **Causal Masking & Softmax**:
   Áp dụng mask và softmax trên chiều cuối cùng ($T$): kết quả là ma trận trọng số $[B, h, T, T]$.
6. **Tổng hợp Values**:
   $$\text{Attn} \times V: [B, h, T, T] \times [B, h, T, d_{\text{head}}] \rightarrow [B, h, T, d_{\text{head}}]$$
7. **Ghép lại (Merge Heads)**:
   $$[B, h, T, d_{\text{head}}] \xrightarrow{\text{swap } 1 \leftrightarrow 2} [B, T, h, d_{\text{head}}] \xrightarrow{\text{view}} [B, T, C]$$
8. **Output Projection $W_O$**:
   $$[B, T, C] \xrightarrow{\times W_O} [B, T, C]$$
   $W_O$ là ma trận có thể học được (trainable) có nhiệm vụ hòa trộn (blend) thông tin từ tất cả các heads lại với nhau.

---

### 4. Góc nhìn bộ nhớ trong C thuần túy

Trong C, không có hàm tensor high-level; toàn bộ tensor đa chiều chỉ là một mảng 1D liên tục trong RAM theo thứ tự **Row-Major**:

- Truy cập phần tử $Q[b, h, t, d]$:
  $$\text{index} = b \cdot (h \cdot T \cdot d_{\text{head}}) + h \cdot (T \cdot d_{\text{head}}) + t \cdot d_{\text{head}} + d$$

- **Trick không cần copy bộ nhớ khi tính $Q K^T$**:
  Thay vì transpose thực tế $K$, ta tính tích vô hướng trực tiếp giữa vector $Q[b, h, i]$ và $K[b, h, j]$:
  $$\text{Score}[b, h, i, j] = \sum_{d=0}^{d_{\text{head}}-1} Q[b, h, i, d] \cdot K[b, h, j, d]$$
  Cả hai đều đọc dữ liệu liên tục theo chiều $d$, cực kỳ thân thiện với CPU Cache và vectorization (SIMD / AVX).

---

## 3.7 Tổng kết các tham số của Attention trong GPT-2

| Cấu hình | GPT-2 Small | GPT-2 Medium | GPT-2 Large | GPT-2 XL |
|----------|-------------|--------------|-------------|----------|
| Embedding Dimension ($C$) | 768 | 1024 | 1280 | 1600 |
| Số Attention Heads ($h$) | 12 | 16 | 20 | 25 |
| Head Dimension ($d_{\text{head}}$) | 64 | 64 | 64 | 64 |
| Context Length ($T$) | 1024 | 1024 | 1024 | 1024 |
| Trọng số MHA 1 Layer | 4 ma trận ($W_Q, W_K, W_V, W_O$) kích thước $C \times C$ |

---

## Checklist Implement cho Repo C

Để sẵn sàng cho Chapter 4 (Ghép nối mô hình GPT hoàn chỉnh), ta cần triển khai:

- [ ] `include/attention.h`: Khai báo struct và API cho Multi-Head Attention.
- [ ] `src/attention.c`:
  - `matmul` & projection cho Q, K, V, O.
  - Scaled dot-product theo từng head.
  - Causal masking & In-place Softmax ổn định số học (subtract max).
  - Kết hợp kết quả các heads và chiếu qua $W_O$.
- [ ] `tests/test_attention.c`: So sánh output với các test cases mẫu của PyTorch.
