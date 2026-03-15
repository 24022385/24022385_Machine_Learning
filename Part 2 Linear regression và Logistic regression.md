# Part 2 : Linear regression và Logistic regression (phần upgrade)

### 1. Cơ chế tối ưu hóa tổng quát

Trong các bài toán hồi quy, chúng ta không "đoán" ngẫu nhiên. Thay vào đó, máy tính thực hiện một quá trình lặp đi lặp lại gồm 3 bước:

1. **Dự đoán:** Đưa dữ liệu qua mô hình để lấy kết quả tạm thời.
2. **Tính lỗi (Loss):** Đo lường khoảng cách giữa kết quả dự đoán và thực tế.
3. **Cập nhật (Update):** Dùng đạo hàm để điều chỉnh trọng số sao cho lỗi ở bước sau nhỏ hơn bước trước.

---

### 2. Triển khai Hồi quy Tuyến tính (Linear Regression)

Đối với bài toán sử dụng phương pháp linear regression thì hàm mất mát ở đây sẽ là MSE (Mean Squared Error) ⇒ chúng ta sẽ cực tiểu hóa hàm này :

$J(w) = \frac{1}{2n} \sum_{i=1}^{n} (\hat{y}^{(i)} - y^{(i)})^2$

Lí do chúng ta sử dụng hàm MSE thay vì hàm MAE (Mean Absolute Error) vì việc đạo hàm sẽ thực hiện được dễ dàng hơn tuy nhiên trong 1 số trường hợp nếu giá trị quá ít và giá trị ngoại lai quá khác biệt thì chúng ta sẽ phải sử dụng hàm MAE :

 $J(w) = \frac{1}{n} \sum_{i=1}^{n} |y^{(i)} - \hat{y}^{(i)}|$

Bây giờ chúng ta sẽ triển khai code để thấy rõ từng bước tối ưu : 

```python
def linear_regression_from_scratch(X, y, lr=0.01, epochs=1000):
    # Khởi tạo trọng số ban đầu = 0 hết
    n_samples, n_features = X.shape
    w = np.zeros((n_features, 1))
    
    for epoch in range(epochs):
        # 1.Dự đoán y_hat = X * w
        y_hat = np.dot(X, w)
        
        # 2.Tính Gradient (Đạo hàm riêng)
        # Với công thức: (1/n) * X^T * (y_hat - y)
        gradient = (1/n_samples) * np.dot(X.T, (y_hat - y))
        
        # 3.Cập nhật trọng số
        w = w - lr * gradient
        
    return w
```

---

### 3. Triển khai Hồi quy Logistic (Logistic Regression)

Khác với Linear, Logistic sử dụng hàm **Sigmoid** để ép giá trị về khoảng (0, 1) và tối ưu hàm **Log-Loss (Cross-Entropy)**.

$s = w^TX$

 $z =\sigma(s) = \frac{1}{1 + e^{-s}}$ (Công thức sigmoid)

⇒ Hàm mất mát (log-loss) với 1 điểm dữ liệu ở đây sẽ là :

$$
J(w;x_i,y_i)=−(y_ilogz_i+(1−y_i)log(1−z_i))
$$

Quan sát code sẽ thấy rõ từng bước tối ưu của mô hình này:

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression_from_scratch(X, y, lr=0.1, epochs=1000):
    n_samples, n_features = X.shape
    w = np.zeros((n_features, 1))
    
    for epoch in range(epochs):
        # 1. Dự đoán xác suất bằng hàm Sigmoid
        z = np.dot(X, w)
        y_prob = sigmoid(z)
        
        # 2. Tính Gradient (Đạo hàm của Log-Loss)
        gradient = (1/n_samples) * np.dot(X.T, (y_prob - y))
        
        # 3. Cập nhật
        w = w - lr * gradient
        
    return w
```

### 4. Phân tích thêm

? Nhận xét về công thức tính đạo hàm :

Dù hàm mất mát khác nhau (MSE vs Log-Loss), nhưng nhờ cấu trúc của hàm Sigmoid, khi đạo hàm Logistic Regression, chúng ta thu được cùng một biểu thức cập nhật trọng số. Điều này thể hiện sự nhất quán trong họ các mô hình tuyến tính (Generalized Linear Models).