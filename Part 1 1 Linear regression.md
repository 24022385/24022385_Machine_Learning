# Part 1.1: Linear regression

## Phần 1: Mô tả thuật toán Linear Regression

Dựa vào dữ liệu thống kê, chúng ta cần xác định mối quan hệ giữa giá nhà và thông số đầu vào

- Đặc trưng đầu vào : vector $X$ = {  $x_1 ,x_2 ,x_3$}
    - $x_1 :$  Diện tích ( $m^2$)
    - $x_2 :$Số phòng ngủ
    - $x_3 :$ Khoảng cách tới trung tâm (km)
- Giá trị đầu ra : $y$ - giá nhà (tỷ đồng)
- Mô hình toán học :  $\hat{y} = f(X) = w_1x_1 + w_2x_2 + w_3x_3 + w_0$
    - Trong đó $w_1 ,w_2 ,w_3$ là các trọng số thể hiện mức độ ảnh hưởng của từng yếu tố đầu vào
    - $w_0$ là hệ số chặn (bias)
    - Tìm bộ số $(w_1,w_2,w_3)$ sao cho giá trị dự đoán $\hat{y}$ gần với giá trị thực $y$ nhất.

---

## Phần 2: Triển khai Code

Vì mô hình của chúng ta gồm 3 biến, ta không thể quan sát được mô hình 4 chiều. Thay vài đó cách tốt nhất để hình dung mô hình này là vẽ mô hình biểu diễn mối quan hệ giữa biến quan trọng nhất ( diện tích ) và giá nhà trong khi các biến khác được giữ cố định.

Đầu tiên ta cần 1 tập huấn luyện :

| Diện tích  | Số phòng | Cách trung tâm | Giá nhà |
| --- | --- | --- | --- |
| 50 | 2 | 5 | 2.1 |
| 65 | 2 | 7 | 2.9 |
| 80 | 3 | 2 | 4.2 |
| 100 | 3 | 10 | 3.8 |
| 120 | 4 | 3 | 6.2 |
| 45 | 1 | 8 | 1.7 |
| 75 | 2 | 4 | 3.4 |
| 95 | 3 | 1 | 5.5 |
| 115 | 4 | 6 | 4.9 |
| 140 | 5 | 2 | 8.1 |

Bây giờ máy sẽ sử dụng những dữ liệu trên để vẽ đồ thị 

```python
model = LinearRegression()
model.fit(X, y)

y_pred_train = model.predict(X)

plt.figure(figsize=(10, 6))

# Lấy cột diện tích để làm trục hoành
dien_tich = X[:, 0] 

# Các điểm thực tế là màu xanh
plt.scatter(dien_tich, y, color='blue', label='Dữ liệu thực tế')

# Các điểm dự báo sẽ là màu đỏ 
sort_idx = dien_tich.argsort()
plt.plot(dien_tich[sort_idx], y_pred_train[sort_idx], color='red', label='Dự báo của máy')

plt.xlabel('Diện tích (m2)')
plt.ylabel('Giá nhà (Tỷ đồng)')
plt.legend()
plt.show()
```

![image.png](image.png)

Ta thấy đồ thị là 1 đường dạng gấp khúc vì là mô hình Hồi quy bội, tiếp tục cho máy học tìm các trọng số $(w_1,w_2,w_3)$ và hệ số $w_0$

```python
print(f"w1, w2, w3: {model.coef_}")
print(f"w0: {model.intercept_}")
```

w1, w2, w3: [ 0.04789807  0.1371651  -0.1987164 ]
w0: 0.5833639705882372

---

## Phần 3: Dự đoán kết quả & dẫn chứng

Sau khi mô hình đã học từ dữ liệu cũ, chúng ta dùng nó để định giá cho một căn nhà mới hoàn toàn.
**Bài toán thực tế:** Dự đoán giá một căn nhà có diện tích **85 $m^2$**, **3 phòng ngủ** và cách trung tâm **4 km**.

```python
nha_moi = np.array([[85, 3, 4]])
gia_du_doan = model.predict(nha_moi)

print(f"==> Giá dự đoán cho căn nhà mới là: {gia_du_doan[0]:.2f} tỷ đồng")
```

==> Giá dự đoán cho căn nhà mới là: 4.27 tỷ đồng

**Dẫn chứng & giải thích:**

1. **Dựa trên trọng số $(w)$:** Nếu $w_1$ dương và lớn, nghĩa là diện tích đóng góp rất nhiều vào việc tăng giá nhà. Ngược lại, nếu $w_3$ âm, nghĩa là khoảng cách càng tăng thì giá nhà càng giảm (đúng như phân tích ban đầu).
2. **Kết luận:** Kết quả dự đoán giúp người mua và người bán có một con số tham chiếu khách quan dựa trên quy luật của thị trường thay vì cảm tính.