# 3-3_aruco_detect
3×3のArUcoの検出

# Install 
```shell
pip install -r requirements.txt
```

# Run 
```shell
python 3×3_aruco_detect.py
```

# デコードの方法
ArUcoのIDは二進数バイナリから算出

```math
ArUcoのID = A \times 2^8 + B \times 2^7 + C \times 2^6 + D \times 2^5 + E \times 2^4 + F \times 2^3 + G \times 2^2 + H \times 2^1 + I \times 2^0
```

<img src=https://github.com/user-attachments/assets/b2f89261-ad55-4fcb-a7b5-79cf9467abfb width="300">


⚠️白がバイナリ1に当たる，黒がバイナリ0

# 注意事項
・回転量に応じてバイナリも変わってしまう．

・ArUcoは白のクワイエットゾーンがあるので二重検出されてしまう．今回は内側の輪郭のみを抽出という無理矢理な手を使用．

・小さい輪郭は弾くようにしている．

・真っ白（000000000）or真っ黒（111111111）は使用不可

・位置姿勢推定の機能は入れていない（GPTに聞けば簡単に実装可能）
