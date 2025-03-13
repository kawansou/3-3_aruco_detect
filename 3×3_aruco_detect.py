import cv2
import numpy as np

# カメラID
cam_id = 0

# カメラキャプチャの初期化
cap = cv2.VideoCapture(cam_id)

def detect_custom_aruco(frame):
    # グレースケール変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 二値化
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # 輪郭検出
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 内側の四角形のみを抽出（親がある輪郭を取得）
    # -- ArUcoが二重に輪郭検出されてしまうのでその無理矢理な対策
    inner_squares = []
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] != -1:  # 親がある（内側の四角形）
            approx = cv2.approxPolyDP(contour, 10, True)  # 四角形の近似
            if len(approx) == 4:  # 四角形のみをフィルタ
                inner_squares.append(approx)

    detected_ids = []
    
    for cnt in inner_squares:
        # 輪郭の面積が一定以上のものを抽出
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue
        
        # 輪郭の近似
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 正方形の領域を検出
        if len(approx) == 4:
            # 領域を射影変換して3×3のマトリックスを解析
            marker_region = extract_marker_region(frame, approx)
            if marker_region is not None:
                marker_id = recognize_marker(marker_region, marker_region.shape[0])
                if marker_id is not None and marker_id != 0 and marker_id != 511:
                    detected_ids.append(marker_id)
                    # IDを画像内に表示
                    cv2.putText(frame, f"ID: {marker_id}", tuple(approx[0][0]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    # 輪郭
                    cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)

    return frame, detected_ids

def extract_marker_region(frame, approx):
    """ クワイエットゾーンを含めたマーカー領域を射影変換 """
    marker_size = 100  # クワイエットゾーンを含む標準サイズ（ピクセル）

    dst_pts = np.array([[0, 0], [marker_size - 1, 0], 
                        [marker_size - 1, marker_size - 1], [0, marker_size - 1]], dtype="float32")

    src_pts = np.array([p[0] for p in approx], dtype="float32")

    # 射影変換
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    marker = cv2.warpPerspective(frame, M, (marker_size, marker_size))

    return marker

def recognize_marker(marker, marker_size):
    """ 5×5のパターンから3×3のデータ部分を読み取ってIDを推定（描画処理あり） """
    gray = cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # 5×5のセルに分割（外周1セルがクワイエットゾーン）
    cell_size = marker_size // 5
    bit_pattern = []

    # クワイエットゾーンの描画
    for y in range(5):
        for x in range(5):
            top_left = (x * cell_size, y * cell_size)
            bottom_right = ((x + 1) * cell_size, (y + 1) * cell_size)

            if 1 <= x <= 3 and 1 <= y <= 3:
                # データ部分 (3x3)
                cell = binary[y * cell_size:(y + 1) * cell_size, x * cell_size:(x + 1) * cell_size]
                mean_val = np.mean(cell)
                bit = 1 if mean_val > 128 else 0
                bit_pattern.append(bit)

                # データセルの枠を描画（白=赤枠, 黒=青枠）
                color = (0, 0, 255) if bit == 1 else (255, 0, 0)
                cv2.rectangle(marker, top_left, bottom_right, color, 2)
                # print(bit, end=" ")
            else:
                # クワイエットゾーンの枠を描画（緑）
                cv2.rectangle(marker, top_left, bottom_right, (0, 255, 0), 1)
                
    # bitを逆順に変換
    bit_pattern = bit_pattern[::-1]

    # パターンをIDに変換（単純なバイナリ変換）
    marker_id = sum([bit * (2 ** i) for i, bit in enumerate(bit_pattern)])
    
    # marker_id=0または511の場合はIDが不正としてNoneを返す
    if marker_id in [0, 511]:
        return None
    
    # print(f" >> ID: {marker_id}")

    # IDをマーカー画像に描画
    cv2.putText(marker, f"ID: {marker_id}", (5, marker_size - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 描画されたマーカー画像を表示
    cv2.imshow("Detected Marker", marker)

    return marker_id

def main():
    while True:
        # カメラ画像の取得
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, detected_ids = detect_custom_aruco(frame)

        # コンソールにIDを出力
        if detected_ids:
            print("Detected IDs:", detected_ids)

        cv2.imshow("Frame", processed_frame)

        # Qキーでループ終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
