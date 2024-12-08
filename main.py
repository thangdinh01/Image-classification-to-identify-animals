import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from src.train_model import train_model
from src.test_model import test_image
from src.camera_inference import predict_from_camera, stop_camera
import threading

def handle_train_model():
    messagebox.showinfo("Thông báo", "Bắt đầu huấn luyện mô hình. Vui lòng chờ...")
    train_model()
    messagebox.showinfo("Thông báo", "Huấn luyện mô hình hoàn tất.")



def handle_test_image():
    file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not file_paths:
        return

    # Xóa nội dung cũ trong canvas
    result_canvas.delete("all")

    # Nhóm ảnh theo kết quả dự đoán
    from collections import defaultdict
    grouped_results = defaultdict(list)
    for file_path in file_paths:
        try:
            result = test_image(file_path)
            grouped_results[result].append(file_path)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi kiểm tra {file_path}:\n{str(e)}")

    # Tổng số nhóm kết quả
    num_groups = len(grouped_results)
    if num_groups == 0:
        return

    # Chiều rộng canvas
    canvas_width = result_canvas.winfo_width()

    # Tính khoảng cách giữa các nhóm để căn giữa
    group_width = 200  # Chiều rộng ước lượng cho mỗi nhóm
    total_width = group_width * num_groups
    x_start = max(10, (canvas_width - total_width) // 2)  # Bắt đầu căn giữa, tối thiểu là 10px

    # Hiển thị kết quả theo nhóm
    x_position = x_start
    for result, images in grouped_results.items():
        y_position = 10  # Bắt đầu từ hàng đầu tiên trong mỗi nhóm

        # Hiển thị tên loài
        result_label = tk.Label(result_canvas, text=f"Kết quả: {result}", font=("Arial", 16), bg="white", anchor="w")
        result_canvas.create_window(x_position, y_position, anchor="nw", window=result_label)
        y_position += 40  # Tăng vị trí y để ảnh nằm dưới nhãn

        # Hiển thị các ảnh của loài đó
        for image_path in images:
            image = Image.open(image_path)
            image.thumbnail((150, 150))  # Giảm kích thước ảnh cho vừa khung
            img_display = ImageTk.PhotoImage(image)

            img_label = tk.Label(result_canvas, image=img_display)
            img_label.image = img_display  # Lưu tham chiếu để không bị xóa
            result_canvas.create_window(x_position, y_position, anchor="nw", window=img_label)
            y_position += 160  # Tăng vị trí y để ảnh tiếp theo nằm dưới ảnh hiện tại

        # Tăng vị trí x để nhóm kế tiếp nằm cạnh
        x_position += group_width

    # Điều chỉnh kích thước canvas để phù hợp nội dung
    result_canvas.config(scrollregion=result_canvas.bbox("all"))



def handle_test_camera():
    # Xóa nội dung cũ trong canvas
    result_canvas.delete("all")

    # Hiển thị thông báo trên canvas
    info_label = tk.Label(result_canvas, text="Mở camera để nhận diện...", font=("Arial", 16), bg="white", anchor="w")
    result_canvas.create_window(10, 10, anchor="nw", window=info_label)

    # Khởi chạy luồng xử lý camera
    try:
        threading.Thread(target=predict_from_camera).start()
    except Exception as e:
        messagebox.showerror("Lỗi", str(e))


def handle_stop_camera():
    stop_camera()

    # Xóa nội dung cũ trong canvas
    result_canvas.delete("all")

    # Hiển thị thông báo trên canvas
    info_label = tk.Label(result_canvas, text="Đã tắt camera.", font=("Arial", 16), bg="white", anchor="w")
    result_canvas.create_window(10, 10, anchor="nw", window=info_label)


def main():
    global result_canvas, result_canvas_frame

    root = tk.Tk()
    root.title("Hệ Thống Nhận Diện Động Vật")
    root.geometry("1270x720")

    # Khung trên cùng
    top_frame = tk.Frame(root, height=100, bg="lightblue")
    top_frame.pack(fill=tk.X)

    # Tạo một frame phụ (middle_frame) trong top_frame để chứa các nút chức năng
    middle_frame = tk.Frame(top_frame, bg="lightblue")
    middle_frame.pack(expand=True)  # Sử dụng `expand` để căn giữa

    # Thêm các nút vào middle_frame
    train_button = tk.Button(middle_frame, text="Huấn Luyện Mô Hình", command=handle_train_model, width=20, height=2)
    train_button.grid(row=0, column=0, padx=10, pady=10)

    test_image_button = tk.Button(middle_frame, text="Kiểm Tra Bằng Ảnh", command=handle_test_image, width=20, height=2)
    test_image_button.grid(row=0, column=1, padx=10, pady=10)

    test_camera_button = tk.Button(middle_frame, text="Kiểm Tra Camera", command=handle_test_camera, width=20, height=2)
    test_camera_button.grid(row=0, column=2, padx=10, pady=10)

    stop_camera_button = tk.Button(middle_frame, text="Tắt Camera", command=handle_stop_camera, width=20, height=2)
    stop_camera_button.grid(row=0, column=3, padx=10, pady=10)

    exit_button = tk.Button(middle_frame, text="Thoát", command=root.quit, width=20, height=2)
    exit_button.grid(row=0, column=4, padx=10, pady=10)

    # Khung chính chứa canvas và scrollbar
    main_frame = tk.Frame(root, bg="white")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Canvas để hiển thị kết quả
    result_canvas = tk.Canvas(main_frame, bg="white")
    result_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Thanh cuộn dọc
    scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=result_canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Frame nội dung hiển thị trong canvas
    result_canvas_frame = tk.Frame(result_canvas, bg="white")
    result_canvas.create_window((0, 0), window=result_canvas_frame, anchor="nw")

    # Kết nối canvas với thanh cuộn
    result_canvas.configure(yscrollcommand=scrollbar.set)

    root.mainloop()


if __name__ == "__main__":
    main()
