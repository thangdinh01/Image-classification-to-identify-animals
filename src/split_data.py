import os
import shutil
import random

def split_data(src_dir, dest_dir, train_ratio=0.85, test_ratio=0.09, val_ratio=0.05):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Tạo thư mục train, val và test
    train_dir = os.path.join(dest_dir, 'train')
    val_dir = os.path.join(dest_dir, 'val')
    test_dir = os.path.join(dest_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Lặp qua từng lớp (mỗi thư mục là một lớp)
    for class_name in os.listdir(src_dir):
        class_path = os.path.join(src_dir, class_name)
        if os.path.isdir(class_path):
            train_class_dir = os.path.join(train_dir, class_name)
            val_class_dir = os.path.join(val_dir, class_name)
            test_class_dir = os.path.join(test_dir, class_name)

            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(val_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)

            # Lấy danh sách các file ảnh trong thư mục lớp
            image_files = os.listdir(class_path)
            random.shuffle(image_files)

            # Chia ảnh thành train/val/test
            total_files = len(image_files)
            train_idx = int(total_files * train_ratio)
            test_idx = int(total_files * (train_ratio + test_ratio))

            train_files = image_files[:train_idx]
            val_files = image_files[train_idx:test_idx]
            test_files = image_files[test_idx:]

            # Copy file vào thư mục tương ứng
            for file_name in train_files:
                shutil.copy(os.path.join(class_path, file_name), os.path.join(train_class_dir, file_name))
            for file_name in val_files:
                shutil.copy(os.path.join(class_path, file_name), os.path.join(val_class_dir, file_name))
            for file_name in test_files:
                shutil.copy(os.path.join(class_path, file_name), os.path.join(test_class_dir, file_name))

    print(f"Đã chia dữ liệu thành công với tỷ lệ {train_ratio*100}% train, {test_ratio*100}% test và {val_ratio*100}% val.")

# Thay đổi đường dẫn nếu cần thiết
split_data('D:/Python/data', 'D:/Python/BTLXLYA/data_split')
