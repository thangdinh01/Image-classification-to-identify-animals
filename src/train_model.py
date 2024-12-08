import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd


def train_model():

    data_dir = r"D:\Python\BTLXLYA\data_split"

    batch_size = 16
    num_epochs = 10
    learning_rate = 0.001

    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_data = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 8)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    model.train()

    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 50 == 49:
                epoch_loss = running_loss / (i + 1)
                accuracy = 100 * correct / total
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step: {i + 1}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
                running_loss = 0.0

        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        elapsed_time = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{num_epochs} completed in {elapsed_time:.2f}s - Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

        train_losses.append(epoch_loss)
        train_accuracies.append(accuracy)

    torch.save(model.state_dict(), "animal_model.pth")
    print("Huấn luyện mô hình hoàn tất.")

    plt.figure(figsize=(12, 5))

    # Đồ thị Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # Đồ thị Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy')
    plt.legend()

    plt.tight_layout()

    # Lưu đồ thị Loss và Accuracy
    plt.savefig(r"D:\Python\BTLXLYA\output_results\loss_accuracy_plot.png")
    plt.show()

    # Vẽ Heatmap (Confusion Matrix)
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    # Vẽ heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_data.classes, yticklabels=train_data.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Heatmap)')

    # Lưu đồ thị Heatmap
    plt.savefig(r"D:\Python\BTLXLYA\output_results\confusion_matrix_heatmap.png")
    plt.show()

    # Tính toán báo cáo độ đo (Precision, Recall, F1-Score)
    report = classification_report(all_labels, all_preds, target_names=train_data.classes, output_dict=True)

    # Chuyển báo cáo thành DataFrame để dễ dàng vẽ đồ thị
    report_df = pd.DataFrame(report).transpose()

    metrics = report_df.iloc[:-3, :]

    metrics[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(12, 6), width=0.8)

    plt.title('So sánh Precision, Recall và F1-Score giữa các lớp')
    plt.xlabel('Tên lớp')
    plt.ylabel('Giá trị')
    plt.xticks(rotation=45)
    plt.savefig(r"D:\Python\BTLXLYA\output_results\metrics_comparison.png")
    plt.show()


if __name__ == "__main__":
    print("Bắt đầu huấn luyện mô hình...")
    train_model()
