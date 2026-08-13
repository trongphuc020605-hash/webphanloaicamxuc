# PhoBERT + PCA + SVM export

Các file chính:
- pca_svm_pipeline.joblib: pipeline StandardScaler + PCA + LinearSVC
- label_encoder.joblib: ánh xạ nhãn
- config.json: cấu hình model PhoBERT, max_len, PCA dim
- metrics.json / metrics.csv: kết quả Accuracy, Weighted F1
- classification_report.txt: precision, recall, f1 từng lớp
- test_predictions.csv: dự đoán trên tập test
- confusion_matrix_test.png: ma trận nhầm lẫn
- inference_sample.py: code mẫu để dự đoán câu mới

Lưu ý:
- Pipeline không chứa trọng số PhoBERT nếu SAVE_PHOBERT_WEIGHTS=False.
- Khi chạy inference_sample.py, máy cần internet lần đầu để tải model vinai/phobert-base từ Hugging Face.
- Nếu muốn gửi trọn bộ offline, trong notebook đặt SAVE_PHOBERT_WEIGHTS=True trước khi export.
