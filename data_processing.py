#数据预处理
import numpy as np
from sklearn.preprocessing import StandardScaler
from read_csv import read_csv

def load_dataset(file_list):
    all_inputs = []
    all_outputs = []
    input_head = None
    output_head = None

    for file in file_list:
        i_head, inputs, o_head, outputs = read_csv(file)
        if input_head is None:
            input_head = i_head
            output_head = o_head
        # 校验文件头一致性
        assert i_head == input_head, "输入特征头不匹配"
        assert o_head == output_head, "输出特征头不匹配"

        all_inputs.extend(inputs)
        all_outputs.extend(outputs)

    return (np.array(all_inputs, dtype=np.float64),
            np.array(all_outputs, dtype=np.float64),
            input_head,
            output_head)


def preprocess_data(X_train, y_train, X_valid, y_valid, X_test, y_test, 
                   input_head, output_head, TARGETS):
    # 提取目标列
    y_train = y_train[:, [output_head.index(t) for t in TARGETS]]
    y_valid = y_valid[:, [output_head.index(t) for t in TARGETS]]
    y_test = y_test[:, [output_head.index(t) for t in TARGETS]]

    # 过滤包含零值的样本
    print("\nFiltering samples where any target is zero...")
    non_zero_mask_train = np.all(y_train != 0, axis=1)
    non_zero_mask_valid = np.all(y_valid != 0, axis=1)
    non_zero_mask_test = np.all(y_test != 0, axis=1)

    X_train, y_train = X_train[non_zero_mask_train], y_train[non_zero_mask_train]
    X_valid, y_valid = X_valid[non_zero_mask_valid], y_valid[non_zero_mask_valid]
    X_test, y_test = X_test[non_zero_mask_test], y_test[non_zero_mask_test]

    print(f"Train samples after filtering: {X_train.shape[0]}")
    print(f"Valid samples after filtering: {X_valid.shape[0]}")
    print(f"Test samples after filtering: {X_test.shape[0]}")
        
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def scale_data(X_train, X_valid, X_test, y_train, y_valid, y_test, TARGETS):
    # 数据标准化
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_valid_scaled = scaler_X.transform(X_valid)
    X_test_scaled = scaler_X.transform(X_test)

    # 为每个目标创建单独的标准化器
    scalers_y = []
    y_train_scaled = np.zeros_like(y_train)
    y_valid_scaled = np.zeros_like(y_valid)
    y_test_scaled = np.zeros_like(y_test)

    for i in range(len(TARGETS)):
        scaler = StandardScaler()
        y_train_scaled[:, i] = scaler.fit_transform(y_train[:, i].reshape(-1, 1)).flatten()
        y_valid_scaled[:, i] = scaler.transform(y_valid[:, i].reshape(-1, 1)).flatten()
        y_test_scaled[:, i] = scaler.transform(y_test[:, i].reshape(-1, 1)).flatten()
        scalers_y.append(scaler)
        print(f"Target {TARGETS[i]} - Mean: {scaler.mean_[0]:.4f}, Scale: {scaler.scale_[0]:.4f}")
        
    return X_train_scaled, X_valid_scaled, X_test_scaled, y_train_scaled, y_valid_scaled, y_test_scaled, scalers_y, scaler_X