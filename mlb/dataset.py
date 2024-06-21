import numpy as np

import os
import shutil

####################

data_l32_h32_part1 = {
    "input_ids": [np.random.randint(0, 32000, (1, 1)).astype(np.int32)],
    "attention_mask": [np.random.randint(0, 65536, (1, 1, 1, 1024)).astype(np.uint16)],
    "position_ids_cos": [np.random.randint(0, 65536, (1, 1, 1, 64)).astype(np.uint16)],
    "position_ids_sin": [np.random.randint(0, 65536, (1, 1, 1, 64)).astype(np.uint16)],
}
for layer in range(8):
    for head in range(32):
        data_l32_h32_part1[f"past_key_{layer}_h{head}"] = [
            np.random.randint(0, 256, (1, 1, 128, 1023)).astype(np.uint8)
        ]
    for head in range(32):
        data_l32_h32_part1[f"past_value_{layer}_h{head}"] = [
            np.random.randint(0, 256, (1, 1, 1023, 128)).astype(np.uint8)
        ]

####################

data_l32_h12 = {
    "input_ids": [np.random.randint(0, 32000, (1, 1)).astype(np.int32)],
    "attention_mask": [np.random.randint(0, 65536, (1, 1, 1, 1024)).astype(np.uint16)],
    "position_ids_cos": [np.random.randint(0, 65536, (1, 1, 1, 64)).astype(np.uint16)],
    "position_ids_sin": [np.random.randint(0, 65536, (1, 1, 1, 64)).astype(np.uint16)],
}
for layer in range(32):
    for head in range(12):
        data_l32_h12[f"past_key_{layer}_h{head}"] = [
            np.random.randint(0, 256, (1, 1, 128, 1023)).astype(np.uint8)
        ]
    for head in range(12):
        data_l32_h12[f"past_value_{layer}_h{head}"] = [
            np.random.randint(0, 256, (1, 1, 1023, 128)).astype(np.uint8)
        ]

####################


def dump_dataset(dataset_dict: dict, output_dir: str, use_float: bool = False):
    if use_float:
        output_dir += "_float"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "inputs"), exist_ok=True)

    file_paths = []
    for name, data_list in dataset_dict.items():
        array = data_list[0]
        if use_float:
            array = array.astype(np.float32)
        print(name)
        file_path = os.path.join(output_dir, "inputs", f"{name}.npy")
        np.save(file_path, array)
        file_paths.append(file_path)

    file_list_path = os.path.join(output_dir, "file_paths.txt")
    with open(file_list_path, "w") as f:
        f.write(" ".join(file_paths))


if __name__ == "__main__":
    # print(f"data_l32_h32_part1: {[key for key in data_l32_h32_part1]}")
    # dump_dataset(data_l32_h32_part1, "data_l32_h32_part1", use_float=True)
    # dump_dataset(data_l32_h32_part1, "data_l32_h12")
    pass
