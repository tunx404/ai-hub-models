import numpy as np

####################

data_l32_h32_part1 = {
    'input_ids': [np.random.randint(0, 32000, (1, 1)).astype(np.int32)],
    'attention_mask': [np.random.randint(0, 65536, (1, 1, 1, 1024)).astype(np.uint16)],
    'position_ids_cos': [np.random.randint(0, 65536, (1, 1, 1, 64)).astype(np.uint16)],
    'position_ids_sin': [np.random.randint(0, 65536, (1, 1, 1, 64)).astype(np.uint16)]
}
for layer in range(8):
    for head in range(32):
        data_l32_h32_part1[f"past_key_{layer}_h{head}"] = [np.random.randint(0, 256, (1, 1, 128, 1023)).astype(np.uint8)]
    for head in range(32):
        data_l32_h32_part1[f"past_value_{layer}_h{head}"] = [np.random.randint(0, 256, (1, 1, 1023, 128)).astype(np.uint8)]

####################

data_l32_h12 = {
    'input_ids': [np.random.randint(0, 32000, (1, 1)).astype(np.int32)],
    'attention_mask': [np.random.randint(0, 65536, (1, 1, 1, 1024)).astype(np.uint16)],
    'position_ids_cos': [np.random.randint(0, 65536, (1, 1, 1, 64)).astype(np.uint16)],
    'position_ids_sin': [np.random.randint(0, 65536, (1, 1, 1, 64)).astype(np.uint16)]
}
for layer in range(32):
    for head in range(12):
        data_l32_h12[f"past_key_{layer}_h{head}"] = [np.random.randint(0, 256, (1, 1, 128, 1023)).astype(np.uint8)]
    for head in range(12):
        data_l32_h12[f"past_value_{layer}_h{head}"] = [np.random.randint(0, 256, (1, 1, 1023, 128)).astype(np.uint8)]

####################

if __name__ == "__main__":
    print(f"data_l32_h32_part1: {[key for key in data_l32_h32_part1]}")