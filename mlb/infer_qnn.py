import qai_hub as hub
import numpy as np

import dataset

####################
# Infer

# # VinAI Mistral 1.1B
# # raise ValueError("Unknown tensor dtype")
# inference_job = hub.submit_inference_job(
#     model=hub.get_model("mlnw641wm"),
#     device=hub.Device("Samsung Galaxy S24"),
#     inputs=dataset.data_l32_h12,
# )
# assert isinstance(inference_job, hub.InferenceJob)

# llama_v2_7b_chat_quantized_Llama2_TokenGenerator_1_Quantized
inference_job = hub.submit_inference_job(
    model=hub.get_model("mlnw64g3m"),
    device=hub.Device("Samsung Galaxy S24"),
    inputs=dataset.data_l32_h32_part1,
)
assert isinstance(inference_job, hub.InferenceJob)

####################
# From ID

# # llama_v2_7b_chat_quantized_Llama2_TokenGenerator_1_Quantized, MLB
# # 29.6 ms
# # past_value_0_h0_out: (1, 1, 1, 128), float32
# # past_value_0_h1_out: (1, 1, 1, 128), float32
# # ...
# # past_key_7_h31_out: (1, 1, 128, 1), float32
# # layers_7_add_out_0: (1, 1, 4096), float32
# inference_job = hub.get_job("jqp48jw8g")

# # llama_v2_7b_chat_quantized_Llama2_TokenGenerator_1_Quantized, from export
# # 22.7 ms
# # Same float32
# inference_job = hub.get_job("jvgd0ovzp")

# output_dataset = inference_job.get_output_dataset()
# print(output_dataset)
# downloaded_dataset: dict = output_dataset.download()
# print(len(downloaded_dataset))
# for key, value in downloaded_dataset.items():
#     print(f"{key}: {value[0].shape}, {value[0].dtype}")

####################
# Profile

# import qai_hub as hub
# # Profile the previously compiled model
# profile_job = hub.submit_profile_job(
#     model=compile_job.get_target_model(),
#     device=hub.Device("Samsung Galaxy S23"),
# )
# assert isinstance(profile_job, hub.ProfileJob)