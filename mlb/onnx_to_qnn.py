import qai_hub as hub
import numpy as np

import dataset

model_path = "/data/mlbooster/exports/qc-hf-llama-7b-v2-chat.aimet"

####################
# Compile to a QNN Model Library

# compile_job = hub.submit_compile_job(
#     model=model_path,
#     device=hub.Device("Samsung Galaxy S24"),
#     options="--target_runtime qnn_lib_aarch64_android --quantize_full_type int4",
# )
# assert isinstance(compile_job, hub.CompileJob)

####################
# Compile to a QNN Model Library

# compile_job = hub.submit_compile_job(
#     model=model_path,
#     device=hub.Device("Samsung Galaxy S24"),
#     options="--target_runtime qnn_context_binary --quantize_full_type w4a16 --truncate_64bit_io",
# )
# # --quantize_io
# assert isinstance(compile_job, hub.CompileJob)

####################
# Dataset

# hub_dataset = hub.upload_dataset(data)

####################
# Infer

# inference_job = hub.submit_inference_job(
#     # model=compile_job.get_target_model(),
#     model="mlnw641wm",
#     device=hub.Device("Samsung Galaxy S23 (Family)"),
#     inputs=dict(image=[sample]),
# )
# assert isinstance(inference_job, hub.InferenceJob)

# # VinAI Mistral 1.1B
# inference_job = hub.submit_inference_job(
#     model=hub.get_model("mlnw641wm"),
#     device=hub.Device("Samsung Galaxy S24"),
#     inputs=dataset.data_l32_h12,
# )
# assert isinstance(inference_job, hub.InferenceJob)

# llama_v2_7b_chat_quantized_Llama2_TokenGenerator_1_Quantized
inference_job = hub.submit_inference_job(
    model=hub.get_model("m1q887dvq"),
    device=hub.Device("Samsung Galaxy S24"),
    inputs=dataset.data_l32_h32,
)
assert isinstance(inference_job, hub.InferenceJob)

####################
# Profile

# import qai_hub as hub
# # Profile the previously compiled model
# profile_job = hub.submit_profile_job(
#     model=compile_job.get_target_model(),
#     device=hub.Device("Samsung Galaxy S23"),
# )
# assert isinstance(profile_job, hub.ProfileJob)