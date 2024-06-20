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
# https://app.aihub.qualcomm.com/docs/hub/api.html#compile-options

# # ONNX only
# compile_job = hub.submit_compile_job(
#     model=f"{model_path}/qc-hf-llama-7b-v2-chat.onnx",
#     device=hub.Device("Samsung Galaxy S24"),
#     options="--target_runtime qnn_context_binary --quantize_full_type w4a16 --truncate_64bit_io --quantize_io false",
# )
# assert isinstance(compile_job, hub.CompileJob)

# # AIMET quantized ONNX
# compile_job = hub.submit_compile_job(
#     model=model_path,
#     device=hub.Device("Samsung Galaxy S24"),
#     options="--target_runtime qnn_context_binary --quantize_full_type w4a16 --truncate_64bit_io",
# )
# assert isinstance(compile_job, hub.CompileJob)

# From model ID
compile_job = hub.submit_compile_job(
    model=hub.get_model("m0q9k2g0m"),
    device=hub.Device("Samsung Galaxy S24"),
    options="--target_runtime qnn_context_binary --quantize_full_type w4a16 --truncate_64_io true --quantize_io true",
)
assert isinstance(compile_job, hub.CompileJob)
