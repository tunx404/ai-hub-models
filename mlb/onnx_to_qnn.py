import qai_hub as hub

model_path = "/data/mlbooster/exports/mistralai_Mistral-7B-Instruct-v0.2-qc-w4a16/mistralai_Mistral-7B-Instruct-v0.2-qc-w4a16-kvcache-1-2.aimet"

# Compile to a QNN Model Library
compile_job = hub.submit_compile_job(
    model=model_path,
    device=hub.Device("Samsung Galaxy S24"),
    options="--target_runtime qnn_lib_aarch64_android --quantize_full_type int4",
)
assert isinstance(compile_job, hub.CompileJob)
