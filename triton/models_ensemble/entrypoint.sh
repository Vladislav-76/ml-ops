LAUNCH_TRITON="tritonserver --model-repository=/models ${LAUNCH_PARAMETERS}"

if [ "${LAUNCH_PARAMETERS#*=}" = "gpu-trt" ]; then
  if [ ! -f /models/inference/2/model.engine ]; then
    /usr/src/tensorrt/bin/trtexec --onnx=/models/inference/1/model.onnx --saveEngine=/models/inference/2/model.engine --minShapes=pixel_values:1x3x224x224 --optShapes=pixel_values:4x3x224x224 --maxShapes=pixel_values:16x3x224x224 --fp16
  fi
fi

eval $LAUNCH_TRITON
