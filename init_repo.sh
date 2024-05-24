
mkdir models
cd models

# SuperPoint.
git clone https://github.com/rpautrat/SuperPoint.git
mv SuperPoint/pretrained_models/sp_v6.tgz . && rm -rf SuperPoint
tar zxvf sp_v6.tgz && rm sp_v6.tgz

# DINOv2 - vit-b14.
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth

# OmniGlue.
wget https://storage.googleapis.com/omniglue/og_export.zip
unzip og_export.zip && rm og_export.zip

cd ..

saved_model=./models/og_export
output_onnx=./models/omniglue.onnx
python -m tf2onnx.convert --saved-model ${saved_model}  --output ${output_onnx} --tag serve


saved_model=./models/sp_v6
output_onnx=./models/sp_v6.onnx
python -m tf2onnx.convert --saved-model ${saved_model}  --output ${output_onnx} --tag serve

