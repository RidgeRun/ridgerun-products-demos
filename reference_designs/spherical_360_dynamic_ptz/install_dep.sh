MODEL_DIR=$HOME/360-inference/peoplenet
if ! [ -f $MODEL_DIR/config_infer_primary_peoplenet.txt ]; then
sudo apt install -y unzip
mkdir -p $MODEL_DIR
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/deployable_quantized_v2.6.1/zip -O peoplenet_deployable_quantized_v2.6.1.zip && unzip peoplenet_deployable_quantized_v2.6.1.zip -d $MODEL_DIR/ && rm peoplenet_deployable_quantized_v2.6.1.zip
cd $MODEL_DIR/
cat > config_infer_primary_peoplenet.txt << EOL
################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
tlt-model-key=tlt_encode
tlt-encoded-model=$MODEL_DIR/resnet34_peoplenet_int8.etlt
labelfile-path=$MODEL_DIR/labels.txt
model-engine-file=$MODEL_DIR/resnet34_peoplenet_int8.etlt_b1_gpu0_int8.engine
int8-calib-file=$MODEL_DIR/resnet34_peoplenet_int8.txt
input-dims=3;544;960;0
uff-input-blob-name=input_1
batch-size=1
process-mode=1
model-color-format=0
## 0=FP32, 1=INT8, 2=FP16 mode
network-mode=1
num-detected-classes=3
cluster-mode=2
interval=0
gie-unique-id=1
output-blob-names=output_bbox/BiasAdd;output_cov/Sigmoid

#Use the config params below for dbscan clustering mode
#[class-attrs-all]
#detected-min-w=4
#detected-min-h=4
#minBoxes=3
#eps=0.7

#Use the config params below for NMS clustering mode
[class-attrs-all]
topk=20
nms-iou-threshold=0.5
pre-cluster-threshold=0.4

## Per class configurations
#[class-attrs-0]
#topk=20
#nms-iou-threshold=0.5
#pre-cluster-threshold=0.4

[class-attrs-1]
#disable bag detection
pre-cluster-threshold=1.0
#eps=0.7
#dbscan-min-score=0.5
EOL
fi 


