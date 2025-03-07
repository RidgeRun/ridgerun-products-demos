INPUT_VIDEO=../samples/brownconrady_distortion.mp4
file=calibration.json
calibration_file="`cat $file | tr -d "\n" | tr -d " "`"

brown_conrady_section=$(echo "$calibration_file" | grep -oP '#BROWN_CONRADY#.*DISTORTION_PARAMETERS="\{[^}]*\}' |  sed 's/[\\{}]//g')

brown_conrady_matrix=$(echo "$brown_conrady_section" | awk -F 'CAMERA_MATRIX="' '{print $2}' | awk -F '[}D]' '{print $1}' | tr -d '"' |  sed 's/[\\{}]//g' )

brown_conrady_distortion=$(echo "$brown_conrady_section" | awk -F 'DISTORTION_PARAMETERS="' '{print $2}' | awk -F '}' '{print $1}' | tr -d '"')

FX=$(echo $brown_conrady_matrix |  awk -F'[:, ]' '{print $2}')
FY=$(echo $brown_conrady_matrix |  awk -F'[:, ]' '{print $4}')
CX=$(echo $brown_conrady_matrix |  awk -F'[:, ]' '{print $6}')
CY=$(echo $brown_conrady_matrix |  awk -F'[:, ]' '{print $8}')
K1=$(echo $brown_conrady_distortion |  awk -F'[:, ]' '{print $2}')
K2=$(echo $brown_conrady_distortion |  awk -F'[:, ]' '{print $4}')
P1=$(echo $brown_conrady_distortion |  awk -F'[:, ]' '{print $6}')
P2=$(echo $brown_conrady_distortion |  awk -F'[:, ]' '{print $8}')
K3=$(echo $brown_conrady_distortion |  awk -F'[:, ]' '{print $10}')
K4=$(echo $brown_conrady_distortion |  awk -F'[:, ]' '{print $12}')
K5=$(echo $brown_conrady_distortion |  awk -F'[:, ]' '{print $14}')
K6=$(echo $brown_conrady_distortion |  awk -F'[:, ]' '{print $16}')

gst-launch-1.0 -v compositor name=comp \
       	sink_0::alpha=1 sink_0::xpos=0 sink_0::ypos=0 sink_1::alpha=1 sink_1::xpos=640 \
	sink_1::ypos=0 ! queue ! videoconvert ! xvimagesink \
	filesrc location=$INPUT_VIDEO !  qtdemux ! avdec_h264 !  nvvidconv ! "video/x-raw(memory:NVMM)"  ! cudaundistort distortion-model=brown-conrady camera-matrix="\"{\"fx\":$FX,\"fy\":$FY,\"cx\":$CX,\"cy\":$CY}\""  distortion-parameters="\"{\"k1\":$K1,\"k2\":$K2,\"p1\":$P1,\"p2\":$P2,\"k3\":$K3,\"k4\":$K4,\"k5\":$K5,\"k6\":$K6}\""  valid-pixels=0 !  nvvidconv  ! "video/x-raw(memory:NVMM),width=600, height=400" ! nvvidconv  ! textoverlay text="Undistorted video" text-x=0 text-y=350  ! \
	comp.sink_1 filesrc location=$INPUT_VIDEO ! qtdemux ! avdec_h264 !  nvvidconv  ! "video/x-raw(memory:NVMM),width=600, height=400" ! nvvidconv  !   textoverlay text="Distorted video" text-x=0 text-y=350 ! comp.sink_0 


