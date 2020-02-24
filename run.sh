if [ -d "res/"$1 ]; then
 rm -rf "res/"$1
fi
mkdir "res/"$1
python3 main.py --inference=model/model_best.pth.tar --rgbpath=/home/menghe/Github/mediapipe/frames/0221/ --sparsepath=/home/menghe/Github/PEAC/sparse_point/0221/
