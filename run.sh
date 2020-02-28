if [ -d "res/"$1 ]; then
 rm -rf "res/"$1
fi
mkdir "res/"$1
python3 main.py --inference=pretrained/model_best.pth.tar --rgbpath=/home/eevee/Github/mediapipe/frames/0221/ --sparsepath=/home/eevee/Github/mediapipe/mappoints/0221/
