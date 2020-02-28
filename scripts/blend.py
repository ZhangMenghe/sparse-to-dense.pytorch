from PIL import Image
from glob import glob

def main(src_dir, mask_dir):
    mask_path = glob(mask_dir + "*.png")
    items = [x.split('/')[-1] for x in mask_path]
    # items = items[:2]
    for idx in range(len(items)):
        Im = Image.open(src_dir + items[idx])
        print(Im.mode,Im.size,Im.format)
        # Im.show()

        newIm = Image.new ("RGBA", (640, 480), (255, 0, 0))
        Im2 = Image.open(mask_path[idx]).convert(Im.mode)
        Im2 = Im2.resize(Im.size)
        # Im2.show()

        img = Image.blend(Im,Im2,0.4)
        # img.show()
        img.save("_blend/" + items[idx])


if __name__ == '__main__':
    # src = "/home/menghe/Github/mediapipe/frame_with_points/0221/"
    src = "../../mediapipe/frame_with_points/0221/"
    mask = "../../PEAC/plane_seg/"
    main(src, mask)
