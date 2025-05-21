
import re
line="time_layer: 642 detections: (804, 884, 848, 936), (707, 0, 738, 26), (229, 520, 264, 562), "
ds=re.findall(r"\((\d+), (\d+), (\d+), (\d+)\)", line)
for box in ds:
    d = list(map(int, box))
    d = (d[1], d[0], d[3] - d[1], d[2] - d[0])
    print(d)