import drjit as dr
import matplotlib.pyplot as plt

import mitsuba as mi

mi.set_variant("llvm_ad_rgb")

scene_str = "radon-scene/radiance_scene.xml"
scene = mi.load_file(scene_str)
image = mi.render(scene)
mi.Bitmap(image)
plt.show()
