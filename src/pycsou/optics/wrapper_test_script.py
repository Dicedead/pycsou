#################################################################################################
# IMPORTS
#################################################################################################

import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

import os
from os.path import join, realpath

import drjit as dr
import matplotlib.pyplot as plt

import pycsou.opt.solver as pyopt
import pycsou.opt.stop as pyst
import pycsou.optics.mitsuba_caustics as pymi

#################################################################################################
# SCENE CREATION FUNCTIONS
#################################################################################################


def create_flat_lens_mesh(resolution):
    # Generate UV coordinates
    U, V = dr.meshgrid(
        dr.linspace(mi.Float, 0, 1, resolution[0]), dr.linspace(mi.Float, 0, 1, resolution[1]), indexing="ij"
    )
    texcoords = mi.Vector2f(U, V)

    # Generate vertex coordinates
    X = 2.0 * (U - 0.5)
    Y = 2.0 * (V - 0.5)
    vertices = mi.Vector3f(X, Y, 0.0)

    # Create two triangles per grid cell
    faces_x, faces_y, faces_z = [], [], []
    for i in range(resolution[0] - 1):
        for j in range(resolution[1] - 1):
            v00 = i * resolution[1] + j
            v01 = v00 + 1
            v10 = (i + 1) * resolution[1] + j
            v11 = v10 + 1
            faces_x.extend([v00, v01])
            faces_y.extend([v10, v10])
            faces_z.extend([v01, v11])

    # Assemble face buffer
    faces = mi.Vector3u(faces_x, faces_y, faces_z)

    # Instantiate the mesh object
    mesh = mi.Mesh("lens-mesh", resolution[0] * resolution[1], len(faces_x), has_vertex_texcoords=True)

    # Set its buffers
    mesh_params = mi.traverse(mesh)
    mesh_params["vertex_positions"] = dr.ravel(vertices)
    mesh_params["vertex_texcoords"] = dr.ravel(texcoords)
    mesh_params["faces"] = dr.ravel(faces)
    mesh_params.update()

    return mesh


def load_ref_image(config, resolution, output_dir):
    b = mi.Bitmap(config["reference"])
    b = b.convert(mi.Bitmap.PixelFormat.RGB, mi.Bitmap.Float32, False)
    if b.size() != resolution:
        b = b.resample(resolution)

    mi.util.write_bitmap(join(output_dir, "out_ref.exr"), b)

    print("[i] Loaded reference image from:", config["reference"])
    return mi.TensorXf(b)


#################################################################################################
# LOSS FUNCTION
#################################################################################################


def scale_independent_loss(image, ref):
    """Brightness-independent L2 loss function."""
    scaled_image = image / dr.mean(dr.detach(image))
    scaled_ref = ref / dr.mean(ref)
    return dr.mean(dr.sqr(scaled_image - scaled_ref))


#################################################################################################
# SCENE SETUP
#################################################################################################

SCENE_DIR = realpath("../../mitsuba/scenes")

CONFIGS = {
    "wave": {
        "emitter": "gray",
        "reference": join(SCENE_DIR, "references/wave-1024.jpg"),
    },
    "sunday": {
        "emitter": "bayer",
        "reference": join(SCENE_DIR, "references/sunday-512.jpg"),
    },
}

# Pick one of the available configs
config_name = "sunday"
# config_name = 'wave'

config = CONFIGS[config_name]
print("[i] Reference image selected:", config["reference"])

config.update(
    {
        "render_resolution": (128, 128),
        "heightmap_resolution": (512, 512),
        "n_upsampling_steps": 4,
        "spp": 32,
        "max_iterations": 100,
        "learning_rate": 3e-5,
    }
)

output_dir = realpath(join("../../mitsuba/caustics", "outputs", config_name))
os.makedirs(output_dir, exist_ok=True)
print("[i] Results will be saved to:", output_dir)
mi.Thread.thread().file_resolver().append(SCENE_DIR)
lens_res = config.get("lens_res", config["heightmap_resolution"])
lens_fname = join(output_dir, "lens_{}_{}.ply".format(*lens_res))

if not os.path.isfile(lens_fname):
    m = create_flat_lens_mesh(lens_res)
    m.write_ply(lens_fname)
    print("[+] Wrote lens mesh ({}x{} tesselation) file to: {}".format(*lens_res, lens_fname))

emitter = None
if config["emitter"] == "gray":
    emitter = {
        "type": "directionalarea",
        "radiance": {"type": "spectrum", "value": 0.8},
    }
elif config["emitter"] == "bayer":
    bayer = dr.zeros(mi.TensorXf, (32, 32, 3))
    bayer[::2, ::2, 2] = 2.2
    bayer[::2, 1::2, 1] = 2.2
    bayer[1::2, 1::2, 0] = 2.2

    emitter = {
        "type": "directionalarea",
        "radiance": {"type": "bitmap", "bitmap": mi.Bitmap(bayer), "raw": True, "filter_type": "nearest"},
    }

integrator = {
    "type": "ptracer",
    "samples_per_pass": 256,
    "max_depth": 4,
    "hide_emitters": False,
}
# Looking at the receiving plane, not looking through the lens
sensor_to_world = mi.ScalarTransform4f.look_at(target=[0, -20, 0], origin=[0, -4.65, 0], up=[0, 0, 1])
resx, resy = config["render_resolution"]
sensor = {
    "type": "perspective",
    "near_clip": 1,
    "far_clip": 1000,
    "fov": 45,
    "to_world": sensor_to_world,
    "sampler": {"type": "independent", "sample_count": 512},  # Not really used
    "film": {
        "type": "hdrfilm",
        "width": resx,
        "height": resy,
        "pixel_format": "rgb",
        "rfilter": {
            # Important: smooth reconstruction filter with a footprint larger than 1 pixel.
            "type": "gaussian"
        },
    },
}

scene = {
    "type": "scene",
    "sensor": sensor,
    "integrator": integrator,
    # Glass BSDF
    "simple-glass": {
        "type": "dielectric",
        "id": "simple-glass-bsdf",
        "ext_ior": "air",
        "int_ior": 1.5,
        "specular_reflectance": {"type": "spectrum", "value": 0},
    },
    "white-bsdf": {
        "type": "diffuse",
        "id": "white-bsdf",
        "reflectance": {"type": "rgb", "value": (1, 1, 1)},
    },
    "black-bsdf": {
        "type": "diffuse",
        "id": "black-bsdf",
        "reflectance": {"type": "spectrum", "value": 0},
    },
    # Receiving plane
    "receiving-plane": {
        "type": "obj",
        "id": "receiving-plane",
        "filename": "meshes/rectangle.obj",
        "to_world": mi.ScalarTransform4f.look_at(target=[0, 1, 0], origin=[0, -7, 0], up=[0, 0, 1]).scale((5, 5, 5)),
        "bsdf": {"type": "ref", "id": "white-bsdf"},
    },
    # Glass slab, excluding the 'exit' face (added separately below)
    "slab": {
        "type": "obj",
        "id": "slab",
        "filename": "meshes/slab.obj",
        "to_world": mi.ScalarTransform4f.rotate(axis=(1, 0, 0), angle=90),
        "bsdf": {"type": "ref", "id": "simple-glass"},
    },
    # Glass rectangle, to be optimized
    "lens": {
        "type": "ply",
        "id": "lens",
        "filename": lens_fname,
        "to_world": mi.ScalarTransform4f.rotate(axis=(1, 0, 0), angle=90),
        "bsdf": {"type": "ref", "id": "simple-glass"},
    },
    # Directional area emitter placed behind the glass slab
    "focused-emitter-shape": {
        "type": "obj",
        "filename": "meshes/rectangle.obj",
        "to_world": mi.ScalarTransform4f.look_at(target=[0, 0, 0], origin=[0, 5, 0], up=[0, 0, 1]),
        "bsdf": {"type": "ref", "id": "black-bsdf"},
        "focused-emitter": emitter,
    },
}
scene = mi.load_dict(scene)
# Make sure the reference image will have a resolution matching the sensor
sensor = scene.sensors()[0]
crop_size = sensor.film().crop_size()
image_ref = load_ref_image(config, crop_size, output_dir=output_dir)

#################################################################################################
# OPTIMIZATION
#################################################################################################

heightmap_shape = (512, 512, 1)
miloss = pymi.MitsubaCausticsOptWrapper(scene, image_ref, heightmap_shape, scale_independent_loss)

prox_ad = pyopt.ProxAdam(miloss)
prox_ad.fit(
    x0=miloss.get_np_heightmap().flatten(),
    a=3e-05,
    stop_crit=pyst.MaxIter(1000),
    stop_crit_sub=prox_ad.default_stop_crit(),
)

final_heightmap = prox_ad.solution()
miloss.apply(final_heightmap)
final_image = miloss.get_curr_image()
final_heightmap = miloss.get_np_heightmap()

#################################################################################################
# PLOTTING
#################################################################################################


def show_image(ax, img, title):
    ax.imshow(mi.util.convert_to_bitmap(img))
    ax.axis("off")
    ax.set_title(title)


def show_heightmap(fig, ax, values, title):
    im = ax.imshow(values.squeeze(), vmax=1e-4)
    fig.colorbar(im, ax=ax)
    ax.axis("off")
    ax.set_title(title)


fig, ax = plt.subplots(1, 3, figsize=(11, 10))
ax = ax.ravel()
show_heightmap(fig, ax[2], final_heightmap, "Final heightmap")
show_image(ax[1], image_ref, "Reference")
show_image(ax[0], final_image, "Final state")
plt.show()

#################################################################################################
# SAVING LENS AND IMAGE
#################################################################################################

fname = join(output_dir, "heightmap_final.exr")
mi.util.write_bitmap(fname, miloss.get_mi_heightmap())
print("[+] Saved final heightmap state to:", fname)

fname = join(output_dir, "lens_displaced.ply")
miloss.apply_displacement()
lens_mesh = [m for m in scene.shapes() if m.id() == "lens"][0]
lens_mesh.write_ply(fname)
print("[+] Saved displaced lens to:", fname)

fname = join(output_dir, "final_image_comparison.png")
fig.savefig(fname, format="png")
print("[+] Saved final figure to:", fname)
