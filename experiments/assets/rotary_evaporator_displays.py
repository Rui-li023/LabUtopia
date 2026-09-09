"""Portable textured LCDs with live seven-segment geometry (no texture writes per frame)."""

from pathlib import Path

from pxr import Gf, Sdf, UsdGeom, UsdShade

LAYOUT = {
    "rpm": ("lift", (-0.049, -0.044, 0.687), (0.040, 0.017), 3),
    "bath": ("base", (0.162, -0.1745, 0.1068), (0.045, 0.021), 3),
}
DIGITS = {
    "0": "abcdef",
    "1": "bc",
    "2": "abdeg",
    "3": "abcdg",
    "4": "bcfg",
    "5": "acdfg",
    "6": "acdefg",
    "7": "abc",
    "8": "abcdefg",
    "9": "abcdfg",
    "-": "g",
    " ": "",
}


def build_displays(stage, root: str, origins: dict) -> None:
    for name, (body, center, size, count) in LAYOUT.items():
        path = f"{root}/{body}/Display_{name}"
        xform = UsdGeom.Xform.Define(stage, path)
        xform.AddTranslateOp().Set(Gf.Vec3d(*(center[i] - origins[body][i] for i in range(3))))
        w, h = size
        mesh = UsdGeom.Mesh.Define(stage, path + "/Background")
        mesh.CreatePointsAttr([(-w / 2, 0, -h / 2), (w / 2, 0, -h / 2), (w / 2, 0, h / 2), (-w / 2, 0, h / 2)])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying").Set(
            [(0, 0), (1, 0), (1, 1), (0, 1)]
        )
        mat = UsdShade.Material.Define(stage, f"{root}/Materials/LCD_{name}")
        preview = UsdShade.Shader.Define(stage, str(mat.GetPath()) + "/Preview")
        preview.CreateIdAttr("UsdPreviewSurface")
        preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.07, 0.014, 0.030))
        mat.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
        mdl = UsdShade.Shader.Define(stage, str(mat.GetPath()) + "/MDL")
        mdl.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        mdl.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        mdl.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1))
        mdl.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(f"./textures/{name}_background.png")
        )
        mdl.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.28)
        mat.CreateSurfaceOutput("mdl").ConnectToSource(mdl.ConnectableAPI(), "out")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
        lightmat = UsdShade.Material.Define(stage, f"{root}/Materials/LCD_{name}_digits")
        lightshader = UsdShade.Shader.Define(stage, str(lightmat.GetPath()) + "/MDL")
        lightshader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        lightshader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        lightshader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.95, 0.28, 0.42))
        lightshader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
        lightshader.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.95, 0.28, 0.42))
        lightshader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(2.0)
        lightmat.CreateSurfaceOutput("mdl").ConnectToSource(lightshader.ConnectableAPI(), "out")
        dx, dy, t = w / 4.3, h * 0.36, h * 0.07
        segments = {
            "a": (0, dy, dx * 0.7, t),
            "g": (0, 0, dx * 0.7, t),
            "d": (0, -dy, dx * 0.7, t),
            "f": (-dx * 0.4, dy / 2, t, dy * 0.8),
            "b": (dx * 0.4, dy / 2, t, dy * 0.8),
            "e": (-dx * 0.4, -dy / 2, t, dy * 0.8),
            "c": (dx * 0.4, -dy / 2, t, dy * 0.8),
        }
        for digit in range(count):
            for label, (x, z, sx, sz) in segments.items():
                cube = UsdGeom.Cube.Define(stage, f"{path}/Digits/d{digit}_{label}")
                cube.CreateSizeAttr(1)
                cube.AddTranslateOp().Set(Gf.Vec3d((digit - 1) * dx * 1.20 + x, -0.0002, z))
                cube.AddScaleOp().Set(Gf.Vec3f(sx, 0.00015, sz))
                UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(lightmat)
        dot = UsdGeom.Cube.Define(stage, path + "/Digits/dot")
        dot.CreateSizeAttr(1)
        dot.AddTranslateOp().Set(Gf.Vec3d(dx * 0.60, -0.0002, -dy))
        dot.AddScaleOp().Set(Gf.Vec3f(t, 0.00015, t))
        UsdShade.MaterialBindingAPI.Apply(dot.GetPrim()).Bind(lightmat)
        set_value(stage, root, name, 0 if name == "rpm" else 40, True)


def set_value(stage, root: str, name: str, value: float, enabled: bool) -> None:
    body = LAYOUT[name][0]
    path = f"{root}/{body}/Display_{name}/Digits"
    chars = f"{round(max(0, min(value, 99.9)) * 10):03d}" if name == "bath" else f"{round(max(0, min(value, 999))):03d}"
    for digit, char in enumerate(chars):
        for segment in "abcdefg":
            attr = UsdGeom.Imageable(stage.GetPrimAtPath(f"{path}/d{digit}_{segment}")).GetVisibilityAttr()
            visibility = "inherited" if enabled and segment in DIGITS[char] else "invisible"
            if attr.Get() != visibility:
                attr.Set(visibility)
    dot = UsdGeom.Imageable(stage.GetPrimAtPath(path + "/dot")).GetVisibilityAttr()
    visibility = "inherited" if name == "bath" and enabled else "invisible"
    if dot.Get() != visibility:
        dot.Set(visibility)


def configure_display(stage, root: str, name: str, settings: dict, config_dir: Path) -> None:
    texture = (config_dir / settings["texture"]).resolve()
    if not texture.is_file():
        raise ValueError(f"Display texture does not exist: {texture}")
    mdl = UsdShade.Shader(stage.GetPrimAtPath(f"{root}/Materials/LCD_{name}/MDL"))
    mdl.GetInput("diffuse_texture").Set(Sdf.AssetPath(str(texture)))
    digit = UsdShade.Shader(stage.GetPrimAtPath(f"{root}/Materials/LCD_{name}_digits/MDL"))
    digit.GetInput("emissive_color").Set(Gf.Vec3f(*settings.get("color", [0.95, 0.28, 0.42])))
    digit.GetInput("emissive_intensity").Set(float(settings.get("brightness", 2)))
