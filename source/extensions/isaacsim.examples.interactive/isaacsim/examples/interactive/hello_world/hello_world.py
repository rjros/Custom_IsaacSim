from isaacsim.examples.interactive.base_sample import BaseSample

from omni.physx.scripts import deformableUtils, physicsUtils
import omni.physx.bindings._physx as physx_settings_bindings
from pxr import UsdGeom, Gf
import carb
from pxr import UsdGeom, Gf, PhysxSchema



class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()

    def setup_scene(self):

        # ---------------------------------------------------
        # Get world and stage
        # ---------------------------------------------------
        world = self.get_world()
        world.scene.add_default_ground_plane()

        # BaseSample's World object exposes the stage directly
        stage = world.stage

        # ---------------------------------------------------
        # Check that PhysX deformable beta is enabled
        # ---------------------------------------------------
        deformable_beta_on = carb.settings.get_settings().get_as_bool(
            physx_settings_bindings.SETTING_ENABLE_DEFORMABLE_BETA
        )
        if not deformable_beta_on:
            carb.log_warn("⚠ Enable PhysX Deformable Beta in the Physics settings panel first!")
            return

        # ---------------------------------------------------
        # Create a deformable jello cube
        # ---------------------------------------------------
        jello_path = self._create_deformable_cube(
            stage=stage,
            prim_path="/World/JelloCube",
            position=Gf.Vec3f(0.0, 0.0, 0.25),  # 25 cm above ground
            size=0.1,                           # 10 cm cube (scaled)
        )

        carb.log_info(f"Deformable cube created at {jello_path}")

    # ------------------------------------------------------------------
    # Internal helper: PhysX 5 deformable cube using auto-volume workflow
    # ------------------------------------------------------------------

    def _create_deformable_cube(self, stage, prim_path, position, size):

        # Root prim for deformable
        xform = UsdGeom.Xform.Define(stage, prim_path)
        xform.AddTranslateOp().Set(position)
        xform.AddScaleOp().Set(Gf.Vec3f(size, size, size))
        xform_prim = xform.GetPrim()

        # Proper UsdGeom.Mesh cube (not analytic Cube)
        skin_path = prim_path + "/mesh"
        skin_mesh = UsdGeom.Mesh.Define(stage, skin_path)
        UsdGeom.Mesh.GenerateCubeMesh(skin_mesh)  # VALID mesh for PhysX

        # Auto volume setup
        sim_mesh_path = prim_path + "/simMesh"
        coll_mesh_path = prim_path + "/collMesh"

        deformableUtils.create_auto_volume_deformable_hierarchy(
            stage,
            root_prim_path=prim_path,
            simulation_tetmesh_path=sim_mesh_path,
            collision_tetmesh_path=coll_mesh_path,
            cooking_src_mesh_path=skin_path,
            simulation_hex_mesh_enabled=True,
            cooking_src_simplification_enabled=True,
            set_visibility_with_guide_purpose=True,
        )

        # Apply deformable body API properly
        api = PhysxSchema.PhysxBaseDeformableBodyAPI.Apply(xform_prim)
        api.GetResolutionAttr().Set(3)
        api.GetSelfCollisionAttr().Set(False)

        # Material
        mat_path = "/World/JelloMaterial"
        deformableUtils.add_deformable_material(
            stage,
            mat_path,
            youngs_modulus=5e5,
            poissons_ratio=0.45,
            dynamic_friction=0.4,
            density=500.0,
        )
        mat_api = PhysxSchema.PhysxDeformableMaterialAPI.Apply(stage.GetPrimAtPath(mat_path))
        mat_api.GetElasticityDampingAttr().Set(0.01)

        physicsUtils.add_physics_material_to_prim(stage, xform_prim, mat_path)

        # Color mesh
        skin_mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.4, 1.0)])

        return prim_path

    async def setup_post_load(self):
        return

    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        return

    def world_cleanup(self):
        return


