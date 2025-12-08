from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
from isaacsim.core.api.tasks import BaseTask


# Deformabale material libraries

import carb
import numpy as np
import omni
from pxr import Gf, Sdf, UsdGeom, UsdShade, PhysxSchema, UsdPhysics
# import omni.physxdemos as demo
from omni.physx.scripts import deformableUtils, utils, physicsUtils
from omni.physx.scripts.assets_paths import AssetFolders
import omni.physx.bindings._physx as physx_settings_bindings
from omni.physxdemos.utils import franka_helpers
from omni.physxdemos.utils import numpy_utils




# Considers the updated deformable model
deformable_beta_on = carb.settings.get_settings().get_as_bool(physx_settings_bindings.SETTING_ENABLE_DEFORMABLE_BETA)
print("deformable_beta_on %b",deformable_beta_on)


class FrankaPlaying(BaseTask):
    #NOTE: we only cover here a subset of the task functions that are available,
    # checkout the base class for all the available functions to override.
    # ex: calculate_metrics, is_done..etc.
    def __init__(self, name):
        super().__init__(name=name, offset=None)
        self._goal_position = np.array([-0.3, -0.3, 0.0515 / 2.0])
        self._task_achieved = False
        return

    # Check pick and place example  for reference regarding  simulation context
    # get_simulation_context
    # set_world_settings

    def create_jello_cube(self, stage, path, name, position, size, mesh_path, phys_material_path, grfx_material):
        if deformable_beta_on:
            xform_path = path.AppendChild(name)
            xform = UsdGeom.Xform.Define(stage, xform_path)
            skinMesh_path = xform_path.AppendChild("mesh")
            stage.DefinePrim(skinMesh_path).GetReferences().AddReference(mesh_path)
            skinMesh = UsdGeom.Mesh.Define(stage, skinMesh_path)
            skinMesh.AddTranslateOp().Set(position)
            skinMesh.AddOrientOp().Set(Gf.Quatf(1.0))
            skinMesh.AddScaleOp().Set(Gf.Vec3f(size, size, size))

            simMesh_path = xform_path.AppendChild("simMesh")
            collMesh_path = xform_path.AppendChild("collMesh")

            deformableUtils.create_auto_volume_deformable_hierarchy(stage,
                root_prim_path = xform_path,
                simulation_tetmesh_path = simMesh_path,
                collision_tetmesh_path = collMesh_path,
                cooking_src_mesh_path = skinMesh_path,
                simulation_hex_mesh_enabled = True,
                cooking_src_simplification_enabled = True,
                set_visibility_with_guide_purpose = True
            )
            # Set resolution attribute of PhysxAutoDeformableHexahedralMeshAPI
            xform.GetPrim().GetAttribute("physxDeformableBody:resolution").Set(3)
            xform.GetPrim().ApplyAPI("PhysxBaseDeformableBodyAPI")
            xform.GetPrim().GetAttribute("physxDeformableBody:selfCollision").Set(False)
            xform.GetPrim().GetAttribute("physxDeformableBody:solverPositionIterationCount").Set(self.pos_iterations)

            # Set collision mesh properties
            collMeshPrim = stage.GetPrimAtPath(collMesh_path)
            physxCollisionAPI = PhysxSchema.PhysxCollisionAPI.Apply(collMeshPrim)
            physxCollisionAPI.GetContactOffsetAttr().Set(0.02)
            physxCollisionAPI.CreateRestOffsetAttr().Set(0.001)

            # Bind material
            physicsUtils.add_physics_material_to_prim(stage, xform.GetPrim(), phys_material_path)
            omni.kit.commands.execute(
                "BindMaterialCommand", prim_path=xform_path, material_path=grfx_material, strength=None
            )
        else:
            cube_path = path.AppendChild(name)
            stage.DefinePrim(cube_path).GetReferences().AddReference(mesh_path)
            skinMesh = UsdGeom.Mesh.Define(stage, cube_path)
            skinMesh.AddTranslateOp().Set(position)
            skinMesh.AddOrientOp().Set(Gf.Quatf(1.0))
            skinMesh.AddScaleOp().Set(Gf.Vec3f(size, size, size))
            deformableUtils.add_physx_deformable_body(
                stage,
                cube_path,
                simulation_hexahedral_resolution=3,
                collision_simplification=True,
                self_collision=False,
                solver_position_iteration_count=self.pos_iterations,
            )
            physicsUtils.add_physics_material_to_prim(stage, skinMesh.GetPrim(), phys_material_path)
            physxCollisionAPI = PhysxSchema.PhysxCollisionAPI.Apply(skinMesh.GetPrim())
            physxCollisionAPI.GetContactOffsetAttr().Set(0.02)
            physxCollisionAPI.CreateRestOffsetAttr().Set(0.001)
            omni.kit.commands.execute(
                "BindMaterialCommand", prim_path=cube_path, material_path=grfx_material, strength=None
            )

    def create(self, stage, Num_Frankas, Num_Greens):
            self.defaultPrimPath = stage.GetDefaultPrim().GetPath()
            self.stage = stage
            self.num_envs = Num_Frankas
            self.num_greens = Num_Greens

            # Physics scene
            scene = UsdPhysics.Scene.Define(stage, self.defaultPrimPath.AppendChild("physicsScene"))
            scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
            scene.CreateGravityMagnitudeAttr().Set(9.807)
            utils.set_physics_scene_asyncsimrender(scene.GetPrim(), False)
            physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
            physxSceneAPI.CreateFrictionOffsetThresholdAttr().Set(0.001)
            physxSceneAPI.CreateFrictionCorrelationDistanceAttr().Set(0.0005)
            physxSceneAPI.CreateGpuTotalAggregatePairsCapacityAttr().Set(10 * 1024)
            physxSceneAPI.CreateGpuFoundLostPairsCapacityAttr().Set(10 * 1024)
            physxSceneAPI.CreateGpuCollisionStackSizeAttr().Set(64 * 1024 * 1024)
            physxSceneAPI.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(10 * 1024)
            # clamp iterations:
            self.pos_iterations = 20
            self.vel_iterations = 1
            physxSceneAPI.GetMaxPositionIterationCountAttr().Set(self.pos_iterations)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)

            # setup ground collision plane:
            utils.addPlaneCollider(stage, "/World/physicsGroundPlaneCollider", "Z")

            # deformable material path
            deformable_material_path = omni.usd.get_stage_next_free_path(stage, "/DeformableBodyMaterial", True)
            deformable_material2_path = omni.usd.get_stage_next_free_path(stage, "/DeformableBodyMaterial2", True)

            if deformable_beta_on:
                deformableUtils.add_deformable_material(stage, deformable_material_path,
                    youngs_modulus=10000000.0,
                    poissons_ratio=0.499,
                    dynamic_friction=1.0,
                    density=100.0
                )
                mat_prim = stage.GetPrimAtPath(deformable_material_path)
                mat_prim.ApplyAPI("PhysxDeformableMaterialAPI")
                mat_prim.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(0.0001)

                deformableUtils.add_deformable_material(stage, deformable_material2_path,
                    youngs_modulus=4000000.0,
                    poissons_ratio=0.499,
                    dynamic_friction=0.05,
                    density=100.0
                )
                mat2_prim = stage.GetPrimAtPath(deformable_material2_path)
                mat2_prim.ApplyAPI("PhysxDeformableMaterialAPI")
                mat2_prim.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(0.005)
            else:
                deformableUtils.add_deformable_body_material(
                    stage,
                    deformable_material_path,
                    youngs_modulus=10000000.0,
                    poissons_ratio=0.499,
                    damping_scale=0.0,
                    elasticity_damping=0.0001,
                    dynamic_friction=1.0,
                    density=100,
                )
                deformableUtils.add_deformable_body_material(
                    stage,
                    deformable_material2_path,
                    youngs_modulus=4000000.0,
                    poissons_ratio=0.499,
                    damping_scale=0.0,
                    elasticity_damping=0.005,
                    dynamic_friction=0.05,
                    density=100,
                )


    # Here we setup all the assets that we care about in this task.
    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        scene.add_default_ground_plane()

        # create soft cube
        # self.create_jello_cube()

        # self._cube = scene.add(DynamicCuboid(prim_path="/World/random_cube",
        #                                     name="fancy_cube",
        #                                     position=np.array([0.3, 0.3, 0.3]),
        #                                     scale=np.array([0.0515, 0.0515, 0.0515]),
        #                                     color=np.array([0, 0, 1.0])))
        self._franka = scene.add(Franka(prim_path="/World/Fancy_Franka",
                                        name="fancy_franka"))
        return
    
    
    # Information exposed to solve the task is returned from the task through get_observations
    def get_observations(self):
        cube_position, _ = self._cube.get_world_pose()
        current_joint_positions = self._franka.get_joint_positions()
        observations = {
            self._franka.name: {
                "joint_positions": current_joint_positions,
            },
            self._cube.name: {
                "position": cube_position,
                "goal_position": self._goal_position
            }
        }
        return observations

    # Called before each physics step,
    # for instance we can check here if the task was accomplished by
    # changing the color of the cube once its accomplished
    def pre_step(self, control_index, simulation_time):
        cube_position, _ = self._cube.get_world_pose()
        if not self._task_achieved and np.mean(np.abs(self._goal_position - cube_position)) < 0.02:
            # Visual Materials are applied by default to the cube
            # in this case the cube has a visual material of type
            # PreviewSurface, we can set its color once the target is reached.
            self._cube.get_applied_visual_material().set_color(color=np.array([0, 1.0, 0]))
            self._task_achieved = True
        return

    # Called after each reset,
    # for instance we can always set the gripper to be opened at the beginning after each reset
    # also we can set the cube's color to be blue
    def post_reset(self):
        self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)
        self._cube.get_applied_visual_material().set_color(color=np.array([0, 0, 1.0]))
        self._task_achieved = False
        return

class PickPlaceSoft(BaseSample):
    def __init__(self):
        super().__init__("my_task")

        with ui.CollapsableFrame("Controls"):
            ui.Button("Start", clicked_fn=self.start_cb)
            ui.Button("Reset", clicked_fn=self.reset_cb)

    def start_cb(self):
        print("Task started!")


class PickPlaceSoft(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):
        world = self.get_world()
        # We add the task to the world here
        world.add_task(FrankaPlaying(name="my_first_task"))
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        # The world already called the setup_scene from the task (with first reset of the world)
        # so we can retrieve the task objects
        self._franka = self._world.scene.get_object("fancy_franka")
        self._controller = PickPlaceController(
            name="pick_place_controller",
            gripper=self._franka.gripper,
            robot_articulation=self._franka,
        )
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        await self._world.play_async()
        return

    async def setup_post_reset(self):
        self._controller.reset()
        await self._world.play_async()
        return

    def physics_step(self, step_size):
        # Gets all the tasks observations
        current_observations = self._world.get_observations()
        actions = self._controller.forward(
            picking_position=current_observations["fancy_cube"]["position"],
            placing_position=current_observations["fancy_cube"]["goal_position"],
            current_joint_positions=current_observations["fancy_franka"]["joint_positions"],
        )
        self._franka.apply_action(actions)
        if self._controller.is_done():
            self._world.pause()
        return