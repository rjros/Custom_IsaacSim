# SPDX-License-Identifier: Apache-2.0


from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.materials.deformable_material import DeformableMaterial
from isaacsim.core.prims import SingleDeformablePrim
import isaacsim.core.utils.deformable_mesh_utils as deformableMeshUtils

from omni.physx.scripts import physicsUtils
from pxr import UsdGeom, UsdPhysics, Gf
import numpy as np
import carb

class SphereMotion():
    def __init__(self):
        self.my_world= World(stage_units_in_meters=1.0)
        self.stage=simulation_app.context.get_stage()
        self.my_world.scene.add_default_ground_plane()
        self.xform = UsdGeom.Xform.Define(self.stage, "/World/Xform")
        self._create_dynamic_cube()
        self._create_deformable_cube()

        # self._create_visual_cube()
        self._create_prismatic_joint()
        # self.cube_2.set_rigid_body_enabled(False)     # freezes dynamics

        # self.rb = PhysxSchema.PhysxRigidBodyAPI.Apply(["/new_cube_1"])
        # self.rb.CreateSolverPositionIterationCountAttr().Set(64)
        # self.rb.CreateSolverVelocityIterationCountAttr().Set(8)


    # def _create_visual_cube(self):
    #     print("Create visual cube")
    #     self.cube_1 = self.my_world.scene.add(
    #     VisualCuboid(
    #         prim_path="/new_cube_1",
    #         name="visual_cube",
    #         position=np.array([0, 0, 0.5]),
    #         size=0.3,
    #         color=np.array([255, 255, 255]),
    #     )
    # )

    def _create_dynamic_cube(self):
        print("Creating dynamic cube")
        # Create dynamic body
        self.cube_1= self.my_world.scene.add(
            DynamicCuboid(
                prim_path="/new_cube_1",
                name="cube_1",
                position=np.array([0, 0, 1.0]),
                scale=np.array([0.6, 0.5, 0.2]),
                size=1.0,
                color=np.array([255, 255, 0]),
            )
        )

    def _create_deformable_cube(self):
        mesh_path = "/DeformableCube"

        # generate triangular surface mesh
        tri_points, tri_indices = deformableMeshUtils.createTriangleMeshCube(6)

        # Convert to numpy for scaling
        tri_points = np.array(tri_points, dtype=float) * 0.30  # 30 cm cube
        tri_points = tri_points.tolist()

        mesh = UsdGeom.Mesh.Define(self.stage, mesh_path)
        mesh.GetPointsAttr().Set(tri_points)
        mesh.GetFaceVertexIndicesAttr().Set(tri_indices)
        mesh.GetFaceVertexCountsAttr().Set([3] * (len(tri_indices) // 3))

        physicsUtils.setup_transform_as_scale_orient_translate(mesh)
        physicsUtils.set_or_add_translate_op(mesh, (0.0, 0.0, 0.35))

        material = DeformableMaterial(
            prim_path="/World/SoftMaterial",
            youngs_modulus=3e4,
            poissons_ratio=0.45,
            damping_scale=0.1,
            elasticity_damping=0.05,
        )

        self.soft_cube = SingleDeformablePrim(
            prim_path=mesh_path,
            name="soft_cube",
            deformable_material=material,
            simulation_hexahedral_resolution=3,
            solver_position_iteration_count=25,
            collision_simplification=True,
            self_collision=True,
        )

        self.my_world.scene.add(self.soft_cube)

    
    def _create_prismatic_joint(self):
        self.prismatic_joint_1 = UsdPhysics.PrismaticJoint.Define(self.stage,"/World/Joint_Z")
        self.prismatic_joint_1.CreateAxisAttr("Z")
        self.prismatic_joint_1.CreateLowerLimitAttr(0.0)
        self.prismatic_joint_1.CreateUpperLimitAttr(1.0)
        self.prismatic_joint_1.CreateBody0Rel().SetTargets([str(self.xform.GetPath())])
        self.prismatic_joint_1.CreateBody1Rel().SetTargets(["/new_cube_1"])
        self.drive = UsdPhysics.DriveAPI.Apply(self.prismatic_joint_1.GetPrim(), "linear")
        self.drive.CreateDampingAttr(10000)
        self.drive.CreateStiffnessAttr(10000)
        # px_joint = PhysxSchema.PhysxJointAPI.Get(stage, str(joint.GetPath()))
        # px_joint.CreateMaxJointVelocityAttr().Set(5.0)


    ## Simulation Loop ##
    def play(self):
        # Attach a drive to the joint
        # drive = UsdPhysics.DriveAPI.Apply(
        #     self.prismatic_joint_1.GetPrim(), Tf.Token("linear")
        # )
        # drive.CreateDriveTypeAttr().Set("position")   # position-controlled drive
        # drive.CreateStiffnessAttr().Set(50000.0)      # how strong it holds the target
        # drive.CreateDampingAttr().Set(2000.0)         # damping to avoid oscillation

        target = 0.0          # relative Z displacement
        going_up = False      # motion direction

        while simulation_app.is_running():

            if self.my_world.is_playing():

                # Move between 0.0 and 1.0 meters
                if going_up:
                    target += 0.001   # upward movement
                    if target >= 1.0:
                        going_up = False
                else:
                    target -= 0.001   # downward movement
                    if target <= 0.0:
                        going_up = True

                # Command displacement
                self.drive.GetTargetPositionAttr().Set(target)

            self.my_world.step(render=True)

        simulation_app.close()

if __name__ == "__main__":
    SphereMotion().play()


