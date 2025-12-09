# SPDX-License-Identifier: Apache-2.0


from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid,VisualCuboid,FixedCuboid
import numpy as np


from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, PhysicsSchemaTools, Gf, Sdf, Tf
import omni.usd

class SphereMotion():
    def __init__(self):
        self.my_world= World(stage_units_in_meters=1.0)
        self.stage=simulation_app.context.get_stage()
        self.my_world.scene.add_default_ground_plane()
        self.xform = UsdGeom.Xform.Define(self.stage, "/World/Xform")
        self._create_dynamic_cube()
        self._create_dynamic_cube_2()

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

    def _create_dynamic_cube_2(self):
        print("Creating dynamic cube")
        # Create dynamic body
        self.cube_2= self.my_world.scene.add(
            FixedCuboid(
                prim_path="/new_cube_2",
                name="cube_2",
                position=np.array([0, 0, 0.5]),
                scale=np.array([0.5, 0.5, 0.5]),
                size=1,
                color=np.array([255, 0, 0]),
            )
        )
    
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


