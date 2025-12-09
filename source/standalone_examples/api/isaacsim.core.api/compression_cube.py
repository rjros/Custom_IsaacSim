# SPDX-License-Identifier: Apache-2.0
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from omni.physx.scripts import physicsUtils
from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema
import carb


class RigidCompressionTest:
    def __init__(self):
        # World and stage
        self.world = World(stage_units_in_meters=1.0)
        self.stage = self.world.stage
        self.world.scene.add_default_ground_plane()

        # --------------------------------------------------------
        # Configure PhysX Scene (Isaac Sim 5.1.0 API)
        # --------------------------------------------------------
        physx_scene = UsdPhysics.Scene.Define(self.stage, "/World/PhysicsScene")
        physx_api = PhysxSchema.PhysxSceneAPI.Apply(physx_scene.GetPrim())

        # physics update frequency
        physx_api.CreateTimeStepsPerSecondAttr().Set(240)

        # enables continuous collision detection (important for contacts)
        physx_api.CreateEnableCCDAttr().Set(True)

        # sphere motion parameters
        self.start_z = 0.50      # 50 cm above block
        self.speed = -0.001      # downward movement per timestep
        self.current_z = self.start_z

        self._create_block()
        self._create_sphere()

        self.world.reset()
        carb.log_info("Rigid compression test initialized.")

    # ------------------------------------------------------------------
    # Create a rigid cube to compress against
    # ------------------------------------------------------------------
    def _create_block(self):
        block_path = "/World/Block"
        block = UsdGeom.Cube.Define(self.stage, block_path)
        block.GetSizeAttr().Set(0.20)  # 20 cm block

        physicsUtils.set_or_add_translate_op(block, Gf.Vec3f(0.0, 0.0, 0.10))

        prim = block.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)

        col = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        col.CreateContactOffsetAttr().Set(0.02)
        col.CreateRestOffsetAttr().Set(0.0)

        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateSolverPositionIterationCountAttr().Set(64)
        rb.CreateSolverVelocityIterationCountAttr().Set(8)

        carb.log_info("Rigid block created.")

    # ------------------------------------------------------------------
    # Create rigid sphere that moves downward
    # ------------------------------------------------------------------
    def _create_sphere(self):
        sphere_path = "/World/Sphere"
        sphere = UsdGeom.Sphere.Define(self.stage, sphere_path)
        sphere.GetRadiusAttr().Set(0.10)  # 10 cm

        physicsUtils.set_or_add_translate_op(
            sphere, Gf.Vec3f(0.0, 0.0, self.start_z)
        )
        physicsUtils.set_or_add_orient_op(
            sphere, Gf.Quatf(1.0, 0.0, 0.0, 0.0)
        )

        prim = sphere.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)



        self.sphere = sphere
        carb.log_info("Sphere created at {:.2f} m".format(self.start_z))

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    def play(self):
        while simulation_app.is_running():

            if self.world.is_playing():
                # Move sphere downwards
                self.current_z += self.speed

                physicsUtils.set_or_add_translate_op(
                    self.sphere, Gf.Vec3f(0.0, 0.0, self.current_z)
                )

                if self.world.current_time_step_index == 200:
                    print("Sphere Z:", round(self.current_z, 3))

            self.world.step(render=True)

        simulation_app.close()


if __name__ == "__main__":
    RigidCompressionTest().play()
