from omni.isaac.kit import SimulationApp

# Start Isaac Sim / Kit
app = SimulationApp({"headless": False})

# Now imports are safe
import omni.usd
import omni.kit.commands
import carb

from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema
from omni.physx.scripts import physicsUtils


# ============================================================
# Stage + /World setup  (this was the issue before)
# ============================================================
ctx = omni.usd.get_context()
stage = ctx.new_stage()

# Create /World so the stage is valid
world_xform = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world_xform.GetPrim())

# Now these calls are valid (stage is not False anymore)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)


# ============================================================
# Physics scene + gravity
# ============================================================
scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
scene.CreateGravityMagnitudeAttr().Set(9.81)

# Ground plane collider
physicsUtils.addGroundPlane(stage, "/World/GroundPlane", "Z")


# ============================================================
# Rigid cube
# ============================================================
cube_path = "/World/RigidCube"

# Create a mesh cube using omni.kit (guaranteed mesh topology)
omni.kit.commands.execute(
    "CreateMeshPrim",
    prim_type="Cube",
    path=cube_path,
    select_new_prim=False,
)

cube_mesh = UsdGeom.Mesh.Get(stage, cube_path)
cube_prim = cube_mesh.GetPrim()

# Position cube half a meter above the ground
# (Mesh is Xformable, so we can add a translate op directly)
cube_mesh.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.5))

# Optional: color it
cube_mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(0.8, 0.1, 0.1)])  # red

# Make it a dynamic rigid body with collision
PhysxSchema.PhysxRigidBodyAPI.Apply(cube_prim)
PhysxSchema.PhysxCollisionAPI.Apply(cube_prim)

# You can tweak solver iterations if you like:
body_api = PhysxSchema.PhysxRigidBodyAPI(cube_prim)
body_api.GetSolverPositionIterationCountAttr().Set(8)
body_api.GetSolverVelocityIterationCountAttr().Set(2)


# ============================================================
# Run simulation
# ============================================================
timeline = omni.timeline.get_timeline_interface()
timeline.play()

carb.log_info("Rigid cube example running…")

while app.is_running():
    app.update()

app.close()
