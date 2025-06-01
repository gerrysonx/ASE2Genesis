import genesis as gs
gs.init(backend=gs.cpu)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.001,
        gravity=(0, 0, -10.0),
    ),
    show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.MJCF(file='/home/gerrysun/work/genesis/mjcf/fgame_skeleton_add_visual.xml'),
)

scene.build()

for i in range(100000):
    scene.step()