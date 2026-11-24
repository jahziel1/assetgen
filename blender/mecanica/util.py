import bpy, numpy as np, math
from mathutils import Vector

def setup_render(res=500, engine='BLENDER_WORKBENCH'):
    sc=bpy.context.scene
    sc.render.engine=engine
    sc.render.resolution_x=res; sc.render.resolution_y=res
    sc.render.film_transparent=False
    if engine=='BLENDER_WORKBENCH':
        sh=sc.display.shading
        sh.light='STUDIO'; sh.color_type='TEXTURE'; sh.show_shadows=False
        sc.view_settings.view_transform='Standard'
        sc.world = sc.world or bpy.data.worlds.new("W")
        sc.world.color=(0.8,0.8,0.85)
    sc.render.image_settings.file_format='PNG'

def make_cam(loc, target=(0,0,0), ortho=None, lens=50):
    cam=bpy.data.objects.get("CAM")
    if cam is None:
        cd=bpy.data.cameras.new("CAM"); cam=bpy.data.objects.new("CAM",cd); bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera=cam
    cam.location=loc
    d=Vector(target)-Vector(loc)
    cam.rotation_euler=d.to_track_quat('-Z','Y').to_euler()
    if ortho: cam.data.type='ORTHO'; cam.data.ortho_scale=ortho
    else: cam.data.type='PERSP'; cam.data.lens=lens
    return cam

def render_to_array(path):
    sc=bpy.context.scene
    sc.render.filepath=path
    bpy.ops.render.render(write_still=True)
    img=bpy.data.images.load(path)
    w,h=img.size
    a=np.array(img.pixels[:],dtype=np.float32).reshape(h,w,4)
    bpy.data.images.remove(img)
    return a

def save_sheet(tiles, cols, path):
    h,w,_=tiles[0].shape
    rows=(len(tiles)+cols-1)//cols
    out=np.ones((rows*h,cols*w,4),dtype=np.float32)
    for i,t in enumerate(tiles):
        r=i//cols; c=i%cols
        out[(rows-1-r)*h:(rows-r)*h, c*w:(c+1)*w]=t
    img=bpy.data.images.new("sheet",cols*w,rows*h,alpha=True)
    img.pixels=out.ravel().tolist()
    img.filepath_raw=path; img.file_format='PNG'; img.save()
    bpy.data.images.remove(img)
