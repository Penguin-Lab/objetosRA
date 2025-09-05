from arClasses import *

if __name__ == "__main__":
    cam1 = Camera('calib_rt1.npz')
    cam1.carregaImagem('camera_1.jpg')
    # name_obj,size_obj,obj_pos,obj_angle,sprite_count_max,mov_count_max
    objetos = [Objeto('armario/G2IHDSH4BYLTD0ZH7H0UMRL46.obj',0.6,np.array([-2.5,0,0.78]),np.array([90,0,-180])),
               Objeto('mario/HK40D7AZNBRT97PMCRL6ICE81.obj',0.3,np.array([0,-1.5,0.35]),np.array([90,0,-90])),
               Objeto('pomo/pomo.obj',0.0005,np.array([[0,0,1],[2,1,2],[-2,1.5,1],[-2,-2.5,0.4]]),np.array([90,0,0]),1,10),
               Objeto('volleyball/volleyball.obj',0.01,np.array([2-0.1,2,0.1]),np.array([0,0,0]))]
    ar_renderer = ARRenderer(cam1,objetos)
    ar_renderer.init(False)
    ar_renderer.run()