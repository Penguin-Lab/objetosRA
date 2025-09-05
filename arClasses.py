#Modified from: https://github.com/cjcarver/OpenGL-OpenCV-AR

import cv2
import os
# --- força PyOpenGL sem accelerate e plataforma WGL ---
os.environ["PYOPENGL_PLATFORM"] = "win32"
import OpenGL
OpenGL.USE_ACCELERATE = False
from PIL import Image
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from threading import Thread
import numpy as np
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

class Camera:
    def __init__(self, arquivo='calib_rt1.npz'):
        self.dados = np.load(arquivo)
        self.imagemNome = None
        self.imagemRaw = None
        self.imagem = None
        self.K = self.dados['K']
        self.dist = self.dados['dist']
        self.roi = self.dados['roi']
        _, _, w, h = self.roi
        self.tamImagem = np.array([h, w])
        self.nK = self.dados['nK']
        self.rt = self.dados['rt']
        self.rt = np.vstack((self.rt, np.array([[0,0,0,1]])))

    def carregaImagem(self, imagemNome):
        self.imagemNome = imagemNome
        self.imagemRaw = cv2.imread(self.imagemNome)
        self.imagem = cv2.undistort(self.imagemRaw, self.K, self.dist, None, self.nK)
        self.imagem = self.imagem[0:self.tamImagem[0], 0:self.tamImagem[1], :]

    def mostraImagem(self):
        cv2.imshow(self.imagemNome, self.imagem)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class ARRenderer:
    def __init__(self, cam, objetos):
        self.cam = cam
        self.objetos = objetos
        self.texture_id = 0
        self.thread_quit = False
        self.cap = cv2.VideoCapture(0)
        self.new_frame = self.cam.imagem#self.cap.read()[1]
        # Matriz da camera
        INV_RT = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0, 0.0],
                           [0.0, 0.0, -1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])
        self.view_matrix = INV_RT @ self.cam.rt
        # Objetos para desenhar
        self.objs = []

    def init(self,drawMode = False):
        self.drawMode = drawMode
        video_thread = Thread(target=self.update, args=())
        video_thread.start()

    def init_gl(self):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glMatrixMode(GL_PROJECTION)
        P = self.KtoP()
        glLoadMatrixd(P.T)
        glMatrixMode(GL_MODELVIEW)
        self.load_objects()

    def KtoP(self, near = 0.1, far = 20.0):
        fx = self.cam.nK[0,0]
        fy = self.cam.nK[1,1]
        cx = self.cam.nK[0,2]
        cy = self.cam.nK[1,2]
        h, w, _ = self.cam.imagem.shape
        left   = -cx*near/fx
        right  =  (w - cx)*near/fx
        bottom = -(h - cy)*near/fy
        top    =  cy*near/fy
        # Matriz estilo glFrustum
        P = np.zeros((4,4))
        P[0,0] = 2*near/(right-left)
        P[0,2] = (right+left)/(right-left)
        P[1,1] = 2*near/(top-bottom)
        P[1,2] = (top+bottom)/ (top-bottom)
        P[2,2] = -(far+near)/(far-near)
        P[2,3] = -2*far*near/(far-near)
        P[3,2] = -1
        return P

    def load_objects(self):
        for objeto in self.objetos:
            obj = OBJ(objeto.name_obj, swapyz=False)
            self.objs.append((obj,objeto.size_obj,objeto.obj_pos,objeto.obj_angle,objeto))
        glEnable(GL_TEXTURE_2D)
        self.texture_id = glGenTextures(1)

    def desenha(self):
        for obj, scale, pos, angle, objeto in self.objs:
            # Matriz do objeto
            T = np.eye(4)
            if pos.shape != (3,):
                if objeto.pos_count_max != 0:
                    objeto.pos_count += 1
                    if objeto.pos_count == objeto.pos_count_max:
                        objeto.pos_count = 0
                        maximo, _ = pos.shape
                        if objeto.pos_vec == maximo-1:
                            objeto.pos_vec = 0
                        else:
                            objeto.pos_vec += 1
                T[:3,3] = pos[objeto.pos_vec,:]
            else:
                T[:3,3] = pos
            rx, ry, rz = np.radians(angle)
            Rx = np.array([[1,0,0,0],[0,np.cos(rx),-np.sin(rx),0],[0,np.sin(rx),np.cos(rx),0],[0,0,0,1]])
            Ry = np.array([[np.cos(ry),0,np.sin(ry),0],[0,1,0,0],[-np.sin(ry),0,np.cos(ry),0],[0,0,0,1]])
            Rz = np.array([[np.cos(rz),-np.sin(rz),0,0],[np.sin(rz),np.cos(rz),0,0],[0,0,1,0],[0,0,0,1]])
            S = np.eye(4)*scale
            S[3,3] = 1.0
            if objeto.count_max != 0:
                objeto.count_estado += 1
                if objeto.count_estado == objeto.count_max:
                    if objeto.estado == 0:
                        objeto.estado = 1
                    else:
                        objeto.estado = 0
                    objeto.count_estado = 0
                if objeto.estado == 0:
                    Rm = np.array([[1,0,0,0],[0,np.cos(np.pi),-np.sin(np.pi),0],[0,np.sin(np.pi),np.cos(np.pi),0],[0,0,0,1]])
                else:
                    Rm = np.eye(4)
                model_matrix = T@Rz@Ry@Rx@Rm@S
            else:
                model_matrix = T@Rz@Ry@Rx@S

            # Matriz completa para o OpenGL
            ModelView = self.view_matrix @ model_matrix
            glPushMatrix()
            glLoadMatrixd(ModelView.T)

            if self.drawMode: # Retira faces com z < 0 no mundo
                verts_world = []
                for v in obj.vertices:
                    v_h = np.array([v[0], v[1], v[2], 1.0])
                    v_w = model_matrix @ v_h  # Vertice no mundo
                    verts_world.append(v_w[:3])

                for face in obj.faces:
                    vertices, normals, texture_coords, material = face

                    verts_f = [verts_world[vi-1] for vi in vertices]
                    if all(v[2] > 0 for v in verts_f):  # Se vertices tem z > 0, desenha a face

                        mtl = obj.mtl[material]
                        if 'texture_Kd' in mtl:
                            glBindTexture(GL_TEXTURE_2D, mtl['texture_Kd'])
                        else:
                            glColor(*mtl['Kd'])

                        glBegin(GL_POLYGON)
                        for i in range(len(vertices)):
                            if normals[i] > 0:
                                glNormal3fv(obj.normals[normals[i] - 1])
                            if texture_coords[i] > 0:
                                glTexCoord2fv(obj.texcoords[texture_coords[i] - 1])
                            glVertex3fv(obj.vertices[vertices[i] - 1])
                        glEnd()
            else:
                obj.render()

            glPopMatrix()

    def update(self):
        while True:
            self.new_frame = self.cam.imagem#self.cap.read()[1]
            if self.thread_quit:
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def draw_gl_scene(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
        glLoadIdentity()
        frame = self.new_frame

        glDisable(GL_DEPTH_TEST)
        self.convert_image_to_texture(frame)
        self.draw_background()
        glEnable(GL_DEPTH_TEST)

        self.desenha()
        glutSwapBuffers()

    def convert_image_to_texture(self, frame):
        tx_image = cv2.flip(frame, 0)
        tx_image = Image.fromarray(tx_image)
        ix, iy = tx_image.size[0], tx_image.size[1]
        tx_image = tx_image.tobytes('raw', 'BGRX', 0, -1)

        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, tx_image)

    def draw_background(self, near = 0.1, far = 20.0):
        fx = self.cam.nK[0,0]
        fy = self.cam.nK[1,1]
        cx = self.cam.nK[0,2]
        cy = self.cam.nK[1,2]
        h, w, _ = self.cam.imagem.shape
        left   = -cx*near/fx
        right  =  (w - cx)*near/fx
        bottom = -(h - cy)*near/fy
        top    =  cy*near/fy

        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glPushMatrix()

        glTranslatef(0.0, 0.0, -near)  # Coloca o quad dentro do frustum

        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0); glVertex3f(left, bottom, 0.0)
        glTexCoord2f(1.0, 1.0); glVertex3f(right, bottom, 0.0)
        glTexCoord2f(1.0, 0.0); glVertex3f(right, top, 0.0)
        glTexCoord2f(0.0, 0.0); glVertex3f(left, top, 0.0)
        glEnd()

        glPopMatrix()
        
    def run(self):
        glutInit(sys.argv)
        h, w, _ = self.cam.imagem.shape
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(w, h)
        window = glutCreateWindow(b'AR camera 1')
        glutDisplayFunc(self.draw_gl_scene)
        glutIdleFunc(self.draw_gl_scene)
        glutKeyboardFunc(self.key_pressed)
        self.init_gl()
        glutMainLoop()

    def key_pressed(self, key, x, y):
        key = key.decode("utf-8") 
        if key == "q":
            self.thread_quit = True
            cv2.destroyAllWindows()
            glutLeaveMainLoop()

class Objeto:
    def __init__(self, nome = 'objeto.obj', tamanho = 1.0, pos = np.array([0,0,0]), angulo = np.array([0,0,0]), count_max = 0, pos_count_max = 0):
        self.name_obj = nome
        self.size_obj = tamanho
        self.obj_pos = pos
        self.obj_angle = angulo
        self.estado = 0
        self.count_estado = 0
        self.count_max = count_max
        self.pos_vec = 0
        self.pos_count = 0
        self.pos_count_max = pos_count_max
        
"""
Modificado de: https://github.com/yarolig/OBJFileLoader/blob/master/OBJFileLoader/objloader.py
"""
class OBJ:
    generate_on_init = True
    @classmethod
    def loadTexture(cls, imagefile):
        surf = pygame.image.load(imagefile)
        image = pygame.image.tostring(surf, 'RGBA', 1)
        ix, iy = surf.get_rect().size
        texid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texid)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
        return texid

    @classmethod
    def loadMaterial(cls, filename):
        contents = {}
        mtl = None
        dirname = os.path.dirname(filename)

        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'newmtl':
                mtl = contents[values[1]] = {}
            elif mtl is None:
                raise ValueError("mtl file doesn't start with newmtl stmt")
            elif values[0] == 'map_Kd':
                # load the texture referred to by this declaration
                mtl[values[0]] = values[1]
                imagefile = os.path.join(dirname, mtl['map_Kd'])
                mtl['texture_Kd'] = cls.loadTexture(imagefile)
            else:
                mtl[values[0]] = list(map(float, values[1:]))
        return contents

    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        self.gl_list = 0
        dirname = os.path.dirname(filename)

        material = None
        print(filename)
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(list(map(float, values[1:3])))
            elif values[0] in ('usemtl', 'usemat'):
                material = values[1]
            elif values[0] == 'mtllib':
                self.mtl = self.loadMaterial(os.path.join(dirname, values[1]))
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords, material))
        if self.generate_on_init:
            self.generate()

    def generate(self):
        self.gl_list = glGenLists(1)
        glNewList(self.gl_list, GL_COMPILE)
        #glEnable(GL_TEXTURE_2D)
        glFrontFace(GL_CCW)
        for face in self.faces:
            vertices, normals, texture_coords, material = face

            mtl = self.mtl[material]
            if 'texture_Kd' in mtl:
                # use diffuse texmap
                glBindTexture(GL_TEXTURE_2D, mtl['texture_Kd'])
            else:
                # just use diffuse colour
                glColor(*mtl['Kd'])

            glBegin(GL_POLYGON)
            for i in range(len(vertices)):
                if normals[i] > 0:
                    glNormal3fv(self.normals[normals[i] - 1])
                if texture_coords[i] > 0:
                    glTexCoord2fv(self.texcoords[texture_coords[i] - 1])
                glVertex3fv(self.vertices[vertices[i] - 1])
            glEnd()
        #glDisable(GL_TEXTURE_2D)
        glEndList()

    def render(self):
        glCallList(self.gl_list)

    def free(self):
        glDeleteLists([self.gl_list])