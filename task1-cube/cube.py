from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from math import tan, pi

window_size = 600
mode = 1  # 0 - orthographical, 1..3 - perspective


def key_press(*args):
    global mode
    if args[0] == '\033':
        sys.exit()
    elif '0' <= args[0] <= '9':
        mode = ord(args[0]) - ord('0')


def draw():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glViewport(0, 0, window_size, window_size)
    glLoadIdentity()

    h = 100  # cube height; however, the view is centered and scaled always the same

    if mode == 0:
        glOrtho(-2 * h, 2 * h, -2 * h, 2 * h, -2 * h, 2 * h)
    else:
        view_angle = 30 * mode
        focus_z = - (h + 2 * h / tan(pi * view_angle / 360))
        gluPerspective(view_angle, 1, 0, 2 * h)
        glTranslatef(0, 0, focus_z)

    # start drawing edges
    glBegin(GL_QUADS)

    # back (front missing)
    glColor3ub(0x64, 0xFF, 0xDA)
    glVertex3f(h, -h, -h)
    glVertex3f(-h, -h, -h)
    glVertex3f(-h, h, -h)
    glVertex3f(h, h, -h)

    # top
    glColor3ub(0x29, 0x79, 0xFF)
    glVertex3f(h, h, -h)
    glVertex3f(-h, h, -h)
    glVertex3f(-h, h, h)
    glVertex3f(h, h, h)

    # bottom
    glColor3ub(0x67, 0x3A, 0xB7)
    glVertex3f(h, -h, h)
    glVertex3f(-h, -h, h)
    glVertex3f(-h, -h, -h)
    glVertex3f(h, -h, -h)

    # left
    glColor3ub(0x76, 0xFF, 0x03)
    glVertex3f(-h, h, h)
    glVertex3f(-h, h, -h)
    glVertex3f(-h, -h, -h)
    glVertex3f(-h, -h, h)

    # right
    glColor3ub(0xFF, 0x17, 0x44)
    glVertex3f(h, h, -h)
    glVertex3f(h, h, h)
    glVertex3f(h, -h, h)
    glVertex3f(h, -h, -h)

    glEnd()
    glutSwapBuffers()


def main():
    # init
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(window_size, window_size)
    glutInitWindowPosition((glutGet(GLUT_SCREEN_WIDTH) - window_size) / 2,
                           (glutGet(GLUT_SCREEN_HEIGHT) - window_size) / 2)  # center the window

    glutCreateWindow("Cube (press 0..3)")

    glutDisplayFunc(draw)
    glutIdleFunc(draw)
    glutKeyboardFunc(key_press)

    glutMainLoop()


main()
