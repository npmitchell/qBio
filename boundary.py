#!/usr/bin/env python
bsum = sum
from numpy import *
import numpy as np
from scipy import spatial
import time
import mesh

_TICK = time.time()


def tick():
    global _TICK
    _TICK = time.time()


def tock(s=''):
    global _TICK
    t = time.time()
    print(s + '%0.2f s' % (t - _TICK))
    _TICK = t


def shift(a, n=1):
    return a[(arange(len(a)) + n) % len(a)]


def delta(a):
    return shift(a) - a


def D(x):
    x = array(x)
    return 0.5 * (shift(x, +1) - shift(x, -1))


def norm_ang(a):
    return (a + pi) % (2 * pi) - pi


def xy_angle(a):
    return arctan2(a[:, 1], a[:, 0])


def torus_knot(p=2, q=3, a=0.25, npts=100, phi_s=None, phi_e=None):
    if phi_s is None:
        phi = 2 * pi * arange(npts) / npts
    else:
        phi = linspace(phi_s, phi_e, npts)

    r = 1 + a * sin(q * phi)

    x = r * cos(p * phi)
    y = r * sin(p * phi)
    z = -a * cos(q * phi)

    return array((x, y, z)).T


def join_points(*points):
    d = max(p.shape[1] if p is not None else 0 for p in points)
    N = sum(p.shape[0] if p is not None else 0 for p in points)
    x = zeros((N, d))
    n = 0
    for p in points:
        if p is not None:
            x[n:n + p.shape[0], :p.shape[1]] = p
            n += p.shape[0]
    return x


def join_points_3D(*points):
    # for p in points: print p.shape #Debugging
    d = 3
    N = sum(p.shape[0] if p is not None else 0 for p in points)
    x = zeros((N, d))
    n = 0
    for p in points:
        if p is not None:
            x[n:n + p.shape[0], :p.shape[1]] = p
            n += p.shape[0]
    return x


class Boundary(object):
    def __init__(self, *paths):
        self.paths = map(array, paths)

    def inverse(self):
        return Boundary(*[p[::-1].copy() for p in self.paths])

    def __add__(self, other):
        if isinstance(other, Boundary):
            return Boundary(*(self.paths + other.paths))

    def __sub__(self, other):
        if isinstance(other, Boundary):
            return Boundary(*(self.paths + other.inverse().paths))

    def winding_number(self, p, absolute=False):
        wn = 0
        for path in self.paths:
            theta = xy_angle((path[:, :2] - p[:2]))
            dtheta = norm_ang(delta(theta))
            dtheta = dtheta.sum()
            if absolute: dtheta = abs(dtheta)
            wn += dtheta
        return int(round(wn / (2 * pi)))

    def winding_number_abs(self, p):
        return self.winding_number(p, absolute=True)

    def clip_points(self, p, offset=0):
        """"""
        wn = array(map(self.winding_number, p))
        return p[where(wn > offset)]

    def mesh_cap(self, offset=0, direction=1, ignore_wind=False):
        points = join_points_3D(*self.paths)

        if ignore_wind:
            triangles = filter(
                lambda t: abs(self.winding_number(points[t].mean(0))) > offset,
                spatial.Delaunay(points[:, :2]).vertices
            )

        else:
            triangles = filter(
                lambda t: self.winding_number(points[t].mean(0)) > offset,
                spatial.Delaunay(points[:, :2]).vertices
            )

        m = mesh.Mesh(points, triangles)
        m.force_z_normal(direction)

        return m

    def make_mesh2(self, points=None, fixed_points=None, offset=0, min_dist=None, relax=None, extrude=None,
                   verbose=False, direction=1):
        """Create a 3d mesh from the supplied points or from self (Boundary instance)

        Parameters
        ----------
        points
        fixed_points
        offset
        min_dist
        relax
        extrude
        verbose
        dir

        Returns
        -------

        """
        all_points = join_points_3D(*(self.paths + [fixed_points]))
        n_bound = len(all_points)

        tick()
        if verbose:
            print('building mesh...')

        if points is not None:
            # Here, we add a third dimension to points
            if points.shape[1] == 2:
                if verbose:
                    print('adding 3rd dimension to points...')
                points = hstack((points, zeros((len(points), 1))))

            # check
            # print 'supplied points = ', points
            # import matplotlib.pyplot as plt
            # plt.close('all')
            # plt.scatter(points[:, 0], points[:, 1], c=points[:, 2])
            # plt.show()

            points = self.clip_points(points, offset)

            if min_dist is not None:
                ap3 = all_points.reshape(len(all_points), 1, 3)[:, :, :2]
                p3 = points.reshape(1, len(points), 3)[:, :, :2]
                points = points[where(
                    sum((ap3 - p3) ** 2, 2).min(0) > (min_dist ** 2)
                )[0]]

            all_points = join_points_3D(all_points, points)
        if verbose:
            tock('   prepared %d points: ' % len(all_points))

        # print all_points.shape
        d = spatial.Delaunay(all_points[:, :2])
        triangles = filter(
            lambda t: self.winding_number(all_points[t].mean(0)) > offset, d.vertices
        )
        if verbose:
            tock('   delaunay triangulation: ')

        m = mesh.Mesh(all_points, triangles)
        # print m.points.shape
        # print m.triangles.shape
        m.force_z_normal(direction=direction)

        if relax:
            fixed = arange(n_bound)
            m.relax_z(fixed, relax)
        if verbose:
            tock('   mesh relaxation: ')

        if extrude is not None:
            npts = len(m.points)
            edge_tris = []
            n0 = 0
            for pp in self.paths:
                pip = len(pp)

                for j in range(pip):
                    j1 = n0 + j
                    j2 = n0 + (j + 1) % pip
                    j3 = j1 + npts
                    j4 = j2 + npts

                    edge_tris += [(j3, j2, j1), (j2, j3, j4)]

                n0 += pip

            print('m.points = ', m.points)
            import matplotlib.pyplot as plt
            plt.close('all')
            plt.scatter(m.points[:, 0], m.points[:, 1], c=m.points[:, 2])
            plt.show()

            m = mesh.Mesh(
                vstack((m.points, m.points - (0, 0, extrude))),
                vstack((m.triangles, m.triangles[:, ::-1] + npts, edge_tris))
            )
            # print 'np.max(m.points) = ', np.max(m.points)
            # sys.exit()

        if verbose:
            # tock('   extrusion: ')
            print('done!')

        return m

    def make_mesh(self, points=None, fixed_points=None, offset=0, min_dist=None, relax=None, extrude=None,
                  verbose=True):
        """

        Parameters
        ----------
        points
        fixed_points
        offset
        min_dist
        relax
        extrude
        verbose

        Returns
        -------

        """
        all_points = join_points_3D(*self.paths)
        n_bound = len(all_points)

        tick()
        if verbose:
            print('building mesh...')

        if points is not None:
            if points.shape[1] == 2:
                points = hstack((points, zeros((len(points), 1))))
            points = self.clip_points(points, offset)

            if min_dist is not None:
                ap3 = all_points.reshape(len(all_points), 1, 3)[:, :, :2]
                p3 = points.reshape(1, len(points), 3)[:, :, :2]
                points = points[where(
                    sum((ap3 - p3) ** 2, 2).min(0) > (min_dist ** 2)
                )[0]]

            all_points = join_points_3D(all_points, points)
        if verbose:
            tock('   prepared %d points: ' % len(all_points))

        # print all_points.shape
        d = spatial.Delaunay(all_points[:, :2])
        triangles = filter(
            lambda t: self.winding_number(all_points[t].mean(0)) > offset, d.vertices
        )
        if verbose:
            tock('   delaunay triangulation: ')

        m = mesh.Mesh(all_points, triangles)
        #        print m.points.shape
        #        print m.triangles.shape
        m.force_z_normal()

        if relax:
            fixed = arange(n_bound)
            if fixed_points is not None:
                fixed = hstack((fixed, fixed_points + n_bound))

            m.relax_z(fixed, relax)
        if verbose:
            tock('   mesh relaxation: ')

        if extrude is not None:
            npts = len(m.points)

            edge_tris = []
            n0 = 0
            for pp in self.paths:
                pip = len(pp)

                for j in range(pip):
                    j1 = n0 + j
                    j2 = n0 + (j + 1) % pip
                    j3 = j1 + npts
                    j4 = j2 + npts

                    edge_tris += [(j3, j2, j1), (j2, j3, j4)]

                n0 += pip

            m = mesh.Mesh(
                vstack((m.points, m.points - (0, 0, extrude))),
                vstack((m.triangles, m.triangles[:, ::-1] + npts, edge_tris))
            )
        if verbose:
            # tock('   extrusion: ')
            print('done!')

        return m

    def outer_trace(self, offset=1E-6):
        b = Boundary()
        segments = []

        for p in self.paths:
            T = D(p)
            N = T.copy()
            N[:, 0] = -T[:, 1]
            N[:, 1] = T[:, 1]

            wn = array(map(self.winding_number_abs, p + offset * N)) + \
                 array(map(self.winding_number_abs, p - offset * N))

            wn //= 2

            N = len(p)
            for i in range(N):  # Find the beginning of an exterior path
                im = (i - 1) % N
                if wn[im] != 0 and wn[i] == 0:
                    break

            else:  # This path is completely exterior, just add it.
                b.paths.append(p)
                continue

            # Break into segments
            p = vstack((p[i:], p[:i]))
            wn = hstack((wn[i:], wn[:i]))

            for i in arange(N):
                im = (i - 1) % N
                if wn[im] != 0 and wn[i] == 0:
                    start = i
                    startm = im

                if wn[im] == 0 and wn[i] != 0:
                    segments.append([p[start:i], p[startm] - p[start], p[im] - p[i]])

        for s in segments:
            print(s[0][0], s[0][-1])

        while segments:
            d = map(lambda s: (sum(segments[0][0][-1][:2] - s[0][0][:2]) ** 2), segments)
            # print d
            i = argmin(d)

            p_int = find_2d_intersection(segments[0][0][-1], segments[0][2], segments[i][0][0], segments[i][1])

            s = segments.pop(i)
            s[0] = vstack((p_int, s[0]))

            if i == 0:
                b.paths.append(s[0])

            else:
                segments[0][0] = vstack((segments[0][0], s[0]))
                segments[0][2] = s[2]

        return b

    def plot_paths(self, plot_func, *args, **kwargs):
        for p in self.paths:
            x, y = vstack((p, p[0])).T
            plot_func(x, y, *args, **kwargs)


def circle(c, r, npts=100):
    theta = arange(npts, dtype='d') / npts * 2 * pi
    p = zeros((npts, len(c)))  # c might be 3D!
    p[:] = c
    p[:, 0] += r * cos(theta)
    p[:, 1] += r * sin(theta)
    return Boundary(p)


def hex_grid(r, a):
    npts = int(r // a)
    xi, yi = mgrid[-npts:npts + 1, -npts:npts + 1]
    xi = xi.flatten()
    xi.shape += (1,)
    yi = yi.flatten()
    yi.shape = xi.shape
    xo = array((1, 0)) * a
    yo = array((0.5, sqrt(3.) / 2.)) * a

    return (xi * xo + yi * yo)[where(abs(xi + yi) <= npts)[0]]


def find_2d_intersection(p0, d0, p1, d1):
    x0, y0 = p0[:2]
    x1, y1 = p1[:2]
    dx0, dy0 = d0[:2]
    dx1, dy1 = d1[:2]

    a = (dy1 * x0 - dy1 * x1 - dx1 * y0 + dx1 * y1) / (dx1 * dy0 - dx0 * dy1)
    return p0 + a * d0


if __name__ == '__main__':
    from pylab import plot, show, fill


    def list_plot(p, *args, **kwargs):
        plot(p[:, 0], p[:, 1], *args, **kwargs)


    def list_fill(p, *args, **kwargs):
        fill(p[:, 0], p[:, 1], *args, **kwargs)


    x, y = mgrid[-1:1:.1, -1:1:.1]
    p = array((x.flatten(), y.flatten())).T

    # b = circle((0, 0), 1.0) + circle((1, 0), 1.0) + circle((0.5, 1), 0.5)
    b = Boundary(torus_knot(3, 5, npts=200))
    #    b = circle((1, 0), 1.0)
    #    for a in p:
    #        print a, b.winding_number(a)

    # print b.clip_points(p)

    for p in b.paths:
        list_plot(p, 'r')
    # list_plot(b.paths[1], 'y')

    ot = b.outer_trace()
    # print ot.paths

    for p in ot.paths:
        list_plot(p, 'g-x')
    # list_plot(b.clip_points(p), 'bx')

    show()
