import numpy as np
from scipy.interpolate import UnivariateSpline, make_interp_spline

def _gaussian(x, a, b, c):
    return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))
#def _gaussian(x, a, b, c):# Really a lorentzian, just checking
#    return a * (gam/2)**2 / ( (gam/2)**2 + ( x - b )**2)

def _multi_gaussian(x, params):
    template = np.zeros_like(x)
    for param in params:
        template += _gaussian(x, *param)
    return template

def _monomial(x,a,n):
    return a*x**n

def _polynomial(x,params):
    template = np.zeros_like(x)
    for param in params:
        template += _monomial(x, *param)
    return template

def _binary_search(nknots,x,y,dy,maxiter=50):
    """
    Binary search for the smoothness parameter "s" to get a spline with nknots.
    """
    m = len(x)
    std = np.mean(np.sqrt(dy))
    low = np.log10((m - np.sqrt(2*m)) * std**2)-100
    hi = np.log10((m + np.sqrt(2*m)) * std**2)+100
    print([hi,low])
    #hi,low = 100, -10
    for _ in range(maxiter):
        s = 10**((hi+low)/2)
        #print(s)
        ss = UnivariateSpline(x,y,k=3,s=s)
        if len(ss.get_knots()) < nknots:
            hi = np.log10(s)
        elif len(ss.get_knots()) > nknots:
            low = np.log10(s)
        else:
            return s,ss
    return s,ss

def get_spline(knots,edges,wl):
    xvals = [wl[0]] + list(knots[:,0]) + [wl[-1]]
    yvals = [edges[0,0]] + list(knots[:,1]) + [edges[0,1]]
    interp_model = make_interp_spline(xvals,yvals,k=3, axis=-1)
    return interp_model

def Radial_Coordinate(p, e, Phi_r):
    return p/(1 + e * np.cos(Phi_r))

def Polar_Coordinate(x, Phi_theta):
    return np.arccos(np.sqrt(1-x**2)*np.cos(Phi_theta))
    #return np.arccos(x*np.cos(Phi_theta))

def sph2cart(r,theta,phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x,y,z

def bl2kep2(t,a,p,e,x,Phi_r, Phi_theta, Phi_phi):
    rr = Radial_Coordinate(p, e, Phi_r)
    tt = Polar_Coordinate(x, Phi_theta)
    return [t, *sph2cart(rr,tt,Phi_phi)]

def bl2kep(t, a, p, e, x, Phi_r, Phi_theta, Phi_phi):
    # Using https://arxiv.org/abs/2411.04955
    bl_angles=[]
    for i in np.arange(1,len(t)): #skipping the first, it always gives nan...
        AA = kac.pyBoyerLindquistPhasesToDarwinPhases(a,p[i],e[i],x[i],Phi_r[i],Phi_theta[i],Phi_phi[i])
        if np.any(np.isnan(AA)):
            print("ALERT! Proper BoyerLidquist2Darwin failed, using less accurate version!!!")
            return bl2kep2(t,a,p,e,x,Phi_r, Phi_theta, Phi_phi)
        else:
            bl_angles.append(AA)
    psi_from_AA,chi_from_AA,phi_from_AA = np.array(bl_angles).T
    tt = t[1:len(bl_angles)+1]
    pp = p[1:len(bl_angles)+1]
    ee = e[1:len(bl_angles)+1]
    xx = x[1:len(bl_angles)+1]
    r_from_AA = Radial_Coordinate(pp, ee, psi_from_AA)
    theta_from_AA = Polar_Coordinate(xx, chi_from_AA)
    x_coord_from_AA, y_coord_from_AA, z_coord_from_AA = sph2cart(r_from_AA ,theta_from_AA, phi_from_AA)
    trajectory = [tt,x_coord_from_AA,y_coord_from_AA,z_coord_from_AA]
    return trajectory

def Keplerian_p_from_f(M, mu, f_orb_0, e0):
    return c**2*(1-e0**2)/(2*np.pi*f_orb_0*G*(M+mu)*MSUN)**(2/3)

def intersect_trajectory_with_disk(trajectory, disk_center, disk_normal, disk_radius):
    intersections = []
    disk_normal = np.array(disk_normal) / np.linalg.norm(disk_normal)  # Ensure unit normal
    disk_center = np.array(disk_center)
    trajectory = np.array(trajectory).T
    for i in range(len(trajectory) - 1):
        p1, p2 = np.array(trajectory[i]), np.array(trajectory[i + 1])
        segment = p2 - p1
        denom = np.dot(segment[1:], disk_normal)
        if np.isclose(denom, 0, atol=1e-8):
            print("Segment is parallel to the disk plane, skipping!") # Should be better about this!!!
            continue  
        t = np.dot(disk_center - p1[1:], disk_normal) / denom
        if 0 <= t <= 1:
            intersection = p1 + t * segment
            if np.linalg.norm(intersection[1:] - disk_center) <= disk_radius:
                intersections.append(intersection.tolist())
    
    return np.array(intersections)

def intersect_trajectory_with_rotating_disk(trajectory, disk_center0, disk_normal0, disk_radius, omega, tol=1e-8, max_iter=50, num_samples=200):
    """
    Finds intersections between a trajectory and a disk that rotates about the z-axis 
    with angular frequency omega.
    
    The disk is defined in its body frame (time zero) by:
      - disk_center0: center (3-vector)
      - disk_normal0: unit normal (3-vector)
      - disk_radius: scalar radius
      
    At time t the disk is rotated in the inertial frame as:
      disk_center(t) = Rz(omega*t) @ disk_center0
      disk_normal(t) = Rz(omega*t) @ disk_normal0
      
    A trajectory point p=(t,x,y,z) lies in the disk plane at time t if, after rotating by -omega*t,
    its spatial part p_body satisfies:
      (p_body - disk_center0) · disk_normal0 = 0.
    
    For non-coplanar segments (i.e. crossing the plane) a bisection routine is used to locate the unique crossing.
    If a segment is (nearly) coplanar, then the entire segment lies in the disk plane, and we search
    for intersections of the segment with the disk’s circular boundary.
    
    Parameters:
      trajectory: list or tuple of 4 array-like objects [t, x, y, z] (each with shape (N,)).
      disk_center0: array-like, initial center of the disk (3-vector).
      disk_normal0: array-like, initial normal of the disk (3-vector; will be normalized).
      disk_radius: scalar, radius of the disk.
      omega: angular frequency of the disk's rotation about the z-axis.
      tol: tolerance for root finding.
      max_iter: maximum iterations for bisection.
      num_samples: number of samples used in the coplanar case.
      
    Returns:
      intersections: np.array of intersection points (each is (t,x,y,z)).
    """
    intersections = []
    disk_normal0 = np.array(disk_normal0, dtype=float)
    disk_normal0 /= np.linalg.norm(disk_normal0)
    disk_center0 = np.array(disk_center0, dtype=float)

    # Expect trajectory to be [t array, x array, y array, z array].
    if len(trajectory) != 4:
        raise ValueError("Trajectory must be a list of 4 arrays: [t, x, y, z].")
    traj = np.column_stack(trajectory)

    # Inline rotation: computes Rz(-omega*tau)*pos for a spatial 3-vector.
    def rotate_z_neg(angle, pos):
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([c * pos[0] + s * pos[1],
                         -s * pos[0] + c * pos[1],
                         pos[2]])

    # Process each segment.
    for i in range(len(traj) - 1):
        p1 = traj[i]
        p2 = traj[i + 1]
        dt = p2[0] - p1[0]
        if dt == 0:
            continue  # Skip segments with zero time difference

        pos_diff = p2[1:] - p1[1:]
        t_diff = dt

        # f(lam) measures how far the rotated spatial point is from the disk plane.
        def f(lam):
            tau = p1[0] + lam * t_diff
            pos = p1[1:] + lam * pos_diff
            pos_body = rotate_z_neg(omega * tau, pos)
            return np.dot(pos_body - disk_center0, disk_normal0)
        
        f0 = f(0.0)
        f1 = f(1.0)

        # Check if the segment is nearly coplanar with the disk.
        if np.abs(f0) < tol and np.abs(f1) < tol:
            # Define g(lam): distance from disk center (in body frame) minus disk_radius.
            def g(lam):
                tau = p1[0] + lam * t_diff
                pos = p1[1:] + lam * pos_diff
                pos_body = rotate_z_neg(omega * tau, pos)
                return np.linalg.norm(pos_body - disk_center0) - disk_radius

            # Sample g(lam) over the segment.
            lam_vals = np.linspace(0, 1, num_samples)
            g_vals = np.array([g(lam) for lam in lam_vals])

            # Look for sign changes or near-zero values.
            for j in range(len(lam_vals) - 1):
                if np.abs(g_vals[j]) < tol:
                    lam_root = lam_vals[j]
                elif g_vals[j] * g_vals[j+1] < 0:
                    lam_low = lam_vals[j]
                    lam_high = lam_vals[j+1]
                    g_low = g_vals[j]
                    for _ in range(max_iter):
                        lam_mid = (lam_low + lam_high) / 2.0
                        g_mid = g(lam_mid)
                        if np.abs(g_mid) < tol:
                            break
                        if g_low * g_mid < 0:
                            lam_high = lam_mid
                        else:
                            lam_low = lam_mid
                            g_low = g_mid
                    lam_root = (lam_low + lam_high) / 2.0
                else:
                    continue
                tau_sol = p1[0] + lam_root * t_diff
                pos_sol = p1[1:] + lam_root * pos_diff
                intersection = np.hstack(([tau_sol], pos_sol))
                if not any(np.allclose(intersection, x, atol=tol) for x in intersections):
                    intersections.append(intersection)
            continue

        # For non-coplanar segments, check endpoints and then use bisection.
        if np.abs(f0) < tol:
            lam_sol = 0.0
        elif np.abs(f1) < tol:
            lam_sol = 1.0
        elif f0 * f1 > 0:
            continue  # No crossing.
        else:
            lam_low = 0.0
            lam_high = 1.0
            f_low = f0
            lam_sol = None
            for _ in range(max_iter):
                lam_mid = (lam_low + lam_high) / 2.0
                f_mid = f(lam_mid)
                if np.abs(f_mid) < tol:
                    lam_sol = lam_mid
                    break
                if f_low * f_mid < 0:
                    lam_high = lam_mid
                else:
                    lam_low = lam_mid
                    f_low = f_mid
            if lam_sol is None:
                lam_sol = (lam_low + lam_high) / 2.0

        if lam_sol < 0 or lam_sol > 1:
            continue

        tau_sol = p1[0] + lam_sol * t_diff
        pos_sol = p1[1:] + lam_sol * pos_diff
        pos_body = rotate_z_neg(omega * tau_sol, pos_sol)
        if np.linalg.norm(pos_body - disk_center0) <= disk_radius:
            intersection = np.hstack(([tau_sol], pos_sol))
            intersections.append(intersection)
    return np.vstack(intersections) if intersections else np.array([])


def plot_inclined_disk(ax, center, radius, normal, num_points=100, color='red', alpha=0.3):
    """
    Plots a transparent disk in 3D, oriented according to the given normal vector.
    
    Parameters:
      ax        : The 3D axes to plot on.
      center    : Tuple (x, y, z) for the disk center.
      radius    : Radius of the disk.
      normal    : The desired normal vector (as an array-like of 3 numbers).
      num_points: Number of points used to sample the circle.
      color     : Disk color.
      alpha     : Transparency (0 is fully transparent, 1 is opaque).
    """
    # Create a circle in the xy-plane (z = 0)
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros_like(x)
    points = np.vstack((x, y, z))  # shape: (3, num_points)
    
    # Normalize the input normal vector
    normal = np.array(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    
    # The default disk normal is along the z-axis
    z_axis = np.array([0, 0, 1])
    
    # Compute the rotation axis (cross product) and angle (via dot product)
    v = np.cross(z_axis, normal)
    s = np.linalg.norm(v)
    c = np.dot(z_axis, normal)
    # If s is close to zero, the normals are parallel.
    if np.isclose(s, 0):
        if c < 0:
            # 180 degree rotation: flip x and y.
            R = np.array([[-1,  0,  0],
                          [ 0, -1,  0],
                          [ 0,  0,  1]])
        else:
            R = np.eye(3)
    else:
        # Rodrigues' rotation formula components
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s**2))
    
    # Rotate the circle points
    points_rot = R.dot(points)
    
    # Translate points to the specified center
    points_rot[0, :] += center[0]
    points_rot[1, :] += center[1]
    points_rot[2, :] += center[2]
    
    # Create a 3D polygon from the points and add it to the axes
    poly = Poly3DCollection([points_rot.T], color=color, alpha=alpha,edgecolors='none')
    ax.add_collection3d(poly)

def gaussian_peaks(t, crossing_times, width=0.05):
    """
    Generates a sum of Gaussian peaks centered at given crossing times.

    Parameters:
    t : array-like
        Time values for evaluation.
    crossing_times : list or array
        List of times where Gaussians should peak.
    width : float
        Standard deviation of the Gaussians (controls peak width).

    Returns:
    array:
        Sum of Gaussian peaks evaluated at t.
    """
    t = np.asarray(t)
    result = np.zeros_like(t)

    for crossing in crossing_times:
        result += np.exp(-((t - crossing) ** 2) / (2 * width ** 2)) #* np.exp(1j * width * (t - crossing)).real

    return result


