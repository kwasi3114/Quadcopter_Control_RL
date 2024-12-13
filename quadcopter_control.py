from os import read
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d

def readTrajectoryFile(trajfile):
  traj = pd.read_csv(trajfile, header=None)
  return traj.to_numpy()

def defineParams(mass, grav, I, arm_length, maxAngle, maxF, minF):
  params = {
    "mass": mass,
    "grav": grav,
    "I": I,
    "invI": np.linalg.inv(I),
    "arm_length": arm_length,
    "maxAngle": maxAngle,
    "maxF": maxF,
    "minF": minF
  }
  return params


def init_state(start, yaw):
    """
    Initialize a 13x1 state vector.

    Parameters:
        start (list or numpy.ndarray): Initial position [x, y, z].
        yaw (float): Initial yaw angle (psi0).

    Returns:
        numpy.ndarray: 13x1 state vector.
    """
    s = np.zeros(13)

    phi0 = 0.0  # Initial roll
    theta0 = 0.0  # Initial pitch
    psi0 = yaw  # Initial yaw

    # Compute initial rotation matrix and quaternion
    Rot0 = rpy_to_rot_zxy(phi0, theta0, psi0)
    Quat0 = rot_to_quat(Rot0)

    # Assign values to the state vector
    s[0] = start[0]  # x
    s[1] = start[1]  # y
    s[2] = start[2]  # z
    s[3] = 0.0       # xdot
    s[4] = 0.0       # ydot
    s[5] = 0.0       # zdot
    s[6] = Quat0[0]  # qw
    s[7] = Quat0[1]  # qx
    s[8] = Quat0[2]  # qy
    s[9] = Quat0[3]  # qz
    s[10] = 0.0      # p (roll rate)
    s[11] = 0.0      # q (pitch rate)
    s[12] = 0.0      # r (yaw rate)

    return s

def LQR_Quad(x, Y, params, Q, R):
    """
    LQR controller for a quadcopter.

    Parameters:
        x (numpy.ndarray): State vector [x, y, z, xd, yd, zd, phi, theta, psi, p, q, r].
        Y (object): Contains trajectory data (Y.y and Y.dy).
        params (object): Parameters with mass, grav, and inertia matrix (I).
        Q (numpy.ndarray): State cost matrix for LQR.
        R (numpy.ndarray): Control input cost matrix for LQR.

    Returns:
        F (float): Total thrust force.
        M (numpy.ndarray): Control torques (moments).
        att (numpy.ndarray): Desired roll, pitch, yaw.
    """
    # Define system matrices
    A = np.block([
        [np.zeros((3, 3)), np.eye(3)],
        [np.zeros((3, 3)), np.zeros((3, 3))]
    ])
    B = (1 / params["mass"]) * np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, params["grav"], 0],
        [0, -1*params["grav"], 0, 0],
        [1, 0, 0, 0]
    ])

    # Solve for the LQR gain matrix K
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P

    # Compute errors
    e_pos = np.array(x[0:3]) - np.array(Y[0][0:3])       # Position error
    e_vel = np.array(x[3:6]) - np.array(Y[1][0:3])      # Velocity error
    xe = np.concatenate((e_pos, e_vel))  # State error

    # Control input
    u = np.array([params["mass"] * params["grav"], 0, 0, 0]) - K @ xe
    F = u[0]            # Total force
    att = u[1:4]        # Desired roll, pitch, yaw

    # Attitude control
    e_att = x[6:9] - att            # Attitude error (phi, theta, psi)
    omega_des = np.array([0, 0, Y[1][3]])  # Desired angular velocity (p, q, r)
    e_omega = x[9:12] - omega_des   # Angular velocity error

    # Compute desired angular acceleration
    #Katt_p = params.Katt_p  # Proportional gain for attitude
    #Katt_d = params.Katt_d  # Derivative gain for attitude
    Katt_p = 5 * np.diag([20, 20, 0.2])
    Katt_d = np.diag([20, 20, 1.2])
    att_ddot_des = -Katt_p @ e_att - Katt_d @ e_omega

    # Compute moments (torques)
    M = params["I"] @ att_ddot_des  # Torques

    return F, M, att


def quadEOM(t, s, F, M, params):
    """
    Solve quadrotor equation of motion.

    Parameters:
        t (float): Time
        s (ndarray): 13x1 state vector [x, y, z, xd, yd, zd, qw, qx, qy, qz, p, q, r]
        F (float): Thrust output from controller (only used in simulation)
        M (ndarray): 3x1 moments output from controller (only used in simulation)
        params (dict): Parameters including mass, gravity, arm_length, maxF, minF, I, invI

    Returns:
        ndarray: 13x1 derivative of state vector s
    """
    # ************ EQUATIONS OF MOTION ************************
    # Limit the force and moments due to actuator limits
    A = np.array([
        [0.25,                      0, -0.5 / params["arm_length"]],
        [0.25,  0.5 / params["arm_length"],                      0],
        [0.25,                      0,  0.5 / params["arm_length"]],
        [0.25, -0.5 / params["arm_length"],                      0]
    ])

    prop_thrusts = A @ np.array([F, M[0], M[1]])  # Not using moment about Z-axis for limits
    prop_thrusts_clamped = np.clip(prop_thrusts, params["minF"] / 4, params["maxF"] / 4)

    B = np.array([
        [1, 1, 1, 1],
        [0, params["arm_length"], 0, -params["arm_length"]],
        [-params["arm_length"], 0, params["arm_length"], 0]
    ])
    F = B[0, :] @ prop_thrusts_clamped
    #M = np.array([B[1:3, :] @ prop_thrusts_clamped, M[2]]).flatten()
    M_clamped = B[1:3, :] @ prop_thrusts_clamped  # Ensure this results in a (2,) shape array
    M = np.array([M_clamped[0], M_clamped[1], M[2]])  # Manually unpack elements and combine with M[2]


    # Assign states
    x, y, z = s[0:3]
    xdot, ydot, zdot = s[3:6]
    qW, qX, qY, qZ = s[6:10]
    p, q, r = s[10:13]

    quat = np.array([qW, qX, qY, qZ])
    bRw = quat_to_rot(quat)
    wRb = bRw.T

    # Acceleration
    accel = (1 / params["mass"]) * (wRb @ np.array([0, 0, F]) - np.array([0, 0, params["mass"] * params["grav"]]))

    # Angular velocity
    K_quat = 2  # Enforces the magnitude 1 constraint for the quaternion
    quaterror = 1 - (qW**2 + qX**2 + qY**2 + qZ**2)
    qdot = (-0.5 * np.array([
        [0, -p, -q, -r],
        [p,  0, -r,  q],
        [q,  r,  0, -p],
        [r, -q,  p,  0]
    ]) @ quat + K_quat * quaterror * quat)

    # Angular acceleration
    omega = np.array([p, q, r])
    pqrdot = params["invI"] @ (M - np.cross(omega, params["I"] @ omega))

    # Assemble sdot
    sdot = np.zeros(13)
    sdot[0:3] = [xdot, ydot, zdot]
    sdot[3:6] = accel
    sdot[6:10] = qdot
    sdot[10:13] = pqrdot

    return sdot

def quat_to_rot(quat):
    """
    Convert a quaternion to a rotation matrix.
    """
    qw, qx, qy, qz = quat
    return np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)]
    ])

import numpy as np

def rot_to_quat(R):
    """
    Converts a rotation matrix into a quaternion.
    """
    tr = R[0, 0] + R[1, 1] + R[2, 2]  # Trace of the rotation matrix

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2  # S = 4 * qw
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    q = np.array([qw, qx, qy, qz])
    q *= np.sign(qw)  # Ensure the quaternion scalar part is non-negative
    return q


def rot_to_rpy_zxy(Rot):
    """Convert a rotation matrix to ZXY Euler angles."""
    phi = np.arcsin(-Rot[2, 1])  # Roll
    theta = np.arctan2(Rot[2, 0], Rot[2, 2])  # Pitch
    psi = np.arctan2(Rot[0, 1], Rot[1, 1])  # Yaw
    return phi, theta, psi

def rpy_to_rot_zxy(phi, theta, psi):
    """
    Converts roll (phi), pitch (theta), and yaw (psi) to a body-to-world
    rotation matrix. The output is a world-to-body rotation matrix [bRw].
    To obtain the body-to-world matrix [wRb], transpose the result.
    """
    R = np.array([
        [np.cos(psi) * np.cos(theta) - np.sin(phi) * np.sin(psi) * np.sin(theta),
         np.cos(theta) * np.sin(psi) + np.cos(psi) * np.sin(phi) * np.sin(theta),
         -np.cos(phi) * np.sin(theta)],
        [-np.cos(phi) * np.sin(psi),
         np.cos(phi) * np.cos(psi),
         np.sin(phi)],
        [np.cos(psi) * np.sin(theta) + np.cos(theta) * np.sin(phi) * np.sin(psi),
         np.sin(psi) * np.sin(theta) - np.cos(psi) * np.cos(theta) * np.sin(phi),
         np.cos(phi) * np.cos(theta)]
    ])
    return R

def state_to_qd(x):
    """
    Converts a state vector `x` used in simulation to a `qd` dictionary
    used in hardware or control systems.

    Args:
        x: A 1x13 numpy array representing state variables
           [pos(3), vel(3), quat(4), omega(3)].

    Returns:
        qd: A dictionary with keys 'pos', 'vel', 'euler', and 'omega'.
    """

    qd = []

    # Extract position and velocity
    qd[0:3] = x[0:3]
    qd[3:6] = x[3:6]

    # Extract and process quaternion
    quat = x[6:10]
    Rot = quat_to_rot(quat)
    phi, theta, psi = rot_to_rpy_zxy(Rot)

    # Store Euler angles and angular velocity
    qd[6:9] = phi, theta, psi
    qd[9:12] = x[10:13]

    return qd


def solveDiffEq(eq, t_span, x0):
  sol = solve_ivp(eq, t_span, x0)
  t = sol.t
  x = sol.y
  return t, x


def calculateMSE(arr1, arr2):
    return mean_squared_error(arr1, arr2)


def getDesiredState(traj, t):
  ts = round(t, 2)
  row = int(ts/0.01)

  Y = [[traj[row, 1], traj[row, 2], traj[row, 3], traj[row, 4]],
       [traj[row, 5], traj[row, 6], traj[row, 7], traj[row, 8]],
       [traj[row, 9], traj[row, 10], traj[row, 11], 0],
       [traj[row, 12], traj[row, 13], traj[row, 14], 0],
       [traj[row, 15], traj[row, 16], traj[row, 17], 0]
  ]

  return Y



def simulate(t, s, params, Q, R, traj):
  current_state = state_to_qd(s)
  desired_state = getDesiredState(traj, t)

  F, M, att = LQR_Quad(current_state, desired_state, params, Q, R)
  sdot = quadEOM(t, s, F, M, params)

  return sdot


def interpolate(actual, desired):
  actual_timestep = np.linspace(0, 10, actual.shape[0])
  desired_timestep = np.linspace(0, 10, desired.shape[0])

  inter = interp1d(desired_timestep, desired, axis=0, kind='linear')
  return inter(actual_timestep)


def plot_trajectories(traj1, traj2, labels=('Desired', 'Actual')):
    """
    Plots two 3D trajectories on the same graph.
    Traj 1 = desired, Traj 2 = actual

    Parameters:
        traj1 (list or numpy.ndarray): First trajectory in the format [(x1, y1, z1), ...].
        traj2 (list or numpy.ndarray): Second trajectory in the format [(x2, y2, z2), ...].
        labels (tuple): Labels for the two trajectories (default is ('Trajectory 1', 'Trajectory 2')).
    """
    # Extract coordinates for trajectory 1
    x1, y1, z1 = zip(*traj1)

    # Extract coordinates for trajectory 2
    x2, y2, z2 = zip(*traj2)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory 1
    ax.plot(x1, y1, z1, label=labels[0], color='blue', linewidth=2)

    # Plot trajectory 2
    ax.plot(x2, y2, z2, label=labels[1], color='red', linewidth=2)

    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectories')
    ax.legend()

    # Show the plot
    plt.show()

def fullRun(Q, R):
  #return pos error given Q and R
  traj = readTrajectoryFile("/content/KDPtraj.csv")
  desired_traj = traj[:,1:4]

  params = defineParams(
      mass=0.32,
      grav=9.81,
      I=[[1.395e-5, 0, 0], [0, 1.436e-5, 0], [0, 0, 2.173e-5]],
      arm_length=0.046,
      maxAngle=(40*np.pi)/180,
      maxF = 2.5*0.32*9.81,
      minF= 0.5*0.32*9.81
  )

  x0 = init_state(traj[0,1:4],0)
  t_span = [0, 10]

  sol = solve_ivp(simulate, t_span, x0, args=(params, Q, R, traj))
  actual_traj = sol.y[:3, :]
  actual_traj = actual_traj.T

  new_des = interpolate(actual_traj, desired_traj)
  return calculateMSE(actual_traj, new_des)



def main():
  #main function
  traj = readTrajectoryFile("/content/KDPtraj.csv")
  desired_traj = traj[:,1:4]

  params = defineParams(
      mass=0.32,
      grav=9.81,
      I=[[1.395e-5, 0, 0], [0, 1.436e-5, 0], [0, 0, 2.173e-5]],
      arm_length=0.046,
      maxAngle=(40*np.pi)/180,
      maxF = 2.5*0.32*9.81,
      minF= 0.5*0.32*9.81
  )

  x0 = init_state(traj[0,1:4],0)
  t_span = [0, 10]

  Q = np.diag([1, 1, 1, 1, 1, 1])
  R = 10*np.eye(4)

  sol = solve_ivp(simulate, t_span, x0, args=(params, Q, R, traj))
  t = sol.t
  x = sol.y

  actual_traj = sol.y[:3, :]
  actual_traj = actual_traj.T

  plot_trajectories(traj[:,1:4], actual_traj, labels=('Desired', 'Actual'))

  #actual_time = traj[:,0].T
  #filtered_traj = []

  #for i in range(len(t)):
  #  if t[i] in actual_time:
  #    #ix = actual_time.index(t[i])
  #    ix = np.where(actual_time == t[i])
  #    filtered_traj.append(desired_traj[ix, 1:4])

  #filtered_traj = np.array(filtered_traj)
  #print(filtered_traj)
  #print(filtered_traj.shape)


  #print("Desired: ")
  #print(traj[:,1:4])
  #print("Actual: ")
  #print(x[:,1:4])
  #print("Time steps: ")
  #print(sol.t)
  #print(traj[:,1:4])
  #print(actual_traj)

  #print("Matrix shapes:")
  #print("x: ", x.shape)
  #print("t: ", t.shape)
  #print("actual traj: ", actual_traj.shape)
  #print("desired traj: ", traj[:,1:4].shape)
  new_des = interpolate(actual_traj, desired_traj)
  print("MSE: " + str(calculateMSE(actual_traj, new_des)))
  #print(new_des)
  #print(new_des.shape)


if __name__=="__main__":
    main()
