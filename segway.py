import gym
import numpy as np
from gym import spaces, logger
from gym.utils import seeding
import matplotlib.pyplot as plt
import matplotlib as mpl

print("Starting the Segway Environment")

class SegwayEnv(gym.Env):

    def __init__(self):
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.4
        self.total_mass = self.mass_cart + self.mass_pole
        self.length = 0.5  # half the pole's length
        self.polemass_length = self.mass_pole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 15 degrees
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max],
            dtype=np.float32)

        self.action_space = spaces.Box(-self.force_mag, self.force_mag, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.figure, self.ax = None, None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        x, x_dot, theta, theta_dot = self.state
        force = action[0]
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # Dynamics for a Segway-like vehicle
        torque = force * self.length  # Torque proportional to force and distance from pivot

        # Equations of motion
        thetaacc = (self.gravity * sintheta + costheta * torque / self.length -
                    self.polemass_length * theta_dot**2 * sintheta / self.length) / \
                   (self.length * (4.0/3.0 - self.mass_pole * costheta**2 / self.total_mass))

        xacc = (torque - self.polemass_length * thetaacc * costheta) / self.total_mass

        # Update the state variables
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)

        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians

        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # Cart position on y-axis
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.figure is None:
            self.figure, self.ax = plt.subplots()
            self.ax.set_xlim(0, screen_width)
            self.ax.set_ylim(0, screen_height)
            # dont show axis
            self.ax.axis('off')
            self.cart = plt.Rectangle((0, 0), cartwidth, cartheight, fill=True, color='blue')
            self.ax.add_patch(self.cart)
            self.pole = plt.Rectangle((0, 0), polewidth, polelen, fill=True, color='red')
            self.ax.add_patch(self.pole)

        # Update positions
        cartx = self.state[0] * scale + screen_width / 2.0 - cartwidth / 2.0
        self.cart.set_xy((cartx, carty))
        self.pole.set_xy((cartx + cartwidth / 2.0 - polewidth / 2.0, carty + cartheight))

        # Rotate the pole
        pole_angle = -self.state[2]
        t = mpl.transforms.Affine2D().rotate_deg_around(cartx + cartwidth / 2.0, carty + cartheight, np.rad2deg(pole_angle)) + self.ax.transData
        self.pole.set_transform(t)

        # Redraw the figure
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        plt.pause(0.01)  # Pause for a brief moment to update plots

    def close(self):
        # if self.viewer:
        #     self.viewer.close()
        #     self.viewer = None
        if self.figure:
            plt.close(self.figure)

env = SegwayEnv()
state = env.reset()
print("Initial State: ", state)
results = []
for _ in range(1000):
    action = env.action_space.sample()  # Random action
    state, reward, done, _ = env.step(action)
    env.render()
    results.append((state, reward, done))
    if done:
        break

env.close()
print(len(results))
