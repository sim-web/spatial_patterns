import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.stats import norm


def fields(params, centers, sigmas):
	"""
	Plotting of Gaussian Fields

	Arguments:
	- centers: numpy array of centers for the gaussians
	- sigmas: numpy array of standard deviations for the gaussians
	"""
	x = np.linspace(0, params['boxlength'], 200)
	it = np.nditer([centers, sigmas])
	summe = 0
	for c, s in it:
		gaussian = norm(loc=c, scale=s).pdf
		plt.plot(x, gaussian(x))
		summe += gaussian(x)
	plt.plot(x, 2*summe/(len(centers)))
	#plt.show()
	return

def sum_of_symmetric_gaussians():
	np.exp(np.power((x - m / N), 2))

def positions_animation(params, positions, save=False):
	"""
	Creates an animation of the rat moving through space.

	- positions: Array: [[x1, y2], [x2, y2], ...]
	- save: Save a video .mp4
	"""
	boxlength = params['boxlength']
	dt = params['dt']
	fig = plt.figure()
	plt.xlim(0, 1)
	ax = fig.add_subplot(111)
	ax.set_xlim(0, boxlength)
	ax.set_ylim(0, boxlength)
	ax.set_xlabel('Position')
	ax.set_title('Moving Rat')

	l, = ax.plot([], [], 'o-')
	txt = ax.text(
			0.02, 0.95, '',  # Specified in axis coords, (see transform below)
			horizontalalignment='left',
			verticalalignment='top',
			bbox=dict(facecolor='gray', alpha=0.2),  # Draw box around text
			transform=ax.transAxes)  # Text positions in axis coords, not in data coords

	def _update_lines_and_text(num, positions, line, text):
		line.set_data(positions[num])
		text.set_text('Time = ' + str(num*dt))
		return line, text

	def _init():
		l.set_data([], [])
		txt.set_text('')
		return l, txt

	ani = animation.FuncAnimation(
			fig, _update_lines_and_text, np.arange(0, len(positions)),
			fargs=(positions, l, txt),
			interval=50, blit=True, repeat_delay=2000, repeat=True, init_func=_init)

	if save is True:
		ani.save(
			'/Users/simonweber/Desktop/im.mp4',
			writer=animation.FFMpegFileWriter(),
			metadata={'artist': 'Simon Weber'})
	else:
		plt.show()


def positions_movie_old(params, positions, save=False):
	"""
	BETA
	Creates a movie of how the rat moves through space.

	- positions: Array: [[x1, y2], [x2, y2], ...]
	"""
	boxlength = params['boxlength']
	fig = plt.figure(0)
	ax = fig.add_subplot(111)
	ax.set_xlim(0, boxlength)
	ax.set_ylim(0, boxlength)
	ims = []
	#ax.text(0.8, 1.02, 'Init Text', fontsize=14)

	#txt=plt.text(0.8, 1.02, 'TEST', fontsize=14,  bbox=dict(alpha=0.1))
	for p in positions:
		#ax.draw_artist(ax.text(0.8, p[0], 'bla', fontsize=14))
		line = ax.plot([p[0]], [p[1]], 'o', color='b', markersize=20)
		#print ax.get_lines()
		# line = ax.get_lines(1)
		#print dir(ax)
		#print line
		ims.append(line)
		#ims.append(text)
		#ims.append(ax.text(0.8, 1.02, str(p), fontsize=14))
	im_ani = animation.ArtistAnimation(
		fig, ims, interval=50, repeat_delay=1000, blit=True)

	if save is True:
		im_ani.save(
			'/Users/simonweber/Desktop/im.mp4',
			writer=animation.FFMpegFileWriter(),
			metadata={'artist': 'Simon Weber'})
	else:
		plt.show()

# def centers(params, centers_array):
# 	x = np.linspace(0, params['boxlength'], 200)
# 	s = 0
# 	for c in centers_array:
# 		y = norm(loc=c, scale=params['sigma_exc']).pdf
# 		s += y(x)
# 		plt.plot(x, y(x))
# 	plt.plot(x, 2*s/(len(centers_array)))
# 	return
