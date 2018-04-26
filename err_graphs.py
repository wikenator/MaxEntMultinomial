#!/usr/bin/python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def vecs(csvf):
	x = []; y = []
	f = open(csvf)

	for line in f:
		ll = line.strip().split(',')
		x.append(int(ll[0]))
		y.append(float(ll[1]))

	f.close()

	return x, y

if __name__ == '__main__':
	x1, y1 = vecs('terr.csv')
	x2, y2 = vecs('verr.csv')
	x3, y3 = vecs('acc.csv')
	plt.plot(x1, y1, 'b-', x2, y2, 'r-', x3, y3, 'g-')
	plt.axis([0, len(x1), 0.6, 1.4])
	plt.legend(['Training Error', 'Validation Error', 'Training Accuracy'])
	plt.savefig('err.png')
