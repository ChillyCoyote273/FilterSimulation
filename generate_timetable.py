import numpy as np
import pandas as pd


def main():
	random_table = np.random.rand(3, 10)
	times = np.array([np.linspace(0, 10, 10)])
	data = np.append(times, random_table, axis=0).T
	dataframe = pd.DataFrame(data, columns=['t', 'y1', 'y2', 'u1'])
	# dataframe['t'] = dataframe['t'].astype(str) + " sec"
	dataframe.to_csv("timetable.csv")


if __name__ == "__main__":
	main()
