import numpy as np
import subprocess
import signal
import os
def __normalization__(data, data_range):
	data_min = data_range[:, 0]
	data_max = data_range[:, 1]
	normalized_data = (data - data_min) / (data_max - data_min)
	return normalized_data

def __random_target__(target_range, output_type):
	rand_output = []
	for output_range in target_range:
		if output_type == "float":
			rand_output.append(round(random.uniform(output_range[0],output_range[1]), 2))
		elif output_type == "int":
			rand_output.append(random.randint(output_range[0],output_range[1]))
	return np.array(rand_output)


class Simulator(object):
	"""docstring for Simlator"""
	def __init__(self):
		super(Simlator, self).__init__()


	def __run_command__(self, command, makefile_dir):
		with open('/dev/null', 'w') as f:
			process = subprocess.Popen(command, cwd=makefile_dir, stdout=f, stderr=f)
		return process

	def __read_file__(self, file_path):
		with open(file_path, 'r') as f:
			#print(file_path)
			data = f.readline().split()
			while not data:
				data = f.readline().split()
			data = f.readline().split()
		f.close()
		return data

	def __read_data__(self, data):
		PowerDC = float(data[4][:-1])
		PowerDC_unit = data[4][-1]
		if PowerDC_unit == "u":
			PowerDC = PowerDC
		elif PowerDC_unit == "n":
			PowerDC = PowerDC * 1e-3
		elif PowerDC_unit == "m":
			PowerDC = PowerDC * 1e3

		GBW = float(data[5][:-1])
		GBW_unit = data[5][-1]
		if GBW_unit == "M":
			GBW = GBW
		elif GBW_unit == "K":
			GBW = GBW * 1e-3
		elif GBW_unit == "G":
			GBW = GBW * 1e3
		GBW = -GBW

		RmsNoise = float(data[6][:-1])
		RmsNoise_unit = data[6][-1]
		if RmsNoise_unit == "n":
			RmsNoise = RmsNoise
		elif RmsNoise_unit == "p":
			RmsNoise = RmsNoise * 1e-3
		elif RmsNoise_unit == "u":
			RmsNoise = RmsNoise * 1e3


		SettlingTime = float(data[7][:-1])
		SettlingTime_unit = data[7][-1]
		if SettlingTime_unit == "u":
			SettlingTime = SettlingTime
		elif SettlingTime_unit == "n":
			SettlingTime = SettlingTime * 1e-3
		elif SettlingTime_unit == "m":
			SettlingTime = SettlingTime * 1e3

		return PowerDC, GBW, RmsNoise, SettlingTime

	def step(self, action, env_id):
		M3_W, M7_W, IN_OFST = action
		if M3_W < 12:
			M3_W = 12
		if M3_W > 60:
			M3_W = 60

		if M7_W < 12:
			M7_W = 12
		if M7_W > 60:
			M7_W = 60

		if IN_OFST < 0:
			IN_OFST = 0
		if IN_OFST > 50:
			IN_OFST = 50

		IN_OFST = IN_OFST / 100.0
		M3_W = str(M3_W)
		M7_W = str(M7_W)
		IN_OFST = str(IN_OFST)
		file_path = '../data/M3W_{}_M7W_{}_INOFST_{}.txt'.format(M3_W, M7_W, IN_OFST)
		while not os.path.exists(file_path):
			#command = "make -C /mnt/mydata/RL_{}/run/ M3_W={} M7_W={} IN_OFST={}".format(self.env_id, M3_W, M7_W, IN_OFST)
			command = ["make", "M3_W={}".format(M3_W), "M7_W={}".format(M7_W), "IN_OFST={}".format(IN_OFST)]
			makefile_dir = "/mnt/mydata/RL_{}/run/".format(env_id)
			process = self.__run_command__(command,makefile_dir)
			try:
				process.wait(timeout=100)
			except:
				pgid = os.getpgid(process.pid)
				os.killpg(pgid, signal.SIGTERM)

		data = self.__read_file__(file_path)
		PowerDC, GBW, RmsNoise, SettlingTime = self.__read_data__(data)

		return np.array([PowerDC, GBW, RmsNoise, SettlingTime])

		