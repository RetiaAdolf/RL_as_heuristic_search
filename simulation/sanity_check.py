import pickle
target_output = [50.59, -3.60, 65.30, 7.63]
with open('SimOutput.pickle', 'rb') as f:
	data = pickle.load(f)
	
for key in data.keys():
	output = list(data[key])
	if all(x < y for x, y in zip(output, target_output)):
		print(key, output, target_output)
		print("feasible")
		break
else:
	print("infeasible")