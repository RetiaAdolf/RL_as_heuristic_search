import pickle
target_output = [34.96, -1.17, 14.61, 13.44]
with open('SimOutput.pickle', 'rb') as f:
	data = pickle.load(f)
	
for key in data.keys():
	output = list(data[key])
	if all(x <= y for x, y in zip(output, target_output)):
		print(key, output, target_output)
		print("feasible")
		break
else:
	print("infeasible")