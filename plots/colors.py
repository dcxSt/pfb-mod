import matplotlib.pyplot as plt
colors=plt.get_cmap('Set2').colors # get list of RGB color values
for i in range(4):
    print(f"#{i} ",colors[i])

