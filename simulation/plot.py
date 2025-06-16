import os
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = './'
path_gpu_proposed = DATA_DIR + "GPUColo/data/used_gpus"
dir_gpu_proposed = os.listdir(path_gpu_proposed)
print("len GPUcolo used gpus %d"%(len(dir_gpu_proposed)))

path_gpu_Separate_burst = DATA_DIR + "Separate/data/used_gpus"
dir_gpu_Separate_burst = os.listdir(path_gpu_Separate_burst)
print("len sepa2 used gpus %d"%(len(dir_gpu_Separate_burst)))

path_gpu_Separate_burst_train = DATA_DIR + "Separate/data/used_gpus_train"
dir_gpu_Separate_burst_train = os.listdir(path_gpu_Separate_burst_train)
print("len sepa2 used gpus train %d"%(len(dir_gpu_Separate_burst_train)))

path_gpu_Separate_burst_infer = DATA_DIR + "Separate/data/used_gpus_infer"
dir_gpu_Separate_burst_infer = os.listdir(path_gpu_Separate_burst_infer)
print("len sepa2 used gpus infer %d"%(len(dir_gpu_Separate_burst_infer)))

path_gpu_non = DATA_DIR + "Non/data/used_gpus"
dir_gpu_non = os.listdir(path_gpu_non)
print("len nocolo used gpus %d"%(len(dir_gpu_non)))

path_gpu_gslice = DATA_DIR + "Gslice/data/used_gpus"
dir_gpu_gslice = os.listdir(path_gpu_gslice)
print("len gslice used gpus %d"%(len(dir_gpu_gslice)))

path_gpu_gslice_train = DATA_DIR + "Gslice/data/used_gpus_train"
dir_gpu_gslice_train  = os.listdir(path_gpu_gslice_train)
print("len gslice used gpus train %d"%(len(dir_gpu_gslice_train)))

path_gpu_gslice_infer = DATA_DIR + "Gslice/data/used_gpus_infer"
dir_gpu_gslice_infer  = os.listdir(path_gpu_gslice_infer)
print("len gslice used gpus infer %d"%(len(dir_gpu_gslice_infer)))

path_ddl_proposed = DATA_DIR + "GPUColo/data/miss_ddl"
dir_ddl_proposed = os.listdir(path_ddl_proposed)

# Train time
path_train_proposed = DATA_DIR + "GPUColo/data/train_time"
dir_train_proposed = os.listdir(path_train_proposed)

path_train_Separate_burst = DATA_DIR + "Separate/data/train_time"
dir_train_Separate_burst = os.listdir(path_train_Separate_burst)

path_train_non = DATA_DIR + "Non/data/train_time"
dir_train_non = os.listdir(path_train_non)

path_train_gslice = DATA_DIR + "Gslice/data/train_time"
dir_train_gslice = os.listdir(path_train_gslice)

gpu_proposed100 = []
# gpu_simple100 = []
# gpu_conserve100 = []
# gpu_separate100 = []
# gpu_separate100_train = []
# gpu_separate100_infer = []
gpu_Separate_burst = []
gpu_Separate_burst_train = []
gpu_Separate_burst_infer = []
gpu_non100 = []
gpu_non100_train = []
gpu_non100_infer = []
gpu_gslice100 = []
gpu_gslice100_train = []
gpu_gslice100_infer = []

for ind in range(len(dir_gpu_proposed)-1):
    with open(DATA_DIR + 'GPUColo/data/used_gpus/used_gpus_slo90_day%d'%(ind+1)) as f:
        lines = f.readlines()
    for l in lines:
        gpu_proposed100.append(float(l))

for ind in range(len(dir_gpu_Separate_burst)-1):
    with open(DATA_DIR + 'Separate/data/used_gpus/used_gpus_slo_day%d'%(ind+1)) as f:
        lines = f.readlines()
    for l in lines:
        gpu_Separate_burst.append(float(l))

for ind in range(len(dir_gpu_non)-1):
    with open(DATA_DIR + 'Non/data/used_gpus/used_gpus_slo_day%d'%(ind+1)) as f:
        lines = f.readlines()
    for l in lines:
        gpu_non100.append(float(l))

    with open(DATA_DIR + 'Non/data/used_gpus_train/used_gpus_slo_day%d'%(ind+1)) as f:
        lines = f.readlines()
    for l in lines:
        gpu_non100_train.append(float(l))
    assert len(gpu_non100)==len(gpu_non100_train)

    with open(DATA_DIR + 'Non/data/used_gpus_infer/used_gpus_slo_day%d'%(ind+1)) as f:
        lines = f.readlines()
    for l in lines:
        gpu_non100_infer.append(float(l))
    assert len(gpu_non100)==len(gpu_non100_infer)

for ind in range(len(dir_gpu_gslice)-1):
    with open(DATA_DIR + 'Gslice/data/used_gpus/used_gpus_slo_day%d'%(ind+1)) as f:
        lines = f.readlines()
    for l in lines:
        gpu_gslice100.append(float(l))

    with open(DATA_DIR + 'Gslice/data/used_gpus_infer/used_gpus_slo_day%d'%(ind+1)) as f:
        lines = f.readlines()
    for l in lines:
        gpu_gslice100_infer.append(float(l))
    # print(ind, len(gpu_gslice100),len(gpu_gslice100_infer))
    assert len(gpu_gslice100_infer)==len(gpu_gslice100)

SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 12


plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

gpu_separate_raw = np.array(gpu_Separate_burst_train)+np.array(gpu_Separate_burst_infer)
gpu_non_raw = np.array(gpu_non100_train)+np.array(gpu_non100_infer)
gpu_separate100=[] #train:infer=5:1
for i in range(len(gpu_Separate_burst_train)):
    from_train = gpu_Separate_burst_train[i]//5+gpu_Separate_burst_train[i] # train:infer=5:1
    from_infer = gpu_Separate_burst_infer[i]*6
    gpu1_5 = max(from_train,from_infer)
    gpu_separate100.append(gpu1_5)

    from_train = gpu_Separate_burst_train[i]*6 # train:infer=1:5
    from_infer = gpu_Separate_burst_infer[i]//5+gpu_Separate_burst_infer[i]
    gpu5_1 = max(from_train,from_infer)
    gpu_Separate_burst[i] = gpu5_1

def avg_hour(ls, interval):
    ls_avg = []
    for i in np.arange(0,len(ls),interval):
        hour=[]
        for j in np.arange(i,i+interval):
            if j >= len(ls):
                break
            hour.append(ls[j])
        ls_avg.append(np.mean(hour))
    return ls_avg

gpu_non100_avg = avg_hour(gpu_non100, 86400)
gpu_non_raw_avg = avg_hour(gpu_non_raw, 86400)
gpu_proposed100_avg = avg_hour(gpu_proposed100, 86400)
gpu_separate100_avg = avg_hour(gpu_separate100, 86400) #1:5
gpu_Separate_burst_avg = avg_hour(gpu_Separate_burst, 86400) #5:1
gpu_Separate_raw_avg = avg_hour(gpu_separate_raw, 86400)
gpu_gslice100_avg = avg_hour(gpu_gslice100, 86400)

fig, ax = plt.subplots()
fig.set_size_inches(4.5, 3)
line1,=ax.plot(np.arange(0,len(gpu_non_raw_avg)),   gpu_non_raw_avg, color="tab:blue", linestyle='dashed',label='No Co-location')
line2,=ax.plot(np.arange(0,len(gpu_proposed100_avg[:-3])), gpu_proposed100_avg[:-3], color="tab:orange",linewidth=2.0,label='GPUColo')
# line3,=ax.plot(np.arange(0,len(gpu_separate100_avg)), gpu_separate100_avg, color="tab:green",linestyle='dotted', label='Separate 1:5')
# line3,=ax.plot(np.arange(0,len(gpu_Separate_raw_avg)), gpu_Separate_raw_avg, color="tab:green",linestyle='dotted', label='Separate 1:1')
line4,=ax.plot(np.arange(0,len(gpu_Separate_burst_avg)), gpu_Separate_burst_avg, color="tab:red",linestyle='dashdot', label='Separate 5:1')
line5,=ax.plot(np.arange(0,len(gpu_gslice100_avg)), gpu_gslice100_avg, color="fuchsia", linestyle=(0, (3, 1, 1, 1, 1, 1)), label='GSLICE')
ax.vlines(x = 20, ymin = 0, ymax = 6000, colors = 'black', linestyle='dashed', label = 'Day 20')
ax.vlines(x = 40, ymin = 0, ymax = 6000, colors = 'black', linestyle='dashed', label = 'Day 40')
ax.hlines(y = 1500, xmin = 0, xmax = 60, colors = 'gray', linestyle='dashed', label = '1500')
ax.hlines(y = 2500, xmin = 0, xmax = 60, colors = 'gray', linestyle='dashed', label = '2000')

lns = [line1,line2,line4,line5]
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='upper center',fancybox=False,
               shadow=False, ncol=2, frameon=False,
           bbox_to_anchor=(0.46, 1.05))
ax.set_xlabel('Time (days)')
ax.set_ylabel('Number of GPUs')
ax.set_xlim(0,58)
ax.set_ylim(0,8000)
plt.yticks(rotation=90)
fig.tight_layout()
plt.savefig('plot.png',format="png",dpi=300)

print("Proposed, avg GPU %d, time %.2f days, aggre num %d"%(np.mean(gpu_proposed100), len(gpu_proposed100)/3600/24, np.mean(gpu_proposed100)*len(gpu_proposed100)/3600))
# print("Separate T:I=5:1, avg GPU %d, time %.2f days, aggre num %d"%(np.mean(gpu_separate100), len(gpu_separate100)*10/3600/24, np.mean(gpu_separate100)*len(gpu_separate100)*10/3600/24))
print("Separate T:I=1:5, avg GPU %d, time %.2f days, aggre num %d"%(np.mean(gpu_Separate_burst), len(gpu_Separate_burst)/3600/24, np.mean(gpu_Separate_burst)*len(gpu_Separate_burst)/3600))
print("No colo, avg GPU %d, time %.2f days, aggre num %d"%(np.mean(gpu_non_raw), len(gpu_non_raw)/3600/24, np.mean(gpu_non_raw)*len(gpu_non_raw)/3600))
print("GSLICE, avg GPU %d, time %.2f days, aggre num %d"%(np.mean(gpu_gslice100), len(gpu_gslice100)/3600/24, np.mean(gpu_gslice100)*len(gpu_gslice100)/3600))