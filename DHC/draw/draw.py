import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

x1 = np.arange(1, 6, 1)
x2 = np.arange(9, 16, 1)
ticks1 = ['1', '2', '3', '4', '5']
ticks2 = ['6', '8', '10', '12', '14']
nlayers1 = np.array([15.01, 15.08, 15.06, 15.07, 15.04])  # transformer_dmodel:64
nlayers2 = np.array([14.87, 14.82, 14.87, 14.90, 14.96])  # transformer_dmodel:128
nlayers3 = np.array([15.04, 14.91, 14.87, 14.94, 15.26])  # transformer_dmodel:256
mae_nlayers1 = np.array([8.88, 8.92, 8.95, 8.90, 8.91])
mae_nlayers2 = np.array([8.88, 8.86, 8.82, 8.81, 8.84])
mae_nlayers3 = np.array([8.94, 8.86, 8.79, 8.85, 9.05])

repetation1 = np.array([14.92, 14.91, 14.94, 14.82, 14.86, 14.91, 14.91])  # stgsp
repetation2 = np.array([14.86, 14.82, 14.80, 14.72, 14.81, 14.82, 14.84])  # stgsp+ssl
mae_repetation1 = np.array([8.82, 8.87, 8.83, 8.86, 8.80])
mae_repetation2 = np.array([8.79, 8.76, 8.80, 8.79, 8.78])
f, (ax1, ax3) = plt.subplots(1, 2, figsize=(9,4), sharex=False)

# sns.lineplot(x=x1, y=nlayers1, palette="coolwarm", ax=ax1, ci=None, markers="o")
ax1.plot(x1,nlayers1,marker="o",label=r"$ d_{model}:64 $",linestyle = ':')
ax1.plot(x1,nlayers2,marker="v",label=r"$ d_{model}:128 $",linestyle = '-.')
ax1.plot(x1,nlayers3,marker="x",label=r"$ d_{model}:256 $",linestyle = '--')
ax1.legend()
ax1.grid(True,linestyle='dotted')
# ax1.set_xlabel("Transformer Encoder Layer")
ax1.set_ylabel("RMSE")
ax1.set_title("(a) Transformer Encoder Layer",y=-0.2)

ax3.plot(x2,repetation1,marker="o",label="ST-GSP",linestyle = '-.')
ax3.plot(x2,repetation2,marker="v",label="ST-GSP+SSL",linestyle = '--')
ax3.legend(loc="upper right")
ax3.grid(True,linestyle='dotted')
# ax3.set_xlabel("Residual Unit Number")
ax3.set_ylabel("RMSE")
ax3.set_ylim([14.70, 14.97])
ax3.set_title("(b) Residual Unit Number",y=-0.2)

# sns.lineplot(x=x2, y=repetation1, palette="coolwarm", ax=ax3, ci=None, markers="o")
# ax3.xaxis.set_major_formatter(mtick.PercentFormatter())
# ax3.set_xticklabels(ticks2)
plt.subplots_adjust(wspace=0.25, bottom=0.14)
plt.show()
