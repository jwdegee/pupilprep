import matplotlib.pyplot as plt
import seaborn as sns
from pupilprep import preprocess_pupil

# files and settings:
filename = 'data/sub-01_block-01.asc'
params = {'buffer':0.2, 'lp':10, 'hp':0.01, 'order':3}

# preprocess:
df, events, fs = preprocess_pupil.preprocess_pupil(filename, params)

# plot:
fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(211)
plt.plot(df['time'], df['pupil'], label='pupil')
plt.plot(df['time'], df['pupil_int'], label='pupil_int')
blinks = events.loc[(events['description']=='blink'), 'onset'].values
for b in blinks:
      plt.axvspan(b-0.05, b+0.1, color='r', alpha=1)
plt.xlabel('Time (s)')
plt.legend(loc=1)
ax = fig.add_subplot(212)
plt.plot(df['time'], df['pupil_int_lp_psc'], label='pupil_int_lp_psc')
plt.plot(df['time'], df['pupil_int_lp_clean_psc'], label='pupil_int_lp_clean_psc')
plt.xlabel('Time (s)')
plt.legend(loc=1)
plt.tight_layout()
sns.despine()

plt.show()