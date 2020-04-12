#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame


start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2020, 6, 4)

df = web.DataReader("AAPL", 'yahoo', start, end)
df.tail()


# In[3]:


close_px = df['Adj Close']
mavg = close_px.rolling(window=100).mean()


# In[4]:


mavg


# In[5]:


mavg.tail()


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib import style

# Adjusting the size of matplotlib
import matplotlib as mpl
mpl.rc('figure', figsize=(8, 7))
mpl.__version__

# Adjusting the style of matplotlib
style.use('ggplot')

close_px.plot(label='AAPL')
mavg.plot(label='mavg')
plt.legend()


# In[7]:


rets = close_px / close_px.shift(1) - 1
rets.plot(label='return')


# In[8]:


dfcomp = web.DataReader(['AAPL', 'GE', 'GOOG', 'IBM', 'MSFT'],'yahoo',start=start,end=end)['Adj Close']


# In[9]:


dfcomp


# In[10]:


retscomp = dfcomp.pct_change()

corr = retscomp.corr()


# In[11]:


corr


# In[12]:


plt.scatter(retscomp.AAPL, retscomp.GE)
plt.xlabel(‘Returns AAPL’)
plt.ylabel(‘Returns GE’)


# In[17]:


plt.scatter(retscomp.AAPL, retscomp.GE)


# In[14]:


plt.xlabel(‘Returns AAPL’)


# In[15]:


plt.xlabel("Returns AAPL")


# In[16]:


plt.ylabel("Returns GE")


# In[18]:


plt.scatter(retscomp.AAPL, retscomp.GE)
plt.xlabel("Returns AAPL")
plt.ylabel("Returns GE")


# In[20]:


pd.scatter_matrix(retscomp, diagonal="kde", figsize=(10, 10))


# In[21]:


from pandas.plotting import scatter_matrix


# In[ ]:





# In[23]:


scatter_matrix(retscomp, diagonal="kde", figsize=(10, 10))


# In[24]:


pd.scatter_matrix(retscomp, diagonal='kde', figsize=(10, 10));


# In[25]:


pd.plotting.scatter_matrix(retscomp, diagonal='kde', figsize=(10, 10))


# In[26]:


plt.imshow(corr, cmap='hot', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns);


# In[27]:


plt.scatter(retscomp.mean(), retscomp.std())
plt.xlabel('Expected returns')
plt.ylabel('Risk')
for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (20, -20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))


# In[ ]:




