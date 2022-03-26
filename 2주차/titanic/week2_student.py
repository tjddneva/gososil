#!/usr/bin/env python
# coding: utf-8

# # 연습문제
# 
# 
# ## numpy
# 
# python에서 array등을 다루는 수학/과학 컴퓨팅을 하기 위한 패키지
# 
# ## matplotlib
# 
# python에서 matlab과 유사한 그래프 표시를 가능하게 하는 패키지연습문제

# ## numpy
# (본 cell은 markdown 형식으로, 더블클릭/Enter로 들어가고 Shift+Enter로 나갈 수 있다. 주석 용으로 사용.)  
# 
# python에서 과학 컴퓨팅을 하기 위해 사용하는 패키지로 python 라이브러리로 Import 하여 사용.

# In[19]:


# 패키지 사용을 위한 약자 지정

import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


# a, b array 선언 및 초기화

a = np.array((1, 2))
b = np.array((2, 3))

print('a: ', a)
print('b: ', b)


# In[21]:


a + b 


# In[22]:


a - b


# In[23]:


a * b


# In[24]:


np.dot(a, b)


# In[25]:


a.dot(b)


# ### 함수

# In[26]:


def test_function(sentence, number=2):
    sentence += sentence + str(number)
    return sentence


# In[27]:


print(test_function('함수동작 원리 확인: '))


# In[28]:


print(test_function('함수동작 원리 확인: ', 10))


# In[29]:


print(test_function('함수동작 원리 확인: ', number=10))


# ### For 문

# In[30]:


for i in range(10):
    print(i)


# In[31]:


print(range(10))
print(list(range(10)))


# In[32]:


for i in [0, 1, 2, 3, 4, 5]:
    print(i)


# In[33]:


for i in ['This', 'is', 'example', 1, 2, ['test', 'list']]:
    print(i)


# ## matplotlib
# 
# matplotlib 은 python 에서 matlab 과 유사한 그래프 표시를 가능하게 하는 라이브러리이다.
# 
# 
# ### np.linspace
# 
# 균등한 간격의 데이터를 뽑아줌
# 
# https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
# 
# 
# 
# ### plt.scatter
# 
# 데이터를 visualizing 해줌
# 
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html

# In[34]:


x = np.linspace(0, 10, 30)
y = np.sin(x)
plt.scatter(x, y, color='red')


# # 실습

# In[35]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[36]:


data = pd.read_csv('./titanic/train.csv')


# In[37]:


print(len(data))


# In[38]:


data.head()


# In[39]:


data.tail()


# In[40]:


data.isnull()


# In[41]:


data.isnull().sum()


# In[42]:


f,ax=plt.subplots(1,2,figsize=(18,8))
data['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data=data,ax=ax[1])
ax[1].set_title('Survived')
plt.show()


# In[43]:


data.groupby(['Sex','Survived'])['Survived'].count()


# In[44]:


f,ax=plt.subplots(1,2,figsize=(18,8))
data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex',hue='Survived',data=data,ax=ax[1])
ax[1].set_title('Sex:Survived vs Dead')
plt.show()


# In[45]:


df = data.corr('pearson')
df


# In[46]:


sns.clustermap(df, 
               annot = True,      # 실제 값 화면에 나타내기
               cmap = 'RdYlBu_r',  # Red, Yellow, Blue 색상으로 표시
               vmin = -1, vmax = 1, #컬러차트 -1 ~ 1 범위로 표시
              )


# # 실습1: Pearson Correlation 함수를 만드시오.
# 
# 이번 실습에서는 dataframe 의 내장 함수 ```corr('pearson')``` 을 함수로 구현합니다.

# ### TODO 1

# In[56]:


import math as m
def pearsonCorrelation(data, source_column, target_column):
    result = 0.
    # TODO
    
    xbar = data[source_column].mean()
    ybar = data[target_column].mean()
    
    cov = sum([(x-xbar)*(y-ybar) for x,y in zip(data[source_column],data[target_column])])
    cox = m.sqrt(sum([pow(x-xbar,2) for x in data[source_column]]))
    coy = m.sqrt(sum([pow(y-ybar,2) for y in data[target_column]]))
    corr = cov / (cox * coy)
    corr = round(corr,6)  
    result = round(corr,6)

    return result


# ## 작성한 함수가 정상적으로 작동하는지 확인한다.

# In[57]:


# DO NOT CHANGE
results = []
column_names = []


# In[58]:


# DO NOT CHANGE
for source_column in data.columns:
    source_results = []
    for target_column in data.columns:
        source_results.append(pearsonCorrelation(data, source_column, target_column))
    results.append(source_results)
    column_names.append(source_column)


# # 실습2: Improved Correlation Heatmap
# 
# 
# 위의 결과에서 pearson correlation 을 활용할 수 없는 column 이 있음을 확인하였다.
# ```Name, Ticket, Cabin column``` 을 삭제하고 실험을 다시 진행한다.

# In[59]:


# DO NOT CHANGE
dropped_data = data.drop(columns=['Name', 'Ticket', 'Cabin'])
dropped_data


# ## 설명
# ```Pearson Correlation``` 을 이용하기 위해선 ```Nan```, ```String``` 데이터가 있으면 안된다. 적절한 조치를 취하여 해당 문제를 해결한다.

# In[60]:


# DO NOT CHANGE
dropped_data.isnull().sum()


# ### TODO 2
# 
# 표시되어야할 Column 은 다음과 같습니다.
# 
# ```['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']```

# In[61]:


# TODO
# 아래에 데이터 수정을 통해 pearson correlation 함수가 정상적으로 작동하게 하라.
mean=dropped_data['Age'].mean()
num=0
for i in dropped_data['Age'] :
    if np.isnan(i) :
        dropped_data['Age'][num]=mean
    num+=1
    
dropped_data['Embarked'].fillna('S',inplace=True) #pandas
dropped_data['Embarked'] = dropped_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

dropped_data['Sex'] = dropped_data['Sex'].map({'female': 0, 'male': 1})
dropped_data

# Do NOT CHANGE
dropped_data


# In[62]:


# DO NOT CHANGE
results = []
column_names = []

for source_column in dropped_data.columns:
    source_results = []
    for target_column in dropped_data.columns:
        source_results.append(pearsonCorrelation(dropped_data, source_column, target_column))
    results.append(source_results)
    column_names.append(source_column)


# ## ```results``` 를 ```pd.DataFrame``` 으로 바꾸고 Heatmap 으로 바꾸시오.

# ### TODO 3
# 
# Pearson Correlation Coefficient 를 Visualizing 하시오.

# In[79]:


# TODO
improved_data = pd.DataFrame(results)
for cols, name in zip(improved_data,column_names):
    improved_data.rename(columns={cols : name},inplace=True)
    improved_data.rename(index={cols : name},inplace=True)
    
improved_data

# Show dataframe table


# In[80]:


# Show Heatmap using sns library
plt.rcParams["figure.figsize"] = (10,10)
sns.heatmap(improved_data,
           annot = True, #실제 값 화면에 나타내기
           cmap = 'Greens', #색상
           vmin = -1, vmax=1 , #컬러차트 영역 -1 ~ +1
          )


# In[85]:


colormap = plt.cm.viridis
plt.figure(figsize=(12,10))
plt.title("Pearson Correlation of Features", y=1.05, size=15)
sns.heatmap(improved_data,
            linewidths=0.1, 
            vmax=1.0, 
            square=True, 
            cmap=colormap,
            linecolor='white',
            annot=True,
            annot_kws={'size':16}
           )


# # 실습4
# 
# 지난 실습에서는 데이터의 column 간의 correlation 을 확인하여 survived 에 영향을 크게 주는 column 을 확인하였다. 
# 
# 이 정보를 바탕으로 데이터를 분석하여 어떠한 상황에서 생존률을 예측할 수 있는지를 분석내용과 함께 4가지를 제시하라.
# 
# ### 주의사항
# 1. 머신러닝은 사용하지 않는다.
# 2. 그림, 그래프, 도표로 표시한다.
# 3. 해당 과제 내용을 바탕으로 Week 3 가 진행된다.
# 4. 타당한 수치로 생존율과 연관이 있어야 한다.

# ### 예시
# 아래의 예시들은 나이와 생존의 관계이다.

# 16세 이하의 생존율은 55% 로 다른 나이에 비해 생존율이 높다.

# In[86]:


# 16세 이하의 생존률
dropped_data[(data['Age'] <= 16) & (data['Survived'] == 1)]['Survived'].count() / dropped_data[data['Age'] <= 16]['Survived'].count()
data1 = dropped_data[(data['Age'] <= 16)]

f,ax=plt.subplots(1, 1,figsize=(9,8))
sns.countplot('Survived',data=data1,ax=ax)
ax.set_title('Age <=16 Survived')
plt.show()
print(f"16세 이하의 생존율 {dropped_data[(data['Age'] <= 16) & (data['Survived'] == 1)]['Survived'].count() / dropped_data[data['Age'] <= 16]['Survived'].count() * 100}%")


# In[87]:


# 16 ~ 32세의 생존률
dropped_data[(data['Age'] > 16) & (data['Age'] <= 32) & (data['Survived'] == 1)]['Survived'].count() / dropped_data[(data['Age'] > 16) & (data['Age'] <= 32)]['Survived'].count()


# In[88]:


# 32 ~ 48세의 생존률
dropped_data[(data['Age'] > 32) & (data['Age'] <= 48) & (data['Survived'] == 1)]['Survived'].count() / dropped_data[(data['Age'] > 16) & (data['Age'] <= 32)]['Survived'].count()


# In[89]:


# 48 ~ 64세의 생존률
dropped_data[(data['Age'] > 48) & (data['Age'] <= 64) & (data['Survived'] == 1)]['Survived'].count() / dropped_data[(data['Age'] > 48) & (data['Age'] <= 64)]['Survived'].count()


# In[90]:


# 64세 이상의 생존률
dropped_data[(data['Age'] > 64) & (data['Survived'] == 1)]['Survived'].count() / dropped_data[(data['Age'] > 64)]['Survived'].count() 




# In[ ]:


plt.rcParams["figure.figsize"] = (20,10)
plt.rcParams['axes.grid'] = True 

def bar_chart(feature):
    survived = dropped_data[dropped_data['Survived']==1][feature].value_counts()
    dead = dropped_data[dropped_data['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True)


# # 분석 1

# In[104]:


# TODO
bar_chart('Pclass')


# In[106]:


dropped_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).sum()


# In[109]:


pd.crosstab(dropped_data['Pclass'], dropped_data['Survived'], margins=True).style.background_gradient(cmap='summer_r')


# In[110]:


dropped_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar()


# In[111]:


y_position = 1.02
f, ax = plt.subplots(1, 2, figsize=(18, 8))
dropped_data['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'], ax=ax[0])
ax[0].set_title('Number of Passengers By Pclass', y=y_position)
ax[0].set_ylabel('Count')
sns.countplot('Pclass', hue='Survived', data=dropped_data, ax=ax[1])
ax[1].set_title('Pclass: Survived vs Dead', y=y_position)
plt.show()


# # 분석 2

# In[105]:


# TODO
bar_chart('Sex')


# In[112]:


f, ax = plt.subplots(1, 2, figsize=(18, 8))
dropped_data[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex', hue='Survived', data=dropped_data, ax=ax[1])
ax[1].set_title('Sex: Survived vs Dead')
plt.show()


# In[113]:


dropped_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[114]:


pd.crosstab(dropped_data['Sex'], dropped_data['Survived'], margins=True).style.background_gradient(cmap='summer_r')


# # 분석 3

# In[ ]:


# TODO


# In[ ]:




