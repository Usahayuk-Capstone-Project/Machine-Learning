#!/usr/bin/env python
# coding: utf-8

# In[1]:
from tokenize import String 
from turtle import st 
from pydantic import BaseModel 
from typing import Optional, List

# In[2]:
class Datas(BaseModel): 
    user_request: List 
    ids: List 
        
# In[3]:
class dataBRecommender(BaseModel): 
    #user_id: str
    pass

# In[ ]:




