#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:


with open("quotes.js","r") as f:
    quotes = f.readlines()


# In[9]:


# parse the quotes from js file into list
quotes_list = []
curr_content = ""
curr_source = ""
reading_content = False
for line in quotes[9:]:
    l = line.lstrip(" ").rstrip("\n")
    if reading_content == False:
        if "<pquote" in l:
            reading_content = True
    else:
        if "</pquote>" in l:
            quotes_list.append((curr_source,curr_content))
            curr_source = ""
            curr_content = ""
            reading_content = False
        else:
            split = l.split("<b>")
            if (len(split)==1) and ("</b>" in split[0]):
                curr_source = split[0].strip("</b>").strip("<br/>")
            elif (len(split)==1) and ("<br/><br/>" in split[0]):
                if "<br/><br/>" == split[0][-10:]:
                    curr_content += split[0][:-10]
                    curr_content += " "
            elif len(split)==1:
                curr_content += split[0]
                curr_content += " "
            elif len(split)==2:
                curr_content += (split[0].strip("<br/><br/>"))
                curr_content += " "
                curr_source += split[1].strip("</b>").strip("<br/>")
            else:
                print("\n\nCurrent line:")
                print(split)
                raise Exception("Something went wrong\n"*20)
        
# In[96]:


# write the list of quotes into file with good format
f = open("quotes.json","w")
f.write("[\n")
for i,j in quotes_list:
    f.write("{\n")
    f.write('\t"source":"{}",\n'.format(i.replace('"',"''"))) # double quotes must be replaced with two 
    f.write('\t"content":"{}"\n'.format(j.replace('"',"''"))) # single quotes for formatting reasons
    f.write("},\n") # need to manually delete the last comma for proper formatting },
f.write("]")
f.close()


# In[97]:


# examine the json file, also this is to make sure that it loads properly
import json
f = open("quotes.json",)
data = json.load(f)
f.close()
print(data)

