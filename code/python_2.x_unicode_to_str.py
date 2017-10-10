#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Function:
【整理】Python中字符编码的总结和对比：Python 2.x的str和unicode vs Python 3.x的bytes和str
http://www.crifan.com/summary_python_string_encoding_decoding_difference_and_comparation_python_2_x_str_unicode_vs_python_3_x_bytes_str

Author:     Crifan
Verison:    2012-11-29
-------------------------------------------------------------------------------
"""

def python2xUnicodeToStr():
    zhcnUnicode = u"1.此处是中文字符；2.而你之所以能正确看到此处中文字符，是因为(1)此处python文件中，通过开始的编码指定为UTF-8(2)并且本身Python文件也是UTF-8编码保存的；3.接下来将要演示的是，将此段中文字符，转换为GBK编码，然后在windows的cmd中输出;";
    print "type(zhcnUnicode)=",type(zhcnUnicode); #type(zhcnUnicode)= <type 'unicode'>
    zhcnGBK = zhcnUnicode.encode("GBK");
    print "You should see these zh-CN chars in windows cmd normally: zhcnGBK=%s"%(zhcnGBK); #You should see these zh-CN chars in windows cmd normally: zhcnGBK=1.此处是中文字符；...... 然后在windows的cmd中输出;
    
###############################################################################
if __name__=="__main__":
    python2xUnicodeToStr();