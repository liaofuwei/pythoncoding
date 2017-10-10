# -*- coding: utf-8 -*-
from dataapi import Client
if __name__ == "__main__":
    try:
        client = Client()
        client.init('ec401353e11cd728472e9027e6d900387511e6c3cf5b54d8f277b18cc45f6bcd')
        url1='/api/fund/getFundNav.json?field=&beginDate=20151228&endDate=20151231&secID=&ticker=150175'
        code, result = client.getData(url1)
        if code==200:
            '''
            print result
            secID="150175.XSHE"
            print result.find(secID)
            lens_secID=len(secID)
            print lens_secID
            str=result[50:lens_secID]
            print str
            '''
            #print "data: ",result['data']
            file_object = open('C:\\pythoncoding\\programming trade\\wmcloseplate\\150175.csv', 'w')
            file_object.write(result)
            file_object.close( )
        else:
            print code
            print result
    except Exception,e:
        raise e