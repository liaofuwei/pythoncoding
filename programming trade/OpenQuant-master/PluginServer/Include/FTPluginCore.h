#pragma once

#include "FTPluginQuoteInterface.h"
#include "FTPluginTradeInterface.h" 

/**
* �ò���ӿ�Ҫ���FTCoreƥ��汾��
*/
#define  FTCore_Support_Ver  113

/**
 *	��ţţ�ͻ���ʵ�ֵĲ�������ӿ�
 */
interface IFTPluginCore 
{ 
	/**
	* Core�ṩ����ؽӿ�֧��

	* @param uid  IID_IFTQuoteData ��
	*		

	* @return  �����ģ�����
	*/ 
	virtual void QueryFTInterface(REFIID uid, void** ppInterface) = 0; 
}; 


/**
* Dll �����Ҫʵ�ֵĽӿڶ��� 
*/ 
interface IFTPluginMoudle
{ 
	/**
	* ��ʼ��������ʼ��
	*/ 
	virtual void Init(IFTPluginCore* pPluginCore) = 0;  
	virtual void Uninit() = 0; 

	/**
	* �������guid  
	*/ 
	virtual LPCWSTR	GetName() = 0;
	virtual GUID    GetGuid() = 0; 

	/**
	* Ӧ�ò�����Ƿ���ʾ������ڣ��������д���) 
	*/
	virtual void 	ShowPlugin(bool bShow) = 0; 

	/*
	* �������¼�֪ͨʱ�� ��Plugin�õ��ص��ӿ� 
	*/
	virtual void  GetPluginCallback_Quote(IQuoteInfoCallback** pCallback) = 0; 
	virtual void  GetPluginCallback_QuoteKL(IQuoteKLCallback** pCallback) = 0; 
	virtual void  GetPluginCallback_TradeHK(ITradeCallBack_HK** pCallback) = 0; 
	virtual void  GetPluginCallback_TradeUS(ITradeCallBack_US** pCallback) = 0; 
}; 


/**
* ���dll �����ӿ�"GetFTPluginMoudle"�� �Ա�ftnn�������ܹ����ظ�ģ�� 

* @param nVerSupport�ò����ҪFTCore��Ͱ汾��,һ�㴫�ض���FTCore_Support_MinVer,
   �������,��������ظò��

* @return  �����ģ�����
*/
typedef IFTPluginMoudle*  (__stdcall* LPGetFTPluginMoudle)(int& nVerSupport); 


/************************************************************************/
/* ���˼��                                                                 */
/************************************************************************/

//Ŀ�ļ���״:
//1. �ռ����齻�׽ӿڵ���ʵ����, �Դ���Ϊ�˴˹�ͨ�����Ϳ��� 
//2. ���ϵĽӿ�Ŀǰ�д��ڳ�������׶Σ���δ���뿪���׶Σ���ӭ��������� 


//��ȫ�ԣ�
//
//1. ��������֤
//2. ��Ʊ���ĵĸ�����Ҫ����
//3. ���׽ӿڵ�Ƶ������
//4. ������ظ������� 

//������
//
//1. ʵ������plugin ��Ҫ����Ϥvc ������python���������ԵĿ���֧�ַ���: 
//   ʵ��һ��plugin, ���ڲ�����һ���򵥵�socket ����, ת��������ݻ�ӿڣ���Ҫ����futu��ʵ��) 
//
//2. ֱ����web����ʽ���Žӿ�(ʱ���Ͽ��ܻ����һЩ)
// 









