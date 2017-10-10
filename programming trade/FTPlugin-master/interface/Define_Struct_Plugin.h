#pragma once

#include "IQuote.h"
#include "ITrade.h" 

static const GUID IID_IFTQuoteData = 
{ 0xb75073e3, 0xaa3a, 0x4717, { 0xac, 0xa2, 0x11, 0x94, 0xa1, 0x3, 0x78, 0xc7 } };

static const GUID IID_IFTQuoteOperation = 
{ 0x9c65990c, 0x903, 0x4185, { 0x97, 0x12, 0x3e, 0xa7, 0xab, 0x34, 0xd, 0xc5 } };

static const GUID IID_IFTTrade = 
{ 0x69a88049, 0x252e, 0x4a12, { 0x83, 0x41, 0xdd, 0x4c, 0x6e, 0x84, 0x8b, 0x27 } };


/**
* �ò���ӿ�Ҫ���FTCoreƥ��汾��
*/
#define  FTCore_Support_Ver  101

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
	virtual void  GetPluginCallback_TradeHK(ITradeCallBack_HK** pCallback) = 0; 
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









