#pragma once

#ifndef interface
#define interface struct
#endif

 
/************************************************* 
Copyright: FUTU
Author: ysq
Date: 2015-03-18
Description: ����API�ͻص��ӿڶ���
**************************************************/  

/**
*��Ʊ���г����� 
*/ 
enum StockMktType 
{	
	StockMkt_HK = 1,  //�۹� 
	StockMkt_US = 2,  //����
	StockMkt_SH = 3,  //����
	StockMkt_SZ = 4,  //���
}; 

enum StockSubErrCode
{
	StockSub_Suc = 0,	//���ĳɹ�
	StockSub_FailUnknown	= 1,	//δ֪��ʧ��
	StockSub_FailMaxSubNum	= 2,	//�����������
	StockSub_FailCodeNoFind = 3,	//����û�ҵ�(Ҳ�п������г����ʹ���)
	StockSub_FailGuidNoFind = 4,	//���GUID����
	StockSub_FailNoImplInf = 5,		//����ӿ�δ���
};

/**
* ��Ʊ����������Ϣ��
* �۸񾫶���3λС���� �籨��8.888�洢ֵ 88888
*/
typedef struct tagQuotePriceBase
{ 
	DWORD dwOpen;		//���̼�
	DWORD dwLastClose;  //���ռ�
	DWORD dwCur;		//��ǰ��
	DWORD dwHigh;		//��߼�
	DWORD dwLow;		//��ͼ�
	INT64 ddwVolume;	//�ɽ���
	INT64 ddwTrunover;	//�ɽ���
	DWORD dwTime;		//����ʱ��
}Quote_PriceBase, *LPQuote_PriceBase;


/**
* ��Ʊʮ������
* IFTQuoteData::FillOrderQueue �Ľӿڲ���  
*/
typedef struct tagQuoteOrderQueue
{
	DWORD	dwBuyPrice, dwSellPrice;  //��� ����
	INT64	ddwBuyVol, ddwSellVol;    //���� ����
	int		nBuyOrders, nSellOrders;  //��λ 
}Quote_OrderQueue, *LPQuote_OrderQueue;  


/**
* ��������ӿ� 
*/
interface IFTQuoteOperation 
{
	//���鶨�ģ����ش�����
	virtual StockSubErrCode Subscribe_PriceBase(const GUID &guidPlugin, LPCWSTR wstrStockCode,  StockMktType eType, bool bSubb) = 0;  
	virtual StockSubErrCode Subscribe_OrderQueue(const GUID &guidPlugin, LPCWSTR wstrStockCode, StockMktType eType, bool bSubb) = 0; 
};

/**
* �������ݵĽӿ�
*/
interface IFTQuoteData
{ 
	/**
	* ��ǰ�Ƿ���ʵʱ����
	*/
	virtual bool   IsRealTimeQuotes() = 0; 

	/**
	* stock ��hashֵ, �ص��ӿڷ��� 
	*/ 
	virtual INT64  GetStockHashVal(LPCWSTR pstrStockCode, StockMktType eMkt) = 0; 

	/**
	* ���������� 
	*/ 
	virtual bool   FillPriceBase(INT64 ddwStockHash,  Quote_PriceBase* pInfo) = 0; 

	/**
	* ���ʮ������
	*/ 
	virtual bool   FillOrderQueue(INT64 ddwStockHash, Quote_OrderQueue* parQuote, int nCount) = 0; 
}; 

/**
* �������ݻص�
*/
interface IQuoteInfoCallback
{ 
	/**
	* ����������Ϣ�仯 
	*/ 
	virtual void  OnChanged_PriceBase(INT64  ddwStockHash) = 0; 

	/**
	* ʮ�����ݱ仯
	*/ 
	virtual void  OnChanged_OrderQueue(INT64 ddwStockHash) = 0; 
};

