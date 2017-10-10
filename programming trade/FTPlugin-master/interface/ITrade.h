#pragma once

#ifndef interface
#define interface struct
#endif


/************************************************* 
Copyright: FUTU
Author: Lin
Date: 2015-03-18
Description: ����API�ͻص��ӿڶ���
**************************************************/  

enum Trade_Env
{
	Trade_Env_Real = 0, //��ʵ������ʵ�̽��ף�
	Trade_Env_Virtual = 1, //���⻷�������潻�׻�ģ�⽻�ף�
};

enum Trade_SvrResult
{
	Trade_SvrResult_Succeed = 0, //����������������ɹ�
	Trade_SvrResult_Failed = -1, //����������������ʧ�ܣ�ʧ�ܵ�ԭ����ܺܶ࣬�������糬ʱ����ʱͳһ����ʧ�ܣ�
};

enum Trade_OrderSide
{
	Trade_OrderSide_Buy = 0, //����
	Trade_OrderSide_Sell = 1, //����
};

enum Trade_OrderType_HK
{
	//Ŀǰֻ֧��0��1��3
	Trade_OrderType_HK_EnhancedLimit = 0, //��ǿ�޼۵�(��ͨ����)
	Trade_OrderType_HK_Auction = 1, //���۵�(���۽���)
	Trade_OrderType_HK_Limit = 2, //�޼۵�
	Trade_OrderType_HK_AuctionLimit = 3, //�����޼۵�(�����޼�)
	Trade_OrderType_HK_SpecialLimit = 4, //�ر��޼۵�
};

enum Trade_OrderStatus_HK
{
	Trade_OrderStatus_HK_Processing = 0, //������������...
	Trade_OrderStatus_HK_WaitDeal = 1, //�ȴ��ɽ�
	Trade_OrderStatus_HK_PartDealt = 2, //���ֳɽ�
	Trade_OrderStatus_HK_AllDealt = 3, //ȫ���ɽ�
	Trade_OrderStatus_HK_Disabled = 4, //��ʧЧ
	Trade_OrderStatus_HK_Failed = 5, //�µ�ʧ�ܣ������Ѳ�����������ʧ�ܣ�
	Trade_OrderStatus_HK_Cancelled = 6, //�ѳ���
	Trade_OrderStatus_HK_Deleted = 7, //��ɾ��
	Trade_OrderStatus_HK_WaitOpen = 8, //�ȴ�����
	Trade_OrderStatus_HK_LocalSent = 21, //�����ѷ���
	Trade_OrderStatus_HK_LocalFailed = 22, //�����ѷ��ͣ������������µ�ʧ�ܣ�û��������
	Trade_OrderStatus_HK_LocalTimeOut = 23, //�����ѷ��ͣ��ȴ����������س�ʱ
};

enum Trade_SetOrderStatus_HK
{
	Trade_SetOrderStatus_HK_Cancel = 0, //����
	Trade_SetOrderStatus_HK_Disable = 1, //ʧЧ
	Trade_SetOrderStatus_HK_Enable = 2, //��Ч
	Trade_SetOrderStatus_HK_Delete = 3, //ɾ��

	//���²���������״̬�����յ�OnOrderErrNotify��Զ�Ӧ�Ĵ�����ʾ����ȷ��
	Trade_SetOrderStatus_HK_SplitLargeOrder = 11, //ȷ�ϲ�ִ�
	Trade_SetOrderStatus_HK_PriceTooFar = 12, //ȷ�ϼ۸�ƫ��̫��Ҳ����
	Trade_SetOrderStatus_HK_BuyWolun = 13, //ȷ����������
	Trade_SetOrderStatus_HK_BuyGuQuan = 14, //ȷ��������Ȩ
	Trade_SetOrderStatus_HK_BuyLowPriceStock = 15, //ȷ�������ͼ۹�
};

enum Trade_OrderErrNotify_HK
{
	Trade_OrderErrNotify_HK_Normal = 0, //��ͨ����

	//���´�������ʾ�Եģ������ȷ�϶�������ʧ��״̬�����ͨ��
	//SetOrderStatus�Զ�Ӧ��������ȷ�ϣ��򶩵�������Ч
	Trade_OrderErrNotify_HK_LargeOrder = 1, //��������̫��
	Trade_OrderErrNotify_HK_PriceTooFar = 2, //�����۸�ƫ��̫��
	Trade_OrderErrNotify_HK_FengXian_Wolun = 3, //���������з���
	Trade_OrderErrNotify_HK_FengXian_GuQuan = 4, //������Ȩ�з���
	Trade_OrderErrNotify_HK_FengXian_LowPriceStock = 5, //�����ͼ۹��з���
};

struct Trade_OrderItem_HK
{
	UINT64 nLocalID; //�ͻ��˲����Ķ���ID���Ƕ���������ID�����ڹ���
	UINT64 nOrderID; //�����ţ������������Ķ���������ID

	Trade_OrderType_HK enType;
	Trade_OrderSide enSide;
	Trade_OrderStatus_HK enStatus;
	WCHAR szCode[16];
	WCHAR szName[128];
	UINT64 nPrice;
	UINT64 nQty;
	UINT64 nDealtQty; //�ɽ�����
	double fDealtAvgPrice; //�ɽ�����
	
	UINT64 nSubmitedTime; //�������յ��Ķ����ύʱ��
	UINT64 nUpdatedTime; //���������µ�ʱ��

	UINT16 nErrCode; //������
};

interface ITrade_HK
{
	/**
	* �µ�

	* @param enEnv ���׻���(ʵ�̽��׻���潻��).
	* @param pCookie ���ձ��ε��ö�Ӧ��Cookieֵ�����ڷ���������ʱ����Ӧ��ϵ�ж�.
	* @param enType ��������.
	* @param enSide �������������.
	* @param lpszCode ��Ʊ����.
	* @param nPrice �����۸�.
	* @param nQty ��������.

	* @return true���ͳɹ���false����ʧ��.
	*/
	virtual bool PlaceOrder(Trade_Env enEnv, UINT* pCookie, Trade_OrderType_HK enType, Trade_OrderSide enSide, LPCWSTR lpszCode, UINT64 nPrice, UINT64 nQty) = 0;

	/**
	* ���ö���״̬

	* @param enEnv ���׻���(ʵ�̽��׻���潻��).
	* @param pCookie ���ձ��ε��ö�Ӧ��Cookieֵ�����ڷ���������ʱ����Ӧ��ϵ�ж�.
	* @param nOrderID ����������ID.
	* @param enStatus ����Ϊ����״̬.

	* @return true���ͳɹ���false����ʧ��.
	*/
	virtual bool SetOrderStatus(Trade_Env enEnv, UINT* pCookie, UINT64 nOrderID, Trade_SetOrderStatus_HK enStatus) = 0;

	/**
	* �ĵ�

	* @param enEnv ���׻���(ʵ�̽��׻���潻��).
	* @param pCookie ���ձ��ε��ö�Ӧ��Cookieֵ�����ڷ���������ʱ����Ӧ��ϵ�ж�.
	* @param nOrderID ����������ID.
	* @param nPrice �µĶ����۸�.
	* @param nQty �µĶ�������.

	* @return true���ͳɹ���false����ʧ��.
	*/
	virtual bool ChangeOrder(Trade_Env enEnv, UINT* pCookie, UINT64 nOrderID, UINT64 nPrice, UINT64 nQty) = 0;

	/**
	* ͨ��������õ���������

	* @param nErrCode ������.
	* @param szErrDesc ��������.

	* @return true��ȡ�ɹ���false��ȡʧ��.
	*/
	virtual bool GetErrDesc(UINT16 nErrCode, WCHAR szErrDesc[256]) = 0;
};

interface ITradeCallBack_HK
{
	/**
	* �µ����󷵻�

	* @param enEnv ���׻���(ʵ�̽��׻���潻��).
	* @param nCookie ����ʱ��Cookie.
	* @param enSvrRet ������������.
	* @param nLocalID �ͻ��˲����Ķ���ID��������������Ͷ�������.
	* @param nErrCode ������.
	*/
	virtual void OnPlaceOrder(Trade_Env enEnv, UINT nCookie, Trade_SvrResult enSvrRet, UINT64 nLocalID, UINT16 nErrCode) = 0;

	/**
	* ������������

	* @param enEnv ���׻���(ʵ�̽��׻���潻��).
	* @param orderItem �����ṹ��.
	*/
	virtual void OnOrderUpdate(Trade_Env enEnv, const Trade_OrderItem_HK& orderItem) = 0;

	/**
	* ���ö���״̬���󷵻�

	* @param enEnv ���׻���(ʵ�̽��׻���潻��).
	* @param nCookie ����ʱ��Cookie.
	* @param enSvrRet ������������.
	* @param nOrderID ������.
	* @param nErrCode ������.
	*/
	virtual void OnSetOrderStatus(Trade_Env enEnv, UINT nCookie, Trade_SvrResult enSvrRet, UINT64 nOrderID, UINT16 nErrCode) = 0;

	/**
	* �ĵ����󷵻�

	* @param enEnv ���׻���(ʵ�̽��׻���潻��).
	* @param nCookie ����ʱ��Cookie.
	* @param enSvrRet ������������.
	* @param nOrderID ������.
	* @param nErrCode ������.
	*/
	virtual void OnChangeOrder(Trade_Env enEnv, UINT nCookie, Trade_SvrResult enSvrRet, UINT64 nOrderID, UINT16 nErrCode) = 0;

	/**
	* ������������
	
	* @param enEnv ���׻���(ʵ�̽��׻���潻��).
	* @param nOrderID ������.
	* @param enErrNotify ������������.
	* @param nErrCode ������.
	*/
	virtual void OnOrderErrNotify(Trade_Env enEnv, UINT64 nOrderID, Trade_OrderErrNotify_HK enErrNotify, UINT16 nErrCode) = 0;
};
