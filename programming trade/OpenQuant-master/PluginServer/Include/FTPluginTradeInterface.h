#pragma once

#ifndef interface
#define interface struct
#endif

/*************************************************
Copyright: FUTU
Author: Lin
Date: 2015-03-18
Description: ����API�ͻص��ӿڶ���
Ver: 1.2

���¼�¼:
1. 2015-12-28 ���Ӳ�ѯ�������˻����ֲֽӿڣ��汾����Ϊ1.1
2. 2016-01-07 ��������API���汾����Ϊ1.2
�ر����ѣ�������������������������������������������������������������
����API�м۸񡢽�����������Ϊ�����ͣ�����ԭʼ����û�б��Ŵ��������ͣ����Ǹ���ֵ��1000������С��λ��0.001Ԫ
**************************************************/

enum Trade_Env
{
	Trade_Env_Real = 0, //��ʵ������ʵ�̽���, Ŀǰ֧�ָ۹�/���ɣ�
	Trade_Env_Virtual = 1, //���⻷�������潻�׻�ģ�⽻��, Ŀǰ��֧�ָ۹ɣ�
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
	Trade_OrderSide_SellShort = 2, //����(Ŀǰ������)
	Trade_OrderSide_BuyBack = 3, //���ղ���(Ŀǰ������)
};

enum Trade_OrderStatus
{
	Trade_OrderStatus_Processing = 0, //������������...
	Trade_OrderStatus_WaitDeal = 1, //�ȴ��ɽ�
	Trade_OrderStatus_PartDealt = 2, //���ֳɽ�
	Trade_OrderStatus_AllDealt = 3, //ȫ���ɽ�
	Trade_OrderStatus_Disabled = 4, //��ʧЧ
	Trade_OrderStatus_Failed = 5, //�µ�ʧ�ܣ������Ѳ�����������ʧ�ܣ�
	Trade_OrderStatus_Cancelled = 6, //�ѳ���
	Trade_OrderStatus_Deleted = 7, //��ɾ��
	Trade_OrderStatus_WaitOpen = 8, //�ȴ�����
	Trade_OrderStatus_LocalSent = 21, //�����ѷ���
	Trade_OrderStatus_LocalFailed = 22, //�����ѷ��ͣ������������µ�ʧ�ܣ�û��������
	Trade_OrderStatus_LocalTimeOut = 23, //�����ѷ��ͣ��ȴ����������س�ʱ
};

//�۹ɵ�һЩö�ٶ���
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum Trade_OrderType_HK
{
	//Ŀǰֻ֧��0��1��3

	Trade_OrderType_HK_EnhancedLimit = 0, //��ǿ�޼۵�(��ͨ����)
	Trade_OrderType_HK_Auction = 1, //���۵�(���۽���)
	Trade_OrderType_HK_Limit = 2, //�޼۵�
	Trade_OrderType_HK_AuctionLimit = 3, //�����޼۵�(�����޼�)
	Trade_OrderType_HK_SpecialLimit = 4, //�ر��޼۵�
};

enum Trade_SetOrderStatus
{
	Trade_SetOrderStatus_Cancel = 0, //����
	Trade_SetOrderStatus_Disable = 1, //ʧЧ
	Trade_SetOrderStatus_Enable = 2, //��Ч
	Trade_SetOrderStatus_Delete = 3, //ɾ��

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

	//���´�������ʾ�Եģ������ȷ�϶�������ʧ��״̬�����ͨ��SetOrderStatus�Զ�Ӧ��������ȷ�ϣ��򶩵�������Ч

	Trade_OrderErrNotify_HK_LargeOrder = 1, //��������̫��
	Trade_OrderErrNotify_HK_PriceTooFar = 2, //�����۸�ƫ��̫��
	Trade_OrderErrNotify_HK_FengXian_Wolun = 3, //���������з���
	Trade_OrderErrNotify_HK_FengXian_GuQuan = 4, //������Ȩ�з���
	Trade_OrderErrNotify_HK_FengXian_LowPriceStock = 5, //�����ͼ۹��з���
};

//���ɵ�һЩö�ٶ���
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum Trade_OrderType_US
{
	Trade_OrderType_US_Market = 1, //�м۵�
	Trade_OrderType_US_Limit = 2, //�޼�

	Trade_OrderType_US_PreMarket = 51, //��ǰ���ף��޼�
	Trade_OrderType_US_PostMarket = 52 //�̺��ף��޼�
};

//�������ɽ���¼���˻����ֲ����ݽṹ
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct Trade_OrderItem
{
	//�ر����ѣ�������������������������������������������������������������
	//����API�м۸񡢽�����������Ϊ�����ͣ�����ԭʼ����û�б��Ŵ��������ͣ����Ǹ���ֵ��1000������С��λ��0.001Ԫ

	UINT64 nLocalID; //�ͻ��˲����Ķ���ID���Ƕ���������ID�����ڹ���
	UINT64 nOrderID; //�����ţ������������Ķ���������ID

	UINT8 nType; //��ͬ�г���ȡֵ��Ӧ�����ö�ٶ��� Trade_OrderType_HK �� Trade_OrderType_US
	Trade_OrderSide enSide;
	UINT8 nStatus; //ȡֵ��Ӧ�����ö�ٶ���Trade_OrderStatus
	WCHAR szCode[16];
	WCHAR szName[128];
	UINT64 nPrice;
	UINT64 nQty;
	UINT64 nDealtQty; //�ɽ�����
	double fDealtAvgPrice; //�ɽ����ۣ�û�зŴ�

	UINT64 nSubmitedTime; //�������յ��Ķ����ύʱ��
	UINT64 nUpdatedTime; //���������µ�ʱ��

	//ֻ֧�ָ۹ɵ���GetErrDesc��nErrCode���õ����������������붼����GetErrDescV2��nErrCode��nErrDescStrHash
	UINT16 nErrCode; //�����룬��֧�ָ۹�
	INT64 nErrDescStrHash; //���������ַ�����hash
};

struct Trade_DealItem
{
	//�ر����ѣ�������������������������������������������������������������
	//����API�м۸񡢽�����������Ϊ�����ͣ�����ԭʼ����û�б��Ŵ��������ͣ����Ǹ���ֵ��1000������С��λ��0.001Ԫ

	UINT64 nOrderID; //�����ţ������������Ķ���������ID
	UINT64 nDealID; //�ɽ���

	Trade_OrderSide enSide; //����

	WCHAR szCode[16]; //����
	WCHAR szName[128]; //����
	UINT64 nPrice; //�ɽ��۸�
	UINT64 nQty; //�ɽ�����

	UINT64 nTime;	//�ɽ�ʱ��
};

struct Trade_AccInfo
{
	//�ر����ѣ�������������������������������������������������������������
	//����API�м۸񡢽�����������Ϊ�����ͣ�����ԭʼ����û�б��Ŵ��������ͣ����Ǹ���ֵ��1000������С��λ��0.001Ԫ

	INT64 nPower; //������
	INT64 nZcjz; //�ʲ���ֵ
	INT64 nZqsz; //֤ȯ��ֵ
	INT64 nXjjy; //�ֽ����
	INT64 nKqxj; //��ȡ�ֽ�
	INT64 nDjzj; //�����ʽ�
	INT64 nZsje; //׷�ս��

	INT64 nZgjde; //��߽����
	INT64 nYyjde; //�����Ŵ���
	INT64 nGpbzj; //��Ʊ��֤��
};

struct Trade_PositionItem
{
	//�ر����ѣ�������������������������������������������������������������
	//����API�м۸񡢽�����������Ϊ�����ͣ�����ԭʼ����û�б��Ŵ��������ͣ����Ǹ���ֵ��1000������С��λ��0.001Ԫ

	WCHAR szCode[16];
	WCHAR szName[128];

	INT64 nQty; //��������
	INT64 nCanSellQty; //��������
	INT64 nNominalPrice; //�м�
	INT64 nMarketVal; //��ֵ

	double fCostPrice; //�ɱ���
	bool bCostPriceValid; //�ɱ����Ƿ���Ч
	INT64 nPLVal; //ӯ�����
	bool bPLValValid; //ӯ������Ƿ���Ч
	double fPLRatio; //ӯ������
	bool bPLRatioValid; //ӯ�������Ƿ���Ч

	INT64 nToday_PLVal; //����ӯ�����
	INT64 nToday_BuyQty; //��������ɽ���
	INT64 nToday_BuyVal; //��������ɽ���
	INT64 nToday_SellQty; //���������ɽ���
	INT64 nToday_SellVal; //���������ɽ���
};

//�۹ɽ���API����/�ص��ӿڶ���
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
*	�۹ɽ��׽ӿ� ITrade_HK, �������ʵ�֣����ͨ����ѯIFTPluginCore::QueryFTInterface�õ�
*/
static const GUID IID_IFTTrade_HK =
{ 0x69a88049, 0x252e, 0x4a12, { 0x83, 0x41, 0xdd, 0x4c, 0x6e, 0x84, 0x8b, 0x27 } };

interface ITrade_HK
{
	/**
	* ����

	* @param pCookie ���ձ��ε��ö�Ӧ��Cookieֵ�����ڷ���������ʱ����Ӧ��ϵ�ж�.
	* @param lpszPassword ����.

	* @return true���ͳɹ���false����ʧ��.
	*/
	virtual bool UnlockTrade(UINT32* pCookie, LPCWSTR lpszPassword) = 0;

	/**
	* �µ�

	* @param enEnv ���׻���(ʵ�̽��׻���潻��).
	* @param pCookie ���ձ��ε��ö�Ӧ��Cookieֵ�����ڷ���������ʱ����Ӧ��ϵ�ж�.
	* @param enType ��������.
	* @param enSide �������������.
	* @param lpszCode ��Ʊ����.
	* @param nPrice �����۸�. ע�⣺�Ǹ���ֵ��1000������С��λ��0.001Ԫ
	* @param nQty ��������.
	* @param pnResult ���ش�����, Ŀǰ���ܵĴ�����: QueryData_FailFreqLimit, QueryData_FailNetwork

	* @return true���ͳɹ���false����ʧ��.
	*/
	virtual bool PlaceOrder(Trade_Env enEnv, UINT32* pCookie, Trade_OrderType_HK enType, 
		Trade_OrderSide enSide, LPCWSTR lpszCode, UINT64 nPrice, UINT64 nQty, int* pnResult = NULL) = 0;

	/**
	* �õ�����״̬

	* @param enEnv ���׻���(ʵ�̽��׻���潻��).
	* @param nOrderID ����������ID.
	* @param eStatus ���ض���״̬
	* @return true/false �����Ƿ����
	*/
	virtual bool GetOrderStatus(Trade_Env enEnv, UINT64 nOrderID, Trade_OrderStatus& eStatus) = 0;

	/**
	* ���ö���״̬

	* @param enEnv ���׻���(ʵ�̽��׻���潻��).
	* @param pCookie ���ձ��ε��ö�Ӧ��Cookieֵ�����ڷ���������ʱ����Ӧ��ϵ�ж�.
	* @param nOrderID ����������ID.
	* @param enStatus ����Ϊ����״̬.
	* @param pnResult ���ش�����, Ŀǰ���ܵĴ�����: QueryData_FailFreqLimit, QueryData_FailNetwork

	* @return true���ͳɹ���false����ʧ��.
	*/
	virtual bool SetOrderStatus(Trade_Env enEnv, UINT32* pCookie, UINT64 nOrderID, 
		Trade_SetOrderStatus enStatus, int* pnResult = NULL) = 0;

	/**
	* �ĵ�

	* @param enEnv ���׻���(ʵ�̽��׻���潻��).
	* @param pCookie ���ձ��ε��ö�Ӧ��Cookieֵ�����ڷ���������ʱ����Ӧ��ϵ�ж�.
	* @param nOrderID ����������ID.
	* @param nPrice �µĶ����۸�. ע�⣺�Ǹ���ֵ��1000������С��λ��0.001Ԫ
	* @param nQty �µĶ�������.
	* @param pnResult ���ش�����, Ŀǰ���ܵĴ�����: QueryData_FailFreqLimit, QueryData_FailNetwork

	* @return true���ͳɹ���false����ʧ��.
	*/
	virtual bool ChangeOrder(Trade_Env enEnv, UINT32* pCookie, UINT64 nOrderID, UINT64 nPrice, 
		UINT64 nQty, int* pnResult = NULL) = 0;

	/**
	* ͨ��������õ���������

	* @param nErrCode ������.
	* @param szErrDesc ��������.

	* @return true��ȡ�ɹ���false��ȡʧ��.
	*/
	virtual bool GetErrDesc(UINT16 nErrCode, WCHAR szErrDesc[256]) = 0;

	/**
	* ͨ����������������Hash�õ���������

	* @param nErrCodeOrHash ��������������Hash.
	* @param szErrDesc ��������.

	* @return true��ȡ�ɹ���false��ȡʧ��.
	*/
	virtual bool GetErrDescV2(INT64 nErrCodeOrHash, WCHAR szErrDesc[256]) = 0;

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//����Ϊ��ѯ����ʱ�����ڴ����ݣ���������������ţţ�ͻ�����ʾ����һ�£��䲻�����������������������ͻ�����TCP�����ӣ���������ʵʱ������������

	/**
	* ��ѯ�����б�

	* @param enEnv ���׻���(ʵ�̽��׻���潻��).
	* @param pCookie ���ձ��ε��ö�Ӧ��Cookieֵ�����ڲ�ѯ����ص�ʱ����Ӧ��ϵ�ж�.

	* @return true��ѯ�ɹ���false��ѯʧ��.
	*/
	virtual bool QueryOrderList(Trade_Env enEnv, UINT32* pCookie) = 0;

	/**
	* ��ѯ�ɽ���¼�б�

	* @param enEnv ���׻���(ʵ�̽��׻���潻��).
	* @param pCookie ���ձ��ε��ö�Ӧ��Cookieֵ�����ڲ�ѯ����ص�ʱ����Ӧ��ϵ�ж�.

	* @return true��ѯ�ɹ���false��ѯʧ��.
	*/
	virtual bool QueryDealList(Trade_Env enEnv, UINT32* pCookie) = 0;

	/**
	* ��ѯ�˻���Ϣ

	* @param enEnv ���׻���(ʵ�̽��׻���潻��).
	* @param pCookie ���ձ��ε��ö�Ӧ��Cookieֵ�����ڲ�ѯ����ص�ʱ����Ӧ��ϵ�ж�.

	* @return true��ѯ�ɹ���false��ѯʧ��.
	*/
	virtual bool QueryAccInfo(Trade_Env enEnv, UINT32* pCookie) = 0;

	/**
	* ��ѯ�ֲ��б�

	* @param enEnv ���׻���(ʵ�̽��׻���潻��).
	* @param pCookie ���ձ��ε��ö�Ӧ��Cookieֵ�����ڲ�ѯ����ص�ʱ����Ӧ��ϵ�ж�.

	* @return true��ѯ�ɹ���false��ѯʧ��.
	*/
	virtual bool QueryPositionList(Trade_Env enEnv, UINT32* pCookie) = 0;

	/**
	* ͨ�����ض���id�õ�svr����id

	* @param Trade_Env  ���׻���(ʵ�̽��׻���潻��).
	* @param nLocalID   ��������id.

	* @return  ����ServerID , �����û���ɻ��߲��Ҳ�������0
	*/
	virtual INT64 FindOrderSvrID(Trade_Env enEnv, INT64 nLocalID) = 0;

};

interface ITradeCallBack_HK
{
	/**
	* �����������󷵻�

	* @param nCookie ����ʱ��Cookie.
	* @param enSvrRet ������������.
	* @param nErrCode ������.
	*/
	virtual void OnUnlockTrade(UINT32 nCookie, Trade_SvrResult enSvrRet, UINT64 nErrCode) = 0;

	/**
	* �µ����󷵻�

	* @param enEnv ���׻���(ʵ�̽��׻���潻��).
	* @param nCookie ����ʱ��Cookie.
	* @param enSvrRet ������������.
	* @param nLocalID �ͻ��˲����Ķ���ID��������������Ͷ�������.
	* @param nErrCode ������.
	*/
	virtual void OnPlaceOrder(Trade_Env enEnv, UINT32 nCookie, Trade_SvrResult enSvrRet, UINT64 nLocalID, UINT16 nErrCode) = 0;

	/**
	* ������������

	* @param enEnv ���׻���(ʵ�̽��׻���潻��).
	* @param orderItem �����ṹ��.
	*/
	virtual void OnOrderUpdate(Trade_Env enEnv, const Trade_OrderItem& orderItem) = 0;

	/**
	* ���ö���״̬���󷵻�

	* @param enEnv ���׻���(ʵ�̽��׻���潻��).
	* @param nCookie ����ʱ��Cookie.
	* @param enSvrRet ������������.
	* @param nOrderID ������.
	* @param nErrCode ������.
	*/
	virtual void OnSetOrderStatus(Trade_Env enEnv, UINT32 nCookie, Trade_SvrResult enSvrRet, UINT64 nOrderID, UINT16 nErrCode) = 0;

	/**
	* �ĵ����󷵻�

	* @param enEnv ���׻���(ʵ�̽��׻���潻��).
	* @param nCookie ����ʱ��Cookie.
	* @param enSvrRet ������������.
	* @param nOrderID ������.
	* @param nErrCode ������.
	*/
	virtual void OnChangeOrder(Trade_Env enEnv, UINT32 nCookie, Trade_SvrResult enSvrRet, UINT64 nOrderID, UINT16 nErrCode) = 0;

	/**
	* ������������

	* @param enEnv ���׻���(ʵ�̽��׻���潻��).
	* @param nOrderID ������.
	* @param enErrNotify ������������.
	* @param nErrCode ������.
	*/
	virtual void OnOrderErrNotify(Trade_Env enEnv, UINT64 nOrderID, Trade_OrderErrNotify_HK enErrNotify, UINT16 nErrCode) = 0;

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//����Ϊ��ѯ����ʱ�����ڴ����ݵĻص���������������ţţ�ͻ�����ʾ����һ�£��䲻�����������������������ͻ�����TCP�����ӣ���������ʵʱ������������

	/**
	* ��ѯ�����б�ص�

	* @param enEnv ���׻���(ʵ�̽��׻���潻��).
	* @param nCookie ����ʱ��Cookie.
	* @param nCount ��������.
	* @param pArrOrder ��������ָ��.
	*/
	virtual void OnQueryOrderList(Trade_Env enEnv, UINT32 nCookie, INT32 nCount, const Trade_OrderItem* pArrOrder) = 0;

	/**
	* ��ѯ�ɽ���¼�б�ص�

	* @param enEnv ���׻���(ʵ�̽��׻���潻��).
	* @param nCookie ����ʱ��Cookie.
	* @param nCount �ɽ���¼����.
	* @param pArrDeal �ɽ���¼����ָ��.
	*/
	virtual void OnQueryDealList(Trade_Env enEnv, UINT32 nCookie, INT32 nCount, const Trade_DealItem* pArrDeal) = 0;

	/**
	* ��ѯ�˻���Ϣ�ص�

	* @param enEnv ���׻���(ʵ�̽��׻���潻��).
	* @param nCookie ����ʱ��Cookie.
	* @param accInfo �˻���Ϣ�ṹ��.
	*/
	virtual void OnQueryAccInfo(Trade_Env enEnv, UINT32 nCookie, const Trade_AccInfo& accInfo, int nResult) = 0;

	/**
	* ��ѯ�ֲ��б�ص�

	* @param enEnv ���׻���(ʵ�̽��׻���潻��).
	* @param nCookie ����ʱ��Cookie.
	* @param nCount �ֲָ���.
	* @param pArrPosition �ֲ�����ָ��.
	*/
	virtual void OnQueryPositionList(Trade_Env enEnv, UINT32 nCookie, INT32 nCount, const Trade_PositionItem* pArrPosition) = 0;
};

//���ɽ���API����/�ص��ӿڶ���
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
*	���ɽ��׽ӿ� ITrade_US, �������ʵ�֣�ͨ����ѯIFTPluginCore::QueryFTInterface�õ�
*/
static const GUID IID_IFTTrade_US =
{ 0x66c2e76d, 0x8786, 0x4bf0, { 0x95, 0x34, 0xd2, 0x86, 0x4d, 0x53, 0x9, 0xc6 } };
interface ITrade_US
{
	/**
	* �µ�

	* @param pCookie ���ձ��ε��ö�Ӧ��Cookieֵ�����ڷ���������ʱ����Ӧ��ϵ�ж�.
	* @param enType ��������.
	* @param enSide �������������.
	* @param lpszCode ��Ʊ����.
	* @param nPrice �����۸�. ע�⣺�Ǹ���ֵ��1000������С��λ��0.001Ԫ
	* @param nQty ��������.
	* @param pnResult ���ش�����, Ŀǰ���ܵĴ�����: QueryData_FailFreqLimit, QueryData_FailNetwork

	* @return true���ͳɹ���false����ʧ��.
	*/
	virtual bool PlaceOrder(UINT32* pCookie, Trade_OrderType_US enType, Trade_OrderSide enSide, 
		LPCWSTR lpszCode, UINT64 nPrice, UINT64 nQty, int* pnResult = NULL) = 0;

	/**
	* ��������

	* @param pCookie ���ձ��ε��ö�Ӧ��Cookieֵ�����ڷ���������ʱ����Ӧ��ϵ�ж�.
	* @param nOrderID ����������ID.
	* @param pnResult ���ش�����, Ŀǰ���ܵĴ�����: QueryData_FailFreqLimit, QueryData_FailNetwork

	* @return true���ͳɹ���false����ʧ��.
	*/
	virtual bool CancelOrder(UINT32* pCookie, UINT64 nOrderID, int* pnResult = NULL) = 0;

	/**
	* �ĵ�

	* @param pCookie ���ձ��ε��ö�Ӧ��Cookieֵ�����ڷ���������ʱ����Ӧ��ϵ�ж�.
	* @param nOrderID ����������ID.
	* @param nPrice �µĶ����۸�.  ע�⣺�Ǹ���ֵ��1000������С��λ��0.001Ԫ
	* @param nQty �µĶ�������.
	* @param pnResult ���ش�����, Ŀǰ���ܵĴ�����: QueryData_FailFreqLimit, QueryData_FailNetwork

	* @return true���ͳɹ���false����ʧ��.
	*/
	virtual bool ChangeOrder(UINT32* pCookie, UINT64 nOrderID, UINT64 nPrice, UINT64 nQty, int* pnResult = NULL) = 0;

	/**
	* ͨ����������Hash�õ���������

	* @param nErrHash ��������Hash.
	* @param szErrDesc ��������.

	* @return true��ȡ�ɹ���false��ȡʧ��.
	*/
	virtual bool GetErrDescV2(INT64 nErrHash, WCHAR szErrDesc[256]) = 0;

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//����Ϊ��ѯ����ʱ�����ڴ����ݣ���������������ţţ�ͻ�����ʾ����һ�£��䲻�����������������������ͻ�����TCP�����ӣ���������ʵʱ������������

	/**
	* ��ѯ�����б�

	* @param pCookie ���ձ��ε��ö�Ӧ��Cookieֵ�����ڲ�ѯ����ص�ʱ����Ӧ��ϵ�ж�.

	* @return true��ѯ�ɹ���false��ѯʧ��.
	*/
	virtual bool QueryOrderList(UINT32* pCookie) = 0;

	/**
	* ��ѯ�ɽ���¼�б�

	* @param pCookie ���ձ��ε��ö�Ӧ��Cookieֵ�����ڲ�ѯ����ص�ʱ����Ӧ��ϵ�ж�.

	* @return true��ѯ�ɹ���false��ѯʧ��.
	*/
	virtual bool QueryDealList(UINT32* pCookie) = 0;

	/**
	* ��ѯ�˻���Ϣ

	* @param pCookie ���ձ��ε��ö�Ӧ��Cookieֵ�����ڲ�ѯ����ص�ʱ����Ӧ��ϵ�ж�.

	* @return true��ѯ�ɹ���false��ѯʧ��.
	*/
	virtual bool QueryAccInfo(UINT32* pCookie) = 0;

	/**
	* ��ѯ�ֲ��б�

	* @param pCookie ���ձ��ε��ö�Ӧ��Cookieֵ�����ڲ�ѯ����ص�ʱ����Ӧ��ϵ�ж�.

	* @return true��ѯ�ɹ���false��ѯʧ��.
	*/
	virtual bool QueryPositionList(UINT32* pCookie) = 0;

	/**
	* ͨ�����ض���id�õ�svr����id

	* @param nLocalID   ��������id.

	* @return  ����ServerID , �����û���ɻ��߲��Ҳ�������0
	*/
	virtual INT64 FindOrderSvrID(INT64 nLocalID) = 0;

	/**
	* �õ�����״̬

	* @param enEnv ���׻���(ʵ�̽��׻���潻��).
	* @param nOrderID ����������ID.
	* @param eStatus ���ض���״̬
	* @return true/false �����Ƿ����
	*/
	virtual bool GetOrderStatus(UINT64 nOrderID, Trade_OrderStatus& eStatus) = 0;

};

interface ITradeCallBack_US
{
	/**
	* �µ����󷵻�

	* @param nCookie ����ʱ��Cookie.
	* @param enSvrRet ������������.
	* @param nLocalID �ͻ��˲����Ķ���ID��������������Ͷ�������.
	* @param nErrHash ��������Hash.
	*/
	virtual void OnPlaceOrder(UINT32 nCookie, Trade_SvrResult enSvrRet, UINT64 nLocalID, INT64 nErrHash) = 0;

	/**
	* ������������

	* @param orderItem �����ṹ��.
	*/
	virtual void OnOrderUpdate(const Trade_OrderItem& orderItem) = 0;

	/**
	* ���ö���״̬���󷵻�

	* @param nCookie ����ʱ��Cookie.
	* @param enSvrRet ������������.
	* @param nOrderID ������.
	* @param nErrHash ��������Hash.
	*/
	virtual void OnCancelOrder(UINT32 nCookie, Trade_SvrResult enSvrRet, UINT64 nOrderID, INT64 nErrHash) = 0;

	/**
	* �ĵ����󷵻�

	* @param nCookie ����ʱ��Cookie.
	* @param enSvrRet ������������.
	* @param nOrderID ������.
	* @param nErrHash ��������Hash.
	*/
	virtual void OnChangeOrder(UINT32 nCookie, Trade_SvrResult enSvrRet, UINT64 nOrderID, INT64 nErrHash) = 0;

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//����Ϊ��ѯ����ʱ�����ڴ����ݵĻص���������������ţţ�ͻ�����ʾ����һ�£��䲻�����������������������ͻ�����TCP�����ӣ���������ʵʱ������������

	/**
	* ��ѯ�����б�ص�

	* @param nCookie ����ʱ��Cookie.
	* @param nCount ��������.
	* @param pArrOrder ��������ָ��.
	*/
	virtual void OnQueryOrderList(UINT32 nCookie, INT32 nCount, const Trade_OrderItem* pArrOrder) = 0;

	/**
	* ��ѯ���׼�¼�б�ص�

	* @param nCookie ����ʱ��Cookie.
	* @param nCount ���׼�¼����.
	* @param pArrDeal ���׼�¼����ָ��.
	*/
	virtual void OnQueryDealList(UINT32 nCookie, INT32 nCount, const Trade_DealItem* pArrDeal) = 0;

	/**
	* ��ѯ�˻���Ϣ�ص�

	* @param nCookie ����ʱ��Cookie.
	* @param accInfo �˻���Ϣ�ṹ��.
	*/
	virtual void OnQueryAccInfo(UINT32 nCookie, const Trade_AccInfo& accInfo, int nResult) = 0;

	/**
	* ��ѯ�ֲ��б�ص�

	* @param nCookie ����ʱ��Cookie.
	* @param nCount �ֲָ���.
	* @param pArrPosition �ֲ�����ָ��.
	*/
	virtual void OnQueryPositionList(UINT32 nCookie, INT32 nCount, const Trade_PositionItem* pArrPosition) = 0;
};
