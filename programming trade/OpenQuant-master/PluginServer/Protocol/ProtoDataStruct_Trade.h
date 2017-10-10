#pragma once
#include <vector>
#include "ProtoDataStruct.h"

//////////////////////////////////////////////////////////////////////////
//���͸۹ɶ���ʵʱ��Ϣ, PROTO_ID_TDHK_PUSH_ORDER_UPDATE
struct	OrderUpdatePushHKReqBody
{
};

struct OrderUpdatePushHKAckBody
{
	int nEnvType;
	INT64 nLocalID;
	INT64 nOrderID;
	int   nOrderDir;
	int	  nOrderTypeHK;
	int   nOrderStatusHK;	
	int   nPrice;
	INT64 nQTY;
	INT64 nDealQTY;
	int   nSubmitTime;
	int   nUpdateTime;
	std::string strStockName;
	std::string strStockCode;
	OrderUpdatePushHKAckBody()
	{
		nEnvType = 0;
		nLocalID = 0;
		nOrderID = 0;
		nOrderDir = 0;
		nOrderTypeHK = 0;
		nOrderStatusHK = 0;
		nPrice = 0;
		nQTY = 0;
		nDealQTY = 0;
		nSubmitTime = 0;
		nUpdateTime = 0;
	}
};

struct	OrderUpdatePushHK_Req
{
	ProtoHead					head;
	OrderUpdatePushHKReqBody	body;
};

struct	OrderUpdatePushHK_Ack
{
	ProtoHead					head;
	OrderUpdatePushHKAckBody	body;
};



//////////////////////////////////////////////////////////////////////////
//���͸۹ɶ���������Ϣ, PROTO_ID_TDHK_PUSH_ORDER_ERROR
struct	OrderErrorPushHKReqBody
{
};

struct OrderErrorPushHKAckBody
{	
	int nEnvType;
	INT64 nOrderID;
	int   nOrderErrNotifyHK;
	int	  nOrderErrCode;
	std::string  strOrderErrDesc;
	
	OrderErrorPushHKAckBody()
	{
		nEnvType = 0;
		nOrderID = 0;
		nOrderErrNotifyHK = 0;
		nOrderErrCode = 0;
	}
};

struct	OrderErrorPushHK_Req
{
	ProtoHead				head;
	OrderErrorPushHKReqBody	body;
};

struct	OrderErrorPushHK_Ack
{
	ProtoHead				head;
	OrderErrorPushHKAckBody	body;
};


//////////////////////////////////////////////////////////////////////////
//�¶��� PROTO_ID_TDHK_PLACE_ORDER 
struct	PlaceOrderReqBody
{
	int nEnvType;
	int nCookie;
	int nOrderDir;
	int nOrderType;
	int nPrice;
	INT64 nQty;
	std::string strCode;

	PlaceOrderReqBody()
	{
		nEnvType = 0;
		nCookie = 0;
		nOrderDir = 0;
		nOrderType = 0;
		nPrice = 0;
		nQty = 0;
	}
};

struct PlaceOrderAckBody
{	
	int nEnvType;
	int nCookie;
	INT64 nLocalID;
	int nSvrResult;	
	INT64	nSvrOrderID;

	PlaceOrderAckBody()
	{
		nEnvType = 0;
		nCookie = 0;
		nLocalID = 0;
		nSvrResult = 0;
		nSvrOrderID = 0;
	}
};

struct	PlaceOrder_Req
{
	ProtoHead			head;
	PlaceOrderReqBody	body;
};

struct	PlaceOrder_Ack
{
	ProtoHead				head;
	PlaceOrderAckBody		body;
};


//////////////////////////////////////////////////////////////////////////
//���ö���״̬ PROTO_ID_TDHK_SET_ORDER_STATUS
struct	SetOrderStatusReqBody
{
	int		nEnvType;
	int		nCookie;
	int		nSetOrderStatus;
	INT64	nSvrOrderID;
	INT64	nLocalOrderID;

	SetOrderStatusReqBody()
	{
		nEnvType = 0;
		nCookie = 0;
		nSetOrderStatus = 0;
		nSvrOrderID = 0;
		nLocalOrderID = 0;
	}

};

struct SetOrderStatusAckBody
{	
	int		nEnvType;
	int		nCookie;
	INT64	nSvrOrderID;
	INT64	nLocalOrderID;
	int		nSvrResult;	

	SetOrderStatusAckBody()
	{
		nEnvType = 0;
		nCookie = 0;
		nSvrOrderID = 0;
		nLocalOrderID = 0;
		nSvrResult = 0;
	}
};

struct	SetOrderStatus_Req
{
	ProtoHead				head;
	SetOrderStatusReqBody	body;
};

struct	SetOrderStatus_Ack
{
	ProtoHead				head;
	SetOrderStatusAckBody	body;
};


//////////////////////////////////////////////////////////////////////////
//��������
struct UnlockTradeReqBody
{
	int			nCookie;
	std::string strPasswd;

	UnlockTradeReqBody()
	{
		nCookie = 0;
	}
};

struct UnlockTradeAckBody
{
	int	nCookie;
	int nSvrResult;	
	std::string strSecNum;

	UnlockTradeAckBody()
	{
		nCookie = 0;
		nSvrResult = 0;
	}
};

struct UnlockTrade_Req
{
	ProtoHead				head;
	UnlockTradeReqBody		body;
};

struct UnlockTrade_Ack
{
	ProtoHead				head;
	UnlockTradeAckBody		body;
};


//////////////////////////////////////////////////////////////////////////
//�۹ɸĵ� PROTO_ID_TDHK_CHANGE_ORDER
struct	ChangeOrderReqBody
{
	int		nEnvType;
	int		nCookie;
	INT64	nSvrOrderID;
	INT64	nLocalOrderID;
	int		nPrice;
	INT64	nQty;

	ChangeOrderReqBody()
	{
		nEnvType = 0;
		nCookie = 0;
		nSvrOrderID = 0;
		nLocalOrderID = 0;
		nPrice = 0;
		nQty = 0;
	}
};

struct ChangeOrderAckBody
{	
	int		nEnvType;
	int		nCookie;
	INT64	nSvrOrderID;
	INT64	nLocalOrderID;
	int		nSvrResult;	

	ChangeOrderAckBody()
	{
		nEnvType = 0;
		nCookie = 0;
		nSvrOrderID = 0;
		nLocalOrderID = 0;
		nSvrResult = 0;
	}
};

struct	ChangeOrder_Req
{
	ProtoHead			head;
	ChangeOrderReqBody	body;
};

struct	ChangeOrder_Ack
{
	ProtoHead				head;
	ChangeOrderAckBody	body;
};

//////////////////////////////////////////////////////////////////////////
//��ȡ�û��۹��ʻ���Ϣ
struct	QueryHKAccInfoReqBody
{
	int		nEnvType;
	int		nCookie;	

	QueryHKAccInfoReqBody()
	{
		nEnvType = 0;
		nCookie = 0;
	}
};

struct QueryHKAccInfoAckBody
{	
	int		nEnvType;
	int		nCookie;

	//������ Trade_AccInfo ͬ��
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

	QueryHKAccInfoAckBody()
	{
		nEnvType = 0;
		nCookie = 0;

		nPower = 0;
		nZcjz = 0;
		nZqsz = 0;
		nXjjy = 0;
		nKqxj = 0;
		nDjzj = 0;
		nZsje = 0;

		nZgjde = 0;
		nYyjde = 0;
		nGpbzj = 0;
	}
};

struct	QueryHKAccInfo_Req
{
	ProtoHead			head;
	QueryHKAccInfoReqBody	body;
};

struct	QueryHKAccInfo_Ack
{
	ProtoHead			head;
	QueryHKAccInfoAckBody	body;
};



//////////////////////////////////////////////////////////////////////////
//��ȡ�û������ʻ���Ϣ
struct	QueryUSAccInfoReqBody
{
	int		nEnvType;
	int		nCookie;	

	QueryUSAccInfoReqBody()
	{
		nEnvType = 0;
		nCookie = 0;
	}
};

struct QueryUSAccInfoAckBody
{	
	int		nEnvType;
	int		nCookie;

	//������ Trade_AccInfo ͬ��
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

	QueryUSAccInfoAckBody()
	{
		nEnvType = 0;
		nCookie = 0;

		nPower = 0;
		nZcjz = 0;
		nZqsz = 0;
		nXjjy = 0;
		nKqxj = 0;
		nDjzj = 0;
		nZsje = 0;

		nZgjde = 0;
		nYyjde = 0;
		nGpbzj = 0;
	}
};

struct	QueryUSAccInfo_Req
{
	ProtoHead			head;
	QueryUSAccInfoReqBody	body;
};

struct	QueryUSAccInfo_Ack
{
	ProtoHead			head;
	QueryUSAccInfoAckBody	body;
};


//////////////////////////////////////////////////////////////////////////
//��ѯ���и۹ɶ���
struct	QueryHKOrderReqBody
{
	int		nEnvType;
	int		nCookie;
	std::string strStatusFilter; //״̬�����ַ����� ��","�ŷָ�����"0,1,2"

	QueryHKOrderReqBody()
	{
		nEnvType = 0;
		nCookie = 0;
	}
};

//�� Trade_OrderItem ͬ��
struct QueryHKOrderAckItem
{
	INT64 nLocalID; //�ͻ��˲����Ķ���ID���Ƕ���������ID�����ڹ���
	INT64 nOrderID; //�����ţ������������Ķ���������ID

	int nOrderType; //��ͬ�г���ȡֵ��Ӧ�����ö�ٶ��� Trade_OrderType_HK �� Trade_OrderType_US
	int/*Trade_OrderSide*/ enSide;
	int nStatus; //ȡֵ��Ӧ�����ö�ٶ���Trade_OrderStatus
	std::wstring strStockCode;
	std::wstring strStockName;	
	INT64 nPrice;
	INT64 nQty;
	INT64 nDealtQty; //�ɽ�����
	int nDealtAvgPrice; //�ɽ����ۣ�û�зŴ�

	INT64 nSubmitedTime; //�������յ��Ķ����ύʱ��
	INT64 nUpdatedTime; //���������µ�ʱ��

	int   nErrCode; //�����룬��֧�ָ۹�

	QueryHKOrderAckItem()
	{
		nLocalID = 0;
		nOrderID = 0;

		nOrderType = 0;
		nStatus = 0;
		nPrice = 0;
		nQty = 0;
		nDealtQty = 0;
		nDealtAvgPrice = 0;

		nSubmitedTime = 0;
		nUpdatedTime = 0;

		nErrCode = 0;
	}
};

typedef std::vector<QueryHKOrderAckItem>	VT_HK_ORDER;

struct QueryHKOrderAckBody
{	
	int		nEnvType;
	int		nCookie;
	VT_HK_ORDER vtOrder;

	QueryHKOrderAckBody()
	{
		nEnvType = 0;
		nCookie = 0;
	}
};

struct	QueryHKOrder_Req
{
	ProtoHead			head;
	QueryHKOrderReqBody	body;
};

struct	QueryHKOrder_Ack
{
	ProtoHead			head;
	QueryHKOrderAckBody	body;
};

//////////////////////////////////////////////////////////////////////////
//��ѯ�������ɶ���
struct	QueryUSOrderReqBody
{
	int		nEnvType;
	int		nCookie;
	std::string strStatusFilter; //״̬�����ַ����� ��","�ŷָ�����"0,1,2"

	QueryUSOrderReqBody()
	{
		nEnvType = 0;
		nCookie = 0;
	}
};

//�� Trade_OrderItem ͬ��_
struct QueryUSOrderAckItem
{
	INT64 nLocalID; //�ͻ��˲����Ķ���ID���Ƕ���������ID�����ڹ���
	INT64 nOrderID; //�����ţ������������Ķ���������ID

	int nOrderType; //��ͬ�г���ȡֵ��Ӧ�����ö�ٶ��� Trade_OrderType_US �� Trade_OrderType_US
	int/*Trade_OrderSide*/ enSide;
	int nStatus; //ȡֵ��Ӧ�����ö�ٶ���Trade_OrderStatus
	std::wstring strStockCode;
	std::wstring strStockName;	
	INT64 nPrice;
	INT64 nQty;
	INT64 nDealtQty; //�ɽ�����
	int   nDealtAvgPrice; //�ɽ����ۣ�û�зŴ�

	INT64 nSubmitedTime; //�������յ��Ķ����ύʱ��
	INT64 nUpdatedTime; //���������µ�ʱ��

	int   nErrCode; //�����룬��֧������

	QueryUSOrderAckItem()
	{
		nLocalID = 0;
		nOrderID = 0;

		nOrderType = 0;
		nStatus = 0;
		nPrice = 0;
		nQty = 0;
		nDealtQty = 0;
		nDealtAvgPrice = 0;

		nSubmitedTime = 0;
		nUpdatedTime = 0;

		nErrCode = 0;
	}
};

typedef std::vector<QueryUSOrderAckItem>	VT_US_ORDER;

struct QueryUSOrderAckBody
{	
	int		nEnvType;
	int		nCookie;
	VT_US_ORDER vtOrder;

	QueryUSOrderAckBody()
	{
		nEnvType = 0;
		nCookie = 0;
	}
};

struct	QueryUSOrder_Req
{
	ProtoHead			head;
	QueryUSOrderReqBody	body;
};

struct	QueryUSOrder_Ack
{
	ProtoHead			head;
	QueryUSOrderAckBody	body;
};


//////////////////////////////////////////////////////////////////////////
//��ѯ�����б�
struct	QueryPositionReqBody
{
	int		nEnvType;
	int		nCookie;

	QueryPositionReqBody()
	{
		nEnvType = 0;
		nCookie = 0;
	}
};

//�� Trade_PositionItem ͬ��_
struct QueryPositionAckItem
{
	std::wstring strStockCode;
	std::wstring strStockName;	

	INT64 nQty; //��������
	INT64 nCanSellQty; //��������
	INT64 nNominalPrice; //�м�
	INT64 nMarketVal; //��ֵ

	int  nCostPrice; //�ɱ���
	int  nCostPriceValid; //�ɱ����Ƿ���Ч
	INT64 nPLVal; //ӯ�����
	int  nPLValValid; //ӯ������Ƿ���Ч
	int nPLRatio; //ӯ������
	int nPLRatioValid; //ӯ�������Ƿ���Ч

	INT64 nToday_PLVal; //����ӯ�����
	INT64 nToday_BuyQty; //��������ɽ���
	INT64 nToday_BuyVal; //��������ɽ���
	INT64 nToday_SellQty; //���������ɽ���
	INT64 nToday_SellVal; //���������ɽ���

	QueryPositionAckItem()
	{
		nQty = 0;
		nCanSellQty = 0;
		nNominalPrice = 0;
		nMarketVal = 0;

		nCostPrice = 0;
		nCostPriceValid = 0;
		nPLVal = 0;
		nPLValValid = 0;
		nPLRatio = 0;
		nPLRatioValid = 0;

		nToday_PLVal = 0;
		nToday_BuyQty = 0;
		nToday_BuyVal = 0;
		nToday_SellQty = 0;
		nToday_SellVal = 0;
	}
};

typedef std::vector<QueryPositionAckItem>	VT_Position;

struct QueryPositionAckBody
{	
	int		nEnvType;
	int		nCookie;
	VT_Position  vtPosition;

	QueryPositionAckBody()
	{
		nEnvType = 0;
		nCookie = 0;
	}
};

struct QueryPosition_Req
{
	ProtoHead				head;
	QueryPositionReqBody	body;
};

struct QueryPosition_Ack
{
	ProtoHead				head;
	QueryPositionAckBody	body;
};
//////////////////////////////////////////////////////////////////////////
//��ѯ���и۹ɳɽ���¼
struct QueryHKDealReqBody
{
	int		nEnvType;
	int		nCookie;

	QueryHKDealReqBody()
	{
		nEnvType = 0;
		nCookie = 0;
	}
};

//�� Trade_DealItem ͬ��
struct QueryHKDealAckItem
{
	//�ر����ѣ�������������������������������������������������������������
	//����API�м۸񡢽�����������Ϊ�����ͣ�����ԭʼ����û�б��Ŵ��������ͣ����Ǹ���ֵ��1000������С��λ��0.001Ԫ

	UINT64 nOrderID; //�����ţ������������Ķ���������ID
	UINT64 nDealID; //�ɽ���

	int enSide; //����

	std::wstring strStockCode;
	std::wstring strStockName;	
	UINT64 nPrice; //�ɽ��۸�
	UINT64 nQty; //�ɽ�����

	UINT64 nTime;	//�ɽ�ʱ��

	QueryHKDealAckItem()
	{
		nOrderID = 0;
		nDealID = 0;

		enSide = 0;
		nPrice = 0;
		nQty = 0;
		nTime = 0;
	}
};

typedef std::vector<QueryHKDealAckItem>	VT_HK_Deal;

struct QueryHKDealAckBody
{	
	int		nEnvType;
	int		nCookie;
	VT_HK_Deal vtDeal;

	QueryHKDealAckBody()
	{
		nEnvType = 0;
		nCookie = 0;
	}
};

struct QueryHKDeal_Req
{
	ProtoHead			head;
	QueryHKDealReqBody	body;
};

struct QueryHKDeal_Ack
{
	ProtoHead			head;
	QueryHKDealAckBody	body;
};

//////////////////////////////////////////////////////////////////////////
//��ѯ�������ɳɽ���¼
struct QueryUSDealReqBody
{
	int		nEnvType;
	int		nCookie;

	QueryUSDealReqBody()
	{
		nEnvType = 0;
		nCookie = 0;
	}
};

//�� Trade_DealItem ͬ��
struct QueryUSDealAckItem
{
	//�ر����ѣ�������������������������������������������������������������
	//����API�м۸񡢽�����������Ϊ�����ͣ�����ԭʼ����û�б��Ŵ��������ͣ����Ǹ���ֵ��1000������С��λ��0.001Ԫ

	UINT64 nOrderID; //�����ţ������������Ķ���������ID
	UINT64 nDealID; //�ɽ���

	int enSide; //����

	std::wstring strStockCode;
	std::wstring strStockName;	

	UINT64 nPrice; //�ɽ��۸�
	UINT64 nQty; //�ɽ�����

	UINT64 nTime;	//�ɽ�ʱ��

	QueryUSDealAckItem()
	{
		nOrderID = 0;
		nDealID = 0;

		enSide = 0;

		nPrice = 0;
		nQty = 0;

		nTime = 0;
	}
};

typedef std::vector<QueryUSDealAckItem>	VT_US_Deal;

struct QueryUSDealAckBody
{	
	int		nEnvType;
	int		nCookie;
	VT_US_Deal vtDeal;

	QueryUSDealAckBody()
	{
		nEnvType = 0;
		nCookie = 0;
	}
};

struct QueryUSDeal_Req
{
	ProtoHead			head;
	QueryUSDealReqBody	body;
};

struct QueryUSDeal_Ack
{
	ProtoHead			head;
	QueryUSDealAckBody	body;
};

//��֤
struct CheckSecNumReqBody
{
	int	nCookie;

	CheckSecNumReqBody()
	{
		nCookie = 0;
	}
};

struct CheckSecNumAckBody
{
	int	nCookie;
	int nSvrResult;	

	CheckSecNumAckBody()
	{
		nCookie = 0;
		nSvrResult = 0;
	}
};

struct CheckSecNum_Req
{
	ProtoHead				head;
	CheckSecNumReqBody		body;
};

struct CheckSecNum_Ack
{
	ProtoHead				head;
	CheckSecNumAckBody		body;
};