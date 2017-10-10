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
};

struct PlaceOrderAckBody
{	
	int nEnvType;
	int nCookie;
	INT64 nLocalID;
	int nSvrResult;	
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

};

struct SetOrderStatusAckBody
{	
	int		nEnvType;
	int		nCookie;
	INT64	nSvrOrderID;
	INT64	nLocalOrderID;
	int		nSvrResult;	
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
};

struct UnlockTradeAckBody
{
	int	nCookie;
	int nSvrResult;	
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
};

struct ChangeOrderAckBody
{	
	int		nEnvType;
	int		nCookie;
	INT64	nSvrOrderID;
	INT64	nLocalOrderID;
	int		nSvrResult;	
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
};

typedef std::vector<QueryHKOrderAckItem>	VT_HK_ORDER;

struct QueryHKOrderAckBody
{	
	int		nEnvType;
	int		nCookie;
	VT_HK_ORDER vtOrder;
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
};

typedef std::vector<QueryUSOrderAckItem>	VT_US_ORDER;

struct QueryUSOrderAckBody
{	
	int		nEnvType;
	int		nCookie;
	VT_US_ORDER vtOrder;
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
};

typedef std::vector<QueryPositionAckItem>	VT_Position;

struct QueryPositionAckBody
{	
	int		nEnvType;
	int		nCookie;
	VT_Position  vtPosition;
};

struct	QueryPosition_Req
{
	ProtoHead				head;
	QueryPositionReqBody	body;
};

struct	QueryPosition_Ack
{
	ProtoHead				head;
	QueryPositionAckBody	body;
};
