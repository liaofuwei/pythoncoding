#pragma once
#include "Include/FTPluginCore.h"
#include "Include/FTPluginQuoteInterface.h"
#include "Include/FTPluginTradeInterface.h"
#include "Protocol/ProtoDataStruct.h"
#include "JsonCpp/json.h"
#include "PluginPlaceOrder_HK.h"
#include "PluginChangeOrder_HK.h"
#include "PluginSetOrderStatus_HK.h"
#include "PluginUnlockTrade.h"
#include "PluginQueryHKAccInfo.h"
#include "PluginQueryHKOrder.h"
#include "PluginQueryHKPosition.h"
#include "PluginQueryHKDeal.h"
#include "IManage_SecurityNum.h"

class CPluginNetwork;

class CPluginHKTradeServer: 	
	public ITradeCallBack_HK
{
public:
	CPluginHKTradeServer();
	virtual ~CPluginHKTradeServer();
	
	void InitTradeSvr(IFTPluginCore* pPluginCore, CPluginNetwork *pNetwork);
	void UninitTradeSvr();	
	void SetTradeReqData(int nCmdID, const Json::Value &jsnVal, SOCKET sock);
	void ReplyTradeReq(int nCmdID, const char *pBuf, int nLen, SOCKET sock);	

	void CloseSocket(SOCKET sock);

protected:
	//ITradeCallBack_HK
	virtual void OnUnlockTrade(UINT32 nCookie, Trade_SvrResult enSvrRet, UINT64 nErrCode);
	virtual void OnPlaceOrder(Trade_Env enEnv, UINT nCookie, Trade_SvrResult enSvrRet,
				UINT64 nLocalID, UINT16 nErrCode);	
	virtual void OnOrderUpdate(Trade_Env enEnv, const Trade_OrderItem& orderItem);
	virtual void OnSetOrderStatus(Trade_Env enEnv, UINT nCookie, Trade_SvrResult enSvrRet, UINT64 nOrderID, UINT16 nErrCode);
	virtual void OnChangeOrder(Trade_Env enEnv, UINT nCookie, Trade_SvrResult enSvrRet, UINT64 nOrderID, UINT16 nErrCode);
	virtual void OnOrderErrNotify(Trade_Env enEnv, UINT64 nOrderID, Trade_OrderErrNotify_HK enErrNotify, UINT16 nErrCode);

	virtual void OnQueryOrderList(Trade_Env enEnv, UINT32 nCookie, INT32 nCount, const Trade_OrderItem* pArrOrder);
	virtual void OnQueryDealList(Trade_Env enEnv, UINT32 nCookie, INT32 nCount, const Trade_DealItem* pArrOrder);
	virtual void OnQueryAccInfo(Trade_Env enEnv, UINT32 nCookie, const Trade_AccInfo& accInfo, int nResult);
	virtual void OnQueryPositionList(Trade_Env enEnv, UINT32 nCookie, INT32 nCount, const Trade_PositionItem* pArrPosition);

protected:
	IFTPluginCore		*m_pPluginCore;
	ITrade_HK			*m_pTradeOp;
	IFTDataReport		*m_pDataReport;
	CPluginNetwork		*m_pNetwork;

	CPluginPlaceOrder_HK	m_PlaceOrder;
	CPluginChangeOrder_HK  m_ChangeOrder;
	CPluginSetOrderStatus_HK m_SetOrderStatus;

	CPluginUnlockTrade		m_UnlockTrade;
	CPluginQueryHKAccInfo	m_QueryAccInfo;
	CPluginQueryHKOrder		m_QueryHKOrder;

	CPluginQueryHKPosition	m_QueryHKPos;
	CPluginQueryHKDeal		m_QueryHKDeal;
};