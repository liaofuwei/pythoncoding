#pragma once
#include "Include/FTPluginCore.h"
#include "Include/FTPluginQuoteInterface.h"
#include "Include/FTPluginTradeInterface.h"
#include "Protocol/ProtoDataStruct_Trade.h"
#include "TimerWnd.h"
#include "MsgHandler.h"
#include "JsonCpp/json.h"
#include "DelayOrderIDCvt_US.h" 

class CPluginUSTradeServer;

class CPluginSetOrderStatus_US : public CTimerWndInterface,
	public CMsgHandlerEventInterface, public IOrderIDCvtNotify_US
{
public:
	CPluginSetOrderStatus_US();
	virtual ~CPluginSetOrderStatus_US();

	void Init(CPluginUSTradeServer* pTradeServer, ITrade_US*  pTradeOp);
	void Uninit();	
	void SetTradeReqData(int nCmdID, const Json::Value &jsnVal, SOCKET sock);
	void NotifyOnSetOrderStatus(Trade_Env enEnv, UINT nCookie, Trade_SvrResult enSvrRet, UINT64 nOrderID, INT64 nErrCode);

	void NotifySocketClosed(SOCKET sock);

protected:
	//CTimerWndInterface 
	virtual void OnTimeEvent(UINT nEventID);

	//CMsgHandlerEventInterface
	virtual void OnMsgEvent(int nEvent,WPARAM wParam,LPARAM lParam);

protected:
	virtual void OnCvtOrderID_Local2Svr(int nResult, Trade_Env eEnv, INT64 nLocalID, INT64 nServerID); 

protected:
	//tomodify 1
	typedef SetOrderStatus_Req	TradeReqType;
	typedef SetOrderStatus_Ack	TradeAckType;

	struct	StockDataReq
	{
		SOCKET	sock;
		DWORD	dwReqTick;
		DWORD	dwLocalCookie;
		TradeReqType req;

		bool   bWaitDelaySvrID; //�ȴ�svr����ID
	};

	typedef std::vector<StockDataReq*>		VT_REQ_TRADE_DATA;	


protected:	
	void HandleTimeoutReq();
	void HandleTradeAck(TradeAckType *pAck, SOCKET	sock);
	void SetTimerHandleTimeout(bool bStartOrStop);
	void ClearAllReqAckData();

private:
	bool	IsReqDataExist(StockDataReq* pReq); 
	void	DoRemoveReqData(StockDataReq* pReq);
	void	DoTryProcessTradeOpt(StockDataReq* pReq);

private:
	void DoClearReqInfo(SOCKET socket);
	bool IsNewStateNotNeedReq(Trade_Env eEnv, INT64 nSvrOrderID, Trade_SetOrderStatus eNewStatus);

protected:
	CPluginUSTradeServer	*m_pTradeServer;
	ITrade_US				*m_pTradeOp;	
	BOOL					m_bStartTimerHandleTimeout;

	CTimerMsgWndEx		m_TimerWnd;
	CMsgHandler			m_MsgHandler;

	VT_REQ_TRADE_DATA	m_vtReqData;

	CDelayOrderIDCvt_US  m_stOrderIDCvt; 

};