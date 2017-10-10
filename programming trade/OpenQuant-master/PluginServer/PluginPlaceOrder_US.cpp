#include "stdafx.h"
#include "PluginPlaceOrder_US.h"
#include "PluginUSTradeServer.h"
#include "Protocol/ProtoPlaceOrder.h"
#include "IManage_SecurityNum.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

#define TIMER_ID_HANDLE_TIMEOUT_REQ	355
#define EVENT_ID_ACK_REQUEST		368

//tomodify 2
#define PROTO_ID_QUOTE		PROTO_ID_TDUS_PLACE_ORDER
typedef CProtoPlaceOrder	CProtoQuote;



//////////////////////////////////////////////////////////////////////////

CPluginPlaceOrder_US::CPluginPlaceOrder_US()
{	
	m_pTradeOp = NULL;
	m_pTradeServer = NULL;
	m_bStartTimerHandleTimeout = FALSE;
}

CPluginPlaceOrder_US::~CPluginPlaceOrder_US()
{
	Uninit();
}

void CPluginPlaceOrder_US::Init(CPluginUSTradeServer* pTradeServer, ITrade_US*  pTradeOp)
{
	if ( m_pTradeServer != NULL )
		return;

	if ( pTradeServer == NULL || pTradeOp == NULL )
	{
		ASSERT(false);
		return;
	}

	m_pTradeServer = pTradeServer;
	m_pTradeOp = pTradeOp;
	m_TimerWnd.SetEventInterface(this);
	m_TimerWnd.Create();

	m_MsgHandler.SetEventInterface(this);
	m_MsgHandler.Create();

	m_stOrderIDCvt.Init(m_pTradeOp, this);
}

void CPluginPlaceOrder_US::Uninit()
{
	if ( m_pTradeServer != NULL )
	{
		m_pTradeServer = NULL;
		m_pTradeOp = NULL;

		m_TimerWnd.Destroy();
		m_TimerWnd.SetEventInterface(NULL);

		m_MsgHandler.Close();
		m_MsgHandler.SetEventInterface(NULL);

		ClearAllReqAckData();
	}
}

void CPluginPlaceOrder_US::SetTradeReqData(int nCmdID, const Json::Value &jsnVal, SOCKET sock)
{
	CHECK_RET(nCmdID == PROTO_ID_QUOTE && sock != INVALID_SOCKET, NORET);
	CHECK_RET(m_pTradeOp && m_pTradeServer, NORET);
	
	CProtoQuote proto;
	CProtoQuote::ProtoReqDataType	req;
	proto.SetProtoData_Req(&req);
	if ( !proto.ParseJson_Req(jsnVal) )
	{
		CHECK_OP(false, NORET);
		TradeAckType ack;
		ack.head = req.head;
		ack.head.ddwErrCode = PROTO_ERR_PARAM_ERR;
		CA::Unicode2UTF(L"��������", ack.head.strErrDesc);
		ack.body.nCookie = req.body.nCookie;
		ack.body.nSvrResult = Trade_SvrResult_Failed;
		ack.body.nEnvType = req.body.nEnvType;
		HandleTradeAck(&ack, sock);
		return;
	}

	if (!IManage_SecurityNum::IsSafeSocket(sock))
	{
		CHECK_OP(false, NORET);
		TradeAckType ack;
		ack.head = req.head;
		ack.head.ddwErrCode = PROTO_ERR_UNKNOWN_ERROR;
		CA::Unicode2UTF(L"�����½�����", ack.head.strErrDesc);
		ack.body.nCookie = req.body.nCookie;
		ack.body.nSvrResult = Trade_SvrResult_Failed;
		ack.body.nEnvType = req.body.nEnvType;
		HandleTradeAck(&ack, sock);
		return;
	}

	CHECK_RET(req.head.nProtoID == nCmdID && req.body.nCookie, NORET);

	StockDataReq *pReq = new StockDataReq;
	CHECK_RET(pReq, NORET);
	pReq->sock = sock;
	pReq->dwReqTick = ::GetTickCount();
	pReq->req = req;	

	//tomodify 3
	PlaceOrderReqBody &body = req.body;
	std::wstring strCode;
	CA::UTF2Unicode(body.strCode.c_str(), strCode);

	bool bRet = false;
	int nReqResult = 0;
	//Ŀǰֻ֧������ʵ��
	if (pReq->req.body.nEnvType == Trade_Env_Real)
	{
		bRet = m_pTradeOp->PlaceOrder((UINT*)&pReq->dwLocalCookie, (Trade_OrderType_US)body.nOrderType, 
			(Trade_OrderSide)body.nOrderDir, strCode.c_str(), body.nPrice, body.nQty, &nReqResult);
	} 

	if ( !bRet )
	{
		TradeAckType ack;
		ack.head = req.head;
		ack.head.ddwErrCode = UtilPlugin::ConvertErrCode((QueryDataErrCode)nReqResult);
		ack.head.strErrDesc = UtilPlugin::GetErrStrByCode((QueryDataErrCode)nReqResult);

		ack.body.nEnvType = body.nEnvType;
		ack.body.nCookie = body.nCookie;
		ack.body.nLocalID = 0;
		ack.body.nSvrResult = Trade_SvrResult_Failed;
		HandleTradeAck(&ack, sock);

		delete pReq;
		pReq = NULL;
		return ;
	}

	m_vtReqData.push_back(pReq);
	SetTimerHandleTimeout(true);
}
void CPluginPlaceOrder_US::NotifyOnPlaceOrder(Trade_Env enEnv, UINT32 nCookie, Trade_SvrResult enSvrRet, UINT64 nLocalID, INT64 nErrCode)
{
	CHECK_RET(nCookie, NORET);
	CHECK_RET(m_pTradeOp && m_pTradeServer, NORET);

	VT_REQ_TRADE_DATA::iterator itReq = m_vtReqData.begin();
	StockDataReq *pFindReq = NULL;
	for ( ; itReq != m_vtReqData.end(); ++itReq )
	{
		StockDataReq *pReq = *itReq;
		CHECK_OP(pReq, continue);
		if ( pReq->dwLocalCookie == nCookie )
		{
			pFindReq = pReq;
			break;
		}
	}
 
	if (!pFindReq)
		return;

	TradeAckType ack;
	ack.head = pFindReq->req.head;
	ack.head.ddwErrCode = nErrCode;
	if (nErrCode != 0 || enSvrRet != Trade_SvrResult_Succeed)
	{
		WCHAR szErr[256] = L"��������ʧ��!";
		if (nErrCode != 0)
			m_pTradeOp->GetErrDescV2(nErrCode, szErr);

		CA::Unicode2UTF(szErr, ack.head.strErrDesc);
	}

	//tomodify 4
	ack.body.nEnvType = enEnv;
	ack.body.nCookie = pFindReq->req.body.nCookie;
	ack.body.nLocalID = nLocalID;
	ack.body.nSvrResult = enSvrRet;
	ack.body.nSvrOrderID = m_stOrderIDCvt.FindSvrOrderID(enEnv, nLocalID, true);
	
	//�ɹ��ˣ� ����svrid ��û�õ��������ȴ�
	if (Trade_SvrResult_Succeed == enSvrRet && 0 == nErrCode && 0 == ack.body.nSvrOrderID)
	{
		pFindReq->bWaitSvrIDAfterPlacedOK = true;
	}
	else
	{
		HandleTradeAck(&ack, pFindReq->sock);

		m_vtReqData.erase(itReq);
		delete pFindReq;
	}
}

void CPluginPlaceOrder_US::NotifySocketClosed(SOCKET sock)
{
	DoClearReqInfo(sock);
}

void CPluginPlaceOrder_US::OnTimeEvent(UINT nEventID)
{
	if ( TIMER_ID_HANDLE_TIMEOUT_REQ == nEventID )
	{
		HandleTimeoutReq();
	}
}

void CPluginPlaceOrder_US::OnMsgEvent(int nEvent,WPARAM wParam,LPARAM lParam)
{
	if ( EVENT_ID_ACK_REQUEST == nEvent )
	{		
	}	
}

void CPluginPlaceOrder_US::OnCvtOrderID_Local2Svr(int nResult, Trade_Env eEnv, INT64 nLocalID, INT64 nServerID)
{
	VT_REQ_TRADE_DATA::iterator it_req = m_vtReqData.begin();
	for (; it_req != m_vtReqData.end();)
	{
		StockDataReq *pReq = *it_req;
		if (pReq == NULL)
		{
			CHECK_OP(false, NOOP);
			++it_req;
			continue;
		}
		if (!pReq->bWaitSvrIDAfterPlacedOK)
		{
			++it_req;
			continue;
		}

		TradeAckType ack;
		ack.head = pReq->req.head;
		ack.head.ddwErrCode = 0;
		ack.body.nEnvType = pReq->req.body.nEnvType;
		ack.body.nCookie = pReq->req.body.nCookie;
		ack.body.nLocalID = nLocalID;
		ack.body.nSvrOrderID = nServerID;
		ack.body.nSvrResult = 0;

		HandleTradeAck(&ack, pReq->sock);
		it_req = m_vtReqData.erase(it_req);
		delete pReq;
	}
}

void CPluginPlaceOrder_US::HandleTimeoutReq()
{
	if ( m_vtReqData.empty() )
	{
		SetTimerHandleTimeout(false);
		return;
	}

	DWORD dwTickNow = ::GetTickCount();	
	VT_REQ_TRADE_DATA::iterator it_req = m_vtReqData.begin();
	for ( ; it_req != m_vtReqData.end(); )
	{
		StockDataReq *pReq = *it_req;	
		if ( pReq == NULL )
		{
			CHECK_OP(false, NOOP);
			++it_req;
			continue;
		}		

		if ( int(dwTickNow - pReq->dwReqTick) > 8000 )
		{
			TradeAckType ack;
			ack.head = pReq->req.head;
			ack.head.ddwErrCode= PROTO_ERR_SERVER_TIMEROUT;
			CA::Unicode2UTF(L"Э�鳬ʱ", ack.head.strErrDesc);

			//tomodify 
			ack.body.nEnvType = pReq->req.body.nEnvType;
			ack.body.nCookie = pReq->req.body.nCookie;
			ack.body.nSvrResult = Trade_SvrResult_Failed;
			ack.body.nLocalID = 0;
			HandleTradeAck(&ack, pReq->sock);
			
			it_req = m_vtReqData.erase(it_req);
			delete pReq;
		}
		else
		{
			++it_req;
		}
	}

	if ( m_vtReqData.empty() )
	{
		SetTimerHandleTimeout(false);
		return;
	}
}

void CPluginPlaceOrder_US::HandleTradeAck(TradeAckType *pAck, SOCKET sock)
{
	CHECK_RET(pAck && pAck->body.nCookie && sock != INVALID_SOCKET, NORET);
	CHECK_RET(m_pTradeServer, NORET);

	CProtoQuote proto;
	proto.SetProtoData_Ack(pAck);

	Json::Value jsnValue;
	bool bRet = proto.MakeJson_Ack(jsnValue);
	CHECK_RET(bRet, NORET);
	
	std::string strBuf;
	CProtoParseBase::ConvJson2String(jsnValue, strBuf, true);
	m_pTradeServer->ReplyTradeReq(PROTO_ID_QUOTE, strBuf.c_str(), (int)strBuf.size(), sock);
}

void CPluginPlaceOrder_US::SetTimerHandleTimeout(bool bStartOrStop)
{
	if ( m_bStartTimerHandleTimeout )
	{
		if ( !bStartOrStop )
		{			
			m_TimerWnd.StopTimer(TIMER_ID_HANDLE_TIMEOUT_REQ);
			m_bStartTimerHandleTimeout = FALSE;
		}
	}
	else
	{
		if ( bStartOrStop )
		{
			m_TimerWnd.StartMillionTimer(500, TIMER_ID_HANDLE_TIMEOUT_REQ);
			m_bStartTimerHandleTimeout = TRUE;
		}
	}
}

void CPluginPlaceOrder_US::ClearAllReqAckData()
{
	VT_REQ_TRADE_DATA::iterator it_req = m_vtReqData.begin();
	for ( ; it_req != m_vtReqData.end(); )
	{
		StockDataReq *pReq = *it_req;
		delete pReq;
	}

	m_vtReqData.clear();
}

void CPluginPlaceOrder_US::DoClearReqInfo(SOCKET socket)
{
	VT_REQ_TRADE_DATA& vtReq = m_vtReqData;

	//���socket��Ӧ��������Ϣ
	auto itReq = vtReq.begin();
	while (itReq != vtReq.end())
	{
		if (*itReq && (*itReq)->sock == socket)
		{
			delete *itReq;
			itReq = vtReq.erase(itReq);
		}
		else
		{
			++itReq;
		}
	}
}
