#include "stdafx.h"
#include "PluginSetOrderStatus.h"
#include "PluginHKTradeServer.h"
#include "Protocol/ProtoSetOrderStatus.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

#define TIMER_ID_HANDLE_TIMEOUT_REQ	355
#define EVENT_ID_ACK_REQUEST		368

//tomodify 2
#define PROTO_ID_QUOTE			PROTO_ID_TDHK_SET_ORDER_STATUS
typedef CProtoSetOrderStatus	CProtoQuote;



//////////////////////////////////////////////////////////////////////////

CPluginSetOrderStatus::CPluginSetOrderStatus()
{	
	m_pTradeOp = NULL;
	m_pHKTradeServer = NULL;
	m_bStartTimerHandleTimeout = FALSE;
}

CPluginSetOrderStatus::~CPluginSetOrderStatus()
{
	Uninit();
}

void CPluginSetOrderStatus::Init(CPluginHKTradeServer* pTradeServer, ITrade_HK*  pTradeOp)
{
	if ( m_pHKTradeServer != NULL )
		return;

	if ( pTradeServer == NULL || pTradeOp == NULL )
	{
		ASSERT(false);
		return;
	}

	m_pHKTradeServer = pTradeServer;
	m_pTradeOp = pTradeOp;
	m_TimerWnd.SetEventInterface(this);
	m_TimerWnd.Create();

	m_MsgHandler.SetEventInterface(this);
	m_MsgHandler.Create();
}

void CPluginSetOrderStatus::Uninit()
{
	if ( m_pHKTradeServer != NULL )
	{
		m_pHKTradeServer = NULL;
		m_pTradeOp = NULL;

		m_TimerWnd.Destroy();
		m_TimerWnd.SetEventInterface(NULL);

		m_MsgHandler.Close();
		m_MsgHandler.SetEventInterface(NULL);

		ClearAllReqAckData();
	}
}

void CPluginSetOrderStatus::SetTradeReqData(int nCmdID, const Json::Value &jsnVal, SOCKET sock)
{
	CHECK_RET(nCmdID == PROTO_ID_QUOTE && sock != INVALID_SOCKET, NORET);
	CHECK_RET(m_pTradeOp && m_pHKTradeServer, NORET);
	
	CProtoQuote proto;
	CProtoQuote::ProtoReqDataType	req;
	proto.SetProtoData_Req(&req);
	if ( !proto.ParseJson_Req(jsnVal) )
	{
		CHECK_OP(false, NORET);
		return;
	}

	CHECK_RET(req.head.nProtoID == nCmdID && req.body.nCookie, NORET);

	StockDataReq *pReq = new StockDataReq;
	CHECK_RET(pReq, NORET);
	pReq->sock = sock;
	pReq->dwReqTick = ::GetTickCount();
	pReq->req = req;

	//tomodify 3
	SetOrderStatusHKReqBody &body = req.body;		
	bool bRet = m_pTradeOp->SetOrderStatus((Trade_Env)body.nEnvType, (UINT*)&pReq->dwLocalCookie, body.nOrderID, 
		(Trade_SetOrderStatus_HK)body.nSetOrderStatusHK);

	if ( !bRet )
	{
		TradeAckType ack;
		ack.head = req.head;
		ack.head.nErrCode = PROTO_ERR_UNKNOWN_ERROR;
		CA::Unicode2UTF(L"����ʧ��", ack.head.strErrDesc);

		ack.body.nCookie = body.nCookie;
		ack.body.nOrderID = body.nOrderID;
		ack.body.nSvrResult = Trade_SvrResult_Failed;
		HandleTradeAck(&ack, sock);

		delete pReq;
		pReq = NULL;
		return ;
	}

	m_vtReqData.push_back(pReq);
	SetTimerHandleTimeout(true);
}

void CPluginSetOrderStatus::NotifyOnSetOrderStatus(Trade_Env enEnv, UINT nCookie, Trade_SvrResult enSvrRet, UINT64 nOrderID, UINT16 nErrCode)
{
	CHECK_RET(nCookie, NORET);
	CHECK_RET(m_pTradeOp && m_pHKTradeServer, NORET);

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

	CHECK_RET(pFindReq, NORET);

	TradeAckType ack;
	ack.head = pFindReq->req.head;
	ack.head.nErrCode = nErrCode;
	if ( nErrCode )
	{
		WCHAR szErr[256] = L"";
		if ( m_pTradeOp->GetErrDesc(nErrCode, szErr) )
			CA::Unicode2UTF(szErr, ack.head.strErrDesc);
	}

	//tomodify 4
	ack.body.nEnvType = enEnv;
	ack.body.nCookie = pFindReq->req.body.nCookie;
	ack.body.nOrderID = pFindReq->req.body.nOrderID;
	ack.body.nSvrResult = enSvrRet;
	HandleTradeAck(&ack, pFindReq->sock);

	m_vtReqData.erase(itReq);
	delete pFindReq;
}

void CPluginSetOrderStatus::OnTimeEvent(UINT nEventID)
{
	if ( TIMER_ID_HANDLE_TIMEOUT_REQ == nEventID )
	{
		HandleTimeoutReq();
	}
}

void CPluginSetOrderStatus::OnMsgEvent(int nEvent,WPARAM wParam,LPARAM lParam)
{
	if ( EVENT_ID_ACK_REQUEST == nEvent )
	{		
	}	
}

void CPluginSetOrderStatus::HandleTimeoutReq()
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
			ack.head.nErrCode= PROTO_ERR_SERVER_TIMEROUT;
			CA::Unicode2UTF(L"Э�鳬ʱ", ack.head.strErrDesc);

			//tomodify 5
			ack.body.nCookie = pReq->req.body.nCookie;
			ack.body.nSvrResult = Trade_SvrResult_Failed;
			ack.body.nOrderID = pReq->req.body.nOrderID;
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

void CPluginSetOrderStatus::HandleTradeAck(TradeAckType *pAck, SOCKET sock)
{
	CHECK_RET(pAck && pAck->body.nCookie && sock != INVALID_SOCKET, NORET);
	CHECK_RET(m_pHKTradeServer, NORET);

	CProtoQuote proto;
	proto.SetProtoData_Ack(pAck);

	Json::Value jsnValue;
	bool bRet = proto.MakeJson_Ack(jsnValue);
	CHECK_RET(bRet, NORET);
	
	std::string strBuf;
	CProtoParseBase::ConvJson2String(jsnValue, strBuf, true);
	m_pHKTradeServer->ReplyTradeReq(PROTO_ID_QUOTE, strBuf.c_str(), (int)strBuf.size(), sock);
}

void CPluginSetOrderStatus::SetTimerHandleTimeout(bool bStartOrStop)
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

void CPluginSetOrderStatus::ClearAllReqAckData()
{
	VT_REQ_TRADE_DATA::iterator it_req = m_vtReqData.begin();
	for ( ; it_req != m_vtReqData.end(); )
	{
		StockDataReq *pReq = *it_req;
		delete pReq;
	}

	m_vtReqData.clear();
}