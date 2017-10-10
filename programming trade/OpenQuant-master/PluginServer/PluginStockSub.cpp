#include "stdafx.h"
#include "PluginStockSub.h"
#include "PluginQuoteServer.h"
#include "Protocol/ProtoStockSub.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

#define TIMER_ID_CLEAR_CACHE		354
#define TIMER_ID_HANDLE_TIMEOUT_REQ	355

#define EVENT_ID_ACK_REQUEST	368

//tomodify 2
#define PROTO_ID_QUOTE		PROTO_ID_QT_SUBSTOCK
#define QUOTE_SERVER_TYPE	QuoteServer_StockSub
typedef CProtoStockSub	CProtoQuote;



//////////////////////////////////////////////////////////////////////////

CPluginStockSub::CPluginStockSub()
{	
	m_pQuoteData = NULL;
	m_pQuoteServer = NULL;

	m_bStartTimerClearCache = FALSE;
	m_bStartTimerHandleTimeout = FALSE;
}

CPluginStockSub::~CPluginStockSub()
{
	Uninit();
}

void CPluginStockSub::Init(CPluginQuoteServer* pQuoteServer, IFTQuoteData*  pQuoteData)
{
	if ( m_pQuoteServer != NULL )
		return;

	if ( pQuoteServer == NULL || pQuoteData == NULL )
	{
		ASSERT(false);
		return;
	}

	m_pQuoteServer = pQuoteServer;
	m_pQuoteData = pQuoteData;
	m_TimerWnd.SetEventInterface(this);
	m_TimerWnd.Create();

	m_MsgHandler.SetEventInterface(this);
	m_MsgHandler.Create();
}

void CPluginStockSub::Uninit()
{
	if ( m_pQuoteServer != NULL )
	{
		m_pQuoteServer = NULL;
		m_pQuoteData = NULL;
		m_TimerWnd.Destroy();
		m_TimerWnd.SetEventInterface(NULL);

		m_MsgHandler.Close();
		m_MsgHandler.SetEventInterface(NULL);

		ClearAllReqCache();
	}
}

void CPluginStockSub::SetQuoteReqData(int nCmdID, const Json::Value &jsnVal, SOCKET sock)
{
	CHECK_RET(nCmdID == PROTO_ID_QUOTE && sock != INVALID_SOCKET, NORET);
	CHECK_RET(m_pQuoteData && m_pQuoteServer, NORET);

	CProtoQuote proto;
	CProtoQuote::ProtoReqDataType	req;
	proto.SetProtoData_Req(&req);
	if ( !proto.ParseJson_Req(jsnVal) )
	{
		CHECK_OP(false, NORET);
		StockDataReq req_info;
		req_info.sock = sock;
		req_info.req = req;
		req_info.dwReqTick = ::GetTickCount();
		ReplyDataReqError(&req_info, PROTO_ERR_PARAM_ERR, L"��������");
		return;
	}

	CHECK_RET(req.head.nProtoID == nCmdID, NORET);

	INT64 nStockID = IFTStockUtil::GetStockHashVal(req.body.strStockCode.c_str(), (StockMktType)req.body.nStockMarket);
	if ( nStockID == 0 )
	{
		StockDataReq req_info;
		req_info.nStockID = nStockID;
		req_info.sock = sock;
		req_info.req = req;
		req_info.dwReqTick = ::GetTickCount();
		ReplyDataReqError(&req_info, PROTO_ERR_STOCK_NOT_FIND, L"�Ҳ�����Ʊ��");
		return;
	}	

	//////////�������ƣ�������

	StockDataReq *pReqInfo = new StockDataReq;
	CHECK_RET(pReqInfo, NORET);
	pReqInfo->nStockID = nStockID;
	pReqInfo->sock = sock;
	pReqInfo->req = req;
	pReqInfo->dwReqTick = ::GetTickCount();

	VT_STOCK_DATA_REQ &vtReq = m_mapReqInfo[std::make_pair(nStockID, req.body.nStockSubType)];
	bool bNeedSub = vtReq.empty();	
	vtReq.push_back(pReqInfo);

	if ( bNeedSub )
	{
		StockSubErrCode SubResult = m_pQuoteServer->SubscribeQuote(req.body.strStockCode, (StockMktType)req.body.nStockMarket, (StockSubType)req.body.nStockSubType, true, sock);
		if ( SubResult == StockSub_FailMaxSubNum )
		{
			////��vtReq�е�ÿһ��
			for (size_t i = 0; i < vtReq.size(); i++)
			{
				StockDataReq *pReqAnswer = vtReq[i];
				ReplyDataReqError(pReqInfo, PROTO_ERR_MAXSUB_ERR, L"�Ѵﵽ�������ޣ�");
			}
			MAP_STOCK_DATA_REQ::iterator it_iterator = m_mapReqInfo.find(std::make_pair(nStockID, req.body.nStockSubType));
			if ( it_iterator != m_mapReqInfo.end() )
			{
				it_iterator = m_mapReqInfo.erase(it_iterator);
			}
			return;
		}
		else if ( SubResult == StockSub_Suc )
		{
			DWORD dwLocalCookie = 0;
			switch((StockSubType)req.body.nStockSubType)
			{
			case StockSubType_RT:
				{
					QueryDataErrCode err_code = m_pQuoteServer->QueryStockRTData((DWORD*)&dwLocalCookie, pReqInfo->req.body.strStockCode, (StockMktType)pReqInfo->req.body.nStockMarket, QuoteServer_RTData);
				}
				break;
			case StockSubType_Ticker:
				{
				}
				break;
			case StockSubType_KL_MIN1:
				{
					QueryDataErrCode err_code = m_pQuoteServer->QueryStockKLData((DWORD*)&dwLocalCookie, pReqInfo->req.body.strStockCode, (StockMktType)pReqInfo->req.body.nStockMarket, QuoteServer_KLData, (int)1);
				}
				break;
			case StockSubType_KL_DAY:
				{
					QueryDataErrCode err_code = m_pQuoteServer->QueryStockKLData((DWORD*)&dwLocalCookie, pReqInfo->req.body.strStockCode, (StockMktType)pReqInfo->req.body.nStockMarket, QuoteServer_KLData, (int)2);
				}
				break;
			case StockSubType_KL_WEEK:
				{
					QueryDataErrCode err_code = m_pQuoteServer->QueryStockKLData((DWORD*)&dwLocalCookie, pReqInfo->req.body.strStockCode, (StockMktType)pReqInfo->req.body.nStockMarket, QuoteServer_KLData, (int)3);
				}
				break;
			case StockSubType_KL_MONTH:
				{
					QueryDataErrCode err_code = m_pQuoteServer->QueryStockKLData((DWORD*)&dwLocalCookie, pReqInfo->req.body.strStockCode, (StockMktType)pReqInfo->req.body.nStockMarket, QuoteServer_KLData, (int)4);
				}
				break;
			case StockSubType_KL_MIN5:
				{
					QueryDataErrCode err_code = m_pQuoteServer->QueryStockKLData((DWORD*)&dwLocalCookie, pReqInfo->req.body.strStockCode, (StockMktType)pReqInfo->req.body.nStockMarket, QuoteServer_KLData, (int)6);
				}
				break;
			case StockSubType_KL_MIN15:
				{
					QueryDataErrCode err_code = m_pQuoteServer->QueryStockKLData((DWORD*)&dwLocalCookie, pReqInfo->req.body.strStockCode, (StockMktType)pReqInfo->req.body.nStockMarket, QuoteServer_KLData, (int)7);
				}
				break;
			case StockSubType_KL_MIN30:
				{
					QueryDataErrCode err_code = m_pQuoteServer->QueryStockKLData((DWORD*)&dwLocalCookie, pReqInfo->req.body.strStockCode, (StockMktType)pReqInfo->req.body.nStockMarket, QuoteServer_KLData, (int)8);
				}
				break;
			case StockSubType_KL_MIN60:
				{
					QueryDataErrCode err_code = m_pQuoteServer->QueryStockKLData((DWORD*)&dwLocalCookie, pReqInfo->req.body.strStockCode, (StockMktType)pReqInfo->req.body.nStockMarket, QuoteServer_KLData, (int)9);
				}
				break;
			}
		}
		QuoteAckDataBody &ack = m_mapCacheData[std::make_pair(nStockID, req.body.nStockSubType)];
	}

	m_MsgHandler.RaiseEvent(EVENT_ID_ACK_REQUEST, 0, 0);
	SetTimerHandleTimeout(true);
}

void CPluginStockSub::NotifyQuoteDataUpdate(int nCmdID, INT64 nStockID)
{
	CHECK_RET(nCmdID == PROTO_ID_QUOTE && nStockID, NORET);
	CHECK_RET(m_pQuoteData, NORET);

	//bool bInReq = (m_mapReqInfo.find(nStockID) != m_mapReqInfo.end());
	//bool bInCache = (m_mapCacheData.find(nStockID) != m_mapCacheData.end());��Ҫ�޸�

	//������ʱ��������ԭ��������������ݵ���ǰ��ǰ������
	//if ( !bInReq && !bInCache )
	//{
	//	//CHECK_OP(false, NOOP);
	//	return;
	//}

}

void CPluginStockSub::NotifySocketClosed(SOCKET sock)
{
	DoClearReqInfo(sock);
}

void CPluginStockSub::OnTimeEvent(UINT nEventID)
{
	if ( TIMER_ID_CLEAR_CACHE == nEventID )
	{
		ClearQuoteDataCache();
	}
	else if ( TIMER_ID_HANDLE_TIMEOUT_REQ == nEventID )
	{
		HandleTimeoutReq();
	}
}

void CPluginStockSub::OnMsgEvent(int nEvent,WPARAM wParam,LPARAM lParam)
{
	if ( EVENT_ID_ACK_REQUEST == nEvent )
	{
		ReplyAllReadyReq();
	}	
}

void CPluginStockSub::ClearQuoteDataCache()
{
	if ( m_mapCacheToDel.empty() )
	{
		SetTimerClearCache(false);
		return ;
	}

	DWORD dwTickNow = ::GetTickCount();

	MAP_CACHE_TO_DESTROY::iterator it_todel = m_mapCacheToDel.begin();
	for ( ; it_todel != m_mapCacheToDel.end(); )
	{
		INT64 nStockID = it_todel->first.first;
		int nStockSubType = it_todel->first.second;
		DWORD dwToDelTick = it_todel->second;

		MAP_STOCK_DATA_REQ::iterator it_req = m_mapReqInfo.find(std::make_pair(nStockID, nStockSubType));
		if ( it_req != m_mapReqInfo.end() )
		{
			it_todel = m_mapCacheToDel.erase(it_todel);
		}
		else
		{
			if ( int(dwTickNow - dwToDelTick) > 60*1000  )
			{
				m_mapCacheData.erase(std::make_pair(nStockID, nStockSubType));
				it_todel = m_mapCacheToDel.erase(it_todel);

				StockMktCodeEx stkMktCode;
				if ( m_pQuoteServer && IFTStockUtil::GetStockMktCode(nStockID, stkMktCode) )
				{				
					//m_pQuoteServer->SubscribeQuote(stkMktCode.strCode, (StockMktType)stkMktCode.nMarketType, QUOTE_SERVER_TYPE, false);					
				}
				else
				{
					CHECK_OP(false, NOOP);
				}
			}
			else
			{
				++it_todel;
			}
		}
	}

	if ( m_mapCacheToDel.empty() )
	{
		SetTimerClearCache(false);		
	}
}

void CPluginStockSub::HandleTimeoutReq()
{
	if ( m_mapReqInfo.empty() )
	{
		SetTimerHandleTimeout(false);
		return;
	}

	ReplyAllReadyReq();

	DWORD dwTickNow = ::GetTickCount();	
	MAP_STOCK_DATA_REQ::iterator it_stock = m_mapReqInfo.begin();
	for ( ; it_stock != m_mapReqInfo.end(); )
	{
		INT64 nStockID = it_stock->first.first;
		int nStockSubType = it_stock->first.second;
		VT_STOCK_DATA_REQ &vtReq = it_stock->second;
		VT_STOCK_DATA_REQ::iterator it_req = vtReq.begin();

		for ( ; it_req != vtReq.end(); )
		{
			StockDataReq *pReq = *it_req;
			if ( pReq == NULL )
			{
				CHECK_OP(false, NOOP);
				it_req = vtReq.erase(it_req);
				continue;
			}

			if (int(dwTickNow - pReq->dwReqTick) > REQ_TIMEOUT_MILLISECOND)
			{
				CStringA strTimeout;
				strTimeout.Format("StockSub req timeout, market=%d, code=%s", pReq->req.body.nStockMarket, pReq->req.body.strStockCode.c_str());
				OutputDebugStringA(strTimeout.GetString());				
				ReplyDataReqError(pReq, PROTO_ERR_SERVER_TIMEROUT, L"����ʱ��");
				it_req = vtReq.erase(it_req);
				delete pReq;
			}
			else
			{
				++it_req;
			}
		}

		if ( vtReq.empty() )
		{
			//���ﲻ�������建��ʱ������Ϊ��ʱû�л��浱ǰ��Ʊ������
			it_stock = m_mapReqInfo.erase(it_stock);			
		}
		else
		{
			++it_stock;
		}
	}

	if ( m_mapReqInfo.empty() )
	{
		SetTimerHandleTimeout(false);
		return;
	}
}

void CPluginStockSub::ReplyAllReadyReq()
{
	DWORD dwTickNow = ::GetTickCount();
	MAP_STOCK_DATA_REQ::iterator it_stock = m_mapReqInfo.begin();
	for ( ; it_stock != m_mapReqInfo.end(); )
	{
		INT64 nStockID = it_stock->first.first;
		int nStockSubType = it_stock->first.second;
		VT_STOCK_DATA_REQ &vtReq = it_stock->second;
		MAP_STOCK_CACHE_DATA::iterator it_data = m_mapCacheData.find(std::make_pair(nStockID, nStockSubType));

		if ( it_data == m_mapCacheData.end() )
		{
			++it_stock;
			continue;
		}

		VT_STOCK_DATA_REQ::iterator it_req = vtReq.begin();
		for ( ; it_req != vtReq.end(); ++it_req )
		{
			StockDataReq *pReq = *it_req;
			CHECK_OP(pReq, NOOP);
			ReplyStockDataReq(pReq, it_data->second);
			delete pReq;
		}

		vtReq.clear();

		it_stock = m_mapReqInfo.erase(it_stock);
		m_mapCacheToDel[std::make_pair(nStockID, nStockSubType)] = dwTickNow;
		SetTimerClearCache(true);
	}

	if ( m_mapReqInfo.empty() )
	{
		SetTimerHandleTimeout(false);
		return;
	}
}

void CPluginStockSub::ReplyStockDataReq(StockDataReq *pReq, const QuoteAckDataBody &data)
{
	CHECK_RET(pReq && m_pQuoteServer, NORET);

	CProtoQuote::ProtoAckDataType ack;
	ack.head = pReq->req.head;
	ack.head.ddwErrCode = 0;
	ack.body = data;

	//tomodify 4
	ack.body.nStockMarket = pReq->req.body.nStockMarket;
	ack.body.strStockCode = pReq->req.body.strStockCode;
	ack.body.nStockSubType = pReq->req.body.nStockSubType;


	CProtoQuote proto;	
	proto.SetProtoData_Ack(&ack);

	Json::Value jsnAck;
	if ( proto.MakeJson_Ack(jsnAck) )
	{
		std::string strOut;
		CProtoParseBase::ConvJson2String(jsnAck, strOut, true);
		m_pQuoteServer->ReplyQuoteReq(pReq->req.head.nProtoID, strOut.c_str(), (int)strOut.size(), pReq->sock);
	}
	else
	{
		CHECK_OP(false, NOOP);
	}
}

void CPluginStockSub::ReplyDataReqError(StockDataReq *pReq, int nErrCode, LPCWSTR pErrDesc)
{
	CHECK_RET(pReq && m_pQuoteServer, NORET);

	CProtoQuote::ProtoAckDataType ack;
	ack.head = pReq->req.head;
	ack.head.ddwErrCode = nErrCode;

	if ( pErrDesc )
	{
		CA::Unicode2UTF(pErrDesc, ack.head.strErrDesc);		 
	}

	CProtoQuote proto;	
	proto.SetProtoData_Ack(&ack);

	Json::Value jsnAck;
	if ( proto.MakeJson_Ack(jsnAck) )
	{
		std::string strOut;
		CProtoParseBase::ConvJson2String(jsnAck, strOut, true);
		m_pQuoteServer->ReplyQuoteReq(pReq->req.head.nProtoID, strOut.c_str(), (int)strOut.size(), pReq->sock);
	}
	else
	{
		CHECK_OP(false, NOOP);
	}

}

void CPluginStockSub::SetTimerHandleTimeout(bool bStartOrStop)
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

void CPluginStockSub::SetTimerClearCache(bool bStartOrStop)
{
	if ( m_bStartTimerClearCache )
	{
		if ( !bStartOrStop )
		{
			m_TimerWnd.StopTimer(TIMER_ID_CLEAR_CACHE);
			m_bStartTimerClearCache = FALSE;
		}
	}
	else
	{
		if ( bStartOrStop )
		{
			m_TimerWnd.StartMillionTimer(50, TIMER_ID_CLEAR_CACHE);
			m_bStartTimerClearCache = TRUE;
		}
	}
}

void CPluginStockSub::ClearAllReqCache()
{
	MAP_STOCK_DATA_REQ::iterator it_stock = m_mapReqInfo.begin();
	for ( ; it_stock != m_mapReqInfo.end(); ++it_stock )
	{
		VT_STOCK_DATA_REQ &vtReq = it_stock->second;
		VT_STOCK_DATA_REQ::iterator it_req = vtReq.begin();
		for ( ; it_req != vtReq.end(); ++it_req )
		{
			StockDataReq *pReq = *it_req;
			delete pReq;
		}
	}

	m_mapReqInfo.clear();
	m_mapCacheData.clear();
	m_mapCacheToDel.clear();
}

void CPluginStockSub::DoClearReqInfo(SOCKET socket)
{
	auto itmap = m_mapReqInfo.begin();
	while (itmap != m_mapReqInfo.end())
	{
		VT_STOCK_DATA_REQ& vtReq = itmap->second;

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
		if (vtReq.size() == 0)
		{
			itmap = m_mapReqInfo.erase(itmap);
		}
		else
		{
			++itmap;
		}
	}
}


