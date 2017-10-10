#include "stdafx.h"
#include "PluginTickerPrice.h"
#include "PluginQuoteServer.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

#define TIMER_ID_CLEAR_CACHE		354
#define TIMER_ID_HANDLE_TIMEOUT_REQ	355

#define EVENT_ID_ACK_REQUEST	368

//tomodify 2
#define PROTO_ID_QUOTE		PROTO_ID_QT_GET_TICKER
#define QUOTE_SERVER_TYPE	QuoteServer_TickerPrice


//////////////////////////////////////////////////////////////////////////

CPluginTickerPrice::CPluginTickerPrice()
{	
	m_pQuoteData = NULL;
	m_pQuoteServer = NULL;

	m_bStartTimerClearCache = FALSE;
	m_bStartTimerHandleTimeout = FALSE;
}

CPluginTickerPrice::~CPluginTickerPrice()
{
	Uninit();
}

void CPluginTickerPrice::Init(CPluginQuoteServer* pQuoteServer, IFTQuoteData*  pQuoteData)
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

void CPluginTickerPrice::Uninit()
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

void CPluginTickerPrice::SetQuoteReqData(int nCmdID, const Json::Value &jsnVal, SOCKET sock)
{
	CHECK_RET(nCmdID == PROTO_ID_QUOTE && sock != INVALID_SOCKET, NORET);
	CHECK_RET(m_pQuoteData && m_pQuoteServer, NORET);
	
	CProtoQuote proto;
	CProtoQuote::ProtoReqDataType	req;
	proto.SetProtoData_Req(&req);
	if ( !proto.ParseJson_Req(jsnVal) )
	{
		CHECK_OP(false, NORET);
		req.head.nProtoID = nCmdID;

		StockDataReq req_info;		
		req_info.sock = sock;
		req_info.req = req;
		req_info.dwReqTick = ::GetTickCount();
		ReplyDataReqError(&req_info, PROTO_ERR_PARAM_ERR, L"��������");
		return;
	}

	CHECK_RET(req.head.nProtoID == nCmdID && req.body.nGetTickNum >= 0, NORET);
	req.body.nGetTickNum = min(req.body.nGetTickNum, 1000);

	INT64 nStockID = IFTStockUtil::GetStockHashVal(req.body.strStockCode.c_str(), (StockMktType)req.body.nStockMarket);
	if ( nStockID == 0 )
	{
		CHECK_OP(false, NOOP);
		StockDataReq req_info;
		req_info.nStockID = nStockID;
		req_info.sock = sock;
		req_info.req = req;
		req_info.dwReqTick = ::GetTickCount();
		ReplyDataReqError(&req_info, PROTO_ERR_STOCK_NOT_FIND, L"�Ҳ�����Ʊ��");
		return;
	}	

	StockDataReq *pReqInfo = new StockDataReq;
	CHECK_RET(pReqInfo, NORET);
	pReqInfo->nStockID = nStockID;
	pReqInfo->sock = sock;
	pReqInfo->req = req;
	pReqInfo->dwReqTick = ::GetTickCount();

	VT_STOCK_DATA_REQ &vtReq = m_mapReqInfo[nStockID];	
	vtReq.push_back(pReqInfo);

	bool bIsSub = m_pQuoteData->IsSubStockOneType(nStockID, StockSubType_Ticker);
	if ( bIsSub )
	{
		//tomodify 3.1
		std::vector<PluginTickItem> vtTickPrice;
		vtTickPrice.resize(req.body.nGetTickNum);
		int nTickFillCount = m_pQuoteData->FillTickArr(nStockID, _vect2Ptr(vtTickPrice), _vectIntSize(vtTickPrice));
		if ( nTickFillCount >= 0 )
		{
			QuoteAckDataBody &ack = m_mapCacheData[nStockID];
			ack.nNextSequence = -1;
			ack.nStockMarket = req.body.nStockMarket;
			ack.strStockCode = req.body.strStockCode;

			int nValidNum = min(nTickFillCount, req.body.nGetTickNum);
			for ( int n = 0; n < nValidNum; n++ )
			{
				TickerAckItem tickItem;
				PluginTickItem &srcItem = vtTickPrice[n];					
				tickItem.nPrice = (int)srcItem.dwPrice;
				tickItem.nDealType = srcItem.nDealType;
				tickItem.nSequence = srcItem.nSequence;
				tickItem.nVolume = srcItem.nVolume;
				tickItem.nTurnover = srcItem.nTurnover;
				tickItem.strTickTime = UtilPlugin::FormatMktTimestamp((int)srcItem.dwTime, (StockMktType)ack.nStockMarket, FormatTime_YMDHMS);
				ack.vtTicker.push_back(tickItem);		
			}				
		}
	}
	else
	{
		////��vtReq�е�ÿһ��
		for (size_t i = 0; i < vtReq.size(); i++)
		{
			StockDataReq *pReqAnswer = vtReq[i];
			ReplyDataReqError(pReqAnswer, PROTO_ERR_UNSUB_ERR, L"��Ʊδ���ģ�");
		}
		MAP_STOCK_DATA_REQ::iterator it_iterator = m_mapReqInfo.find(nStockID);
		if ( it_iterator != m_mapReqInfo.end() )
		{
			it_iterator = m_mapReqInfo.erase(it_iterator);
		}
		return;
	}


	ReplyAllReadyReq();
	//m_MsgHandler.RaiseEvent(EVENT_ID_ACK_REQUEST, 0, 0);
	SetTimerHandleTimeout(true);
}

void CPluginTickerPrice::NotifyQuoteDataUpdate(int nCmdID, INT64 nStockID)
{
	CHECK_RET(nCmdID == PROTO_ID_QUOTE && nStockID, NORET);
	CHECK_RET(m_pQuoteData, NORET);
	
	bool bIsSub = m_pQuoteData->IsSubStockOneType(nStockID, StockSubType_Simple);
	if ( !bIsSub )
	{
		return;
	}

	bool bInReq = (m_mapReqInfo.find(nStockID) != m_mapReqInfo.end());
	bool bInCache = (m_mapCacheData.find(nStockID) != m_mapCacheData.end());
	
	//������ʱ��������ԭ��������������ݵ���ǰ��ǰ������
	if ( !bInReq && !bInCache )
	{
		//CHECK_OP(false, NOOP);
		return;
	}

	//tomodify 3.2
	//�ȵ�ӿڵ����ݲ��û���
}

void CPluginTickerPrice::NotifySocketClosed(SOCKET sock)
{
	DoClearReqInfo(sock);
}

void CPluginTickerPrice::OnTimeEvent(UINT nEventID)
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

void CPluginTickerPrice::OnMsgEvent(int nEvent,WPARAM wParam,LPARAM lParam)
{
	if ( EVENT_ID_ACK_REQUEST == nEvent )
	{
		ReplyAllReadyReq();
	}	
}

void CPluginTickerPrice::ClearQuoteDataCache()
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
		INT64 nStockID = it_todel->first;
		DWORD dwToDelTick = it_todel->second;

		MAP_STOCK_DATA_REQ::iterator it_req = m_mapReqInfo.find(nStockID);
		if ( it_req != m_mapReqInfo.end() )
		{
			it_todel = m_mapCacheToDel.erase(it_todel);
		}
		else
		{
			if ( int(dwTickNow - dwToDelTick) > 60*1000  )
			{
				m_mapCacheData.erase(nStockID);
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

void CPluginTickerPrice::HandleTimeoutReq()
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
		INT64 nStockID = it_stock->first;
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
				strTimeout.Format("BasicPrice req timeout, market=%d, code=%s", pReq->req.body.nStockMarket, pReq->req.body.strStockCode.c_str());
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

void CPluginTickerPrice::ReplyAllReadyReq()
{
	DWORD dwTickNow = ::GetTickCount();
	MAP_STOCK_DATA_REQ::iterator it_stock = m_mapReqInfo.begin();
	for ( ; it_stock != m_mapReqInfo.end(); )
	{
		INT64 nStockID = it_stock->first;
		VT_STOCK_DATA_REQ &vtReq = it_stock->second;
		MAP_STOCK_CACHE_DATA::iterator it_data = m_mapCacheData.find(nStockID);

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
		m_mapCacheData.erase(it_data);

		it_stock = m_mapReqInfo.erase(it_stock);
		m_mapCacheToDel[nStockID] = dwTickNow;
		SetTimerClearCache(true);
	}

	if ( m_mapReqInfo.empty() )
	{
		SetTimerHandleTimeout(false);
		return;
	}
}

void CPluginTickerPrice::ReplyStockDataReq(StockDataReq *pReq, const QuoteAckDataBody &data)
{
	CHECK_RET(pReq && m_pQuoteServer, NORET);

	CProtoQuote::ProtoAckDataType ack;
	ack.head = pReq->req.head;
	ack.head.ddwErrCode = 0;
	ack.body = data;

	//tomodify 4
	ack.body.nStockMarket = pReq->req.body.nStockMarket;
	ack.body.strStockCode = pReq->req.body.strStockCode;


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

void CPluginTickerPrice::ReplyDataReqError(StockDataReq *pReq, int nErrCode, LPCWSTR pErrDesc)
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

void CPluginTickerPrice::SetTimerHandleTimeout(bool bStartOrStop)
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

void CPluginTickerPrice::SetTimerClearCache(bool bStartOrStop)
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

void CPluginTickerPrice::ClearAllReqCache()
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

void CPluginTickerPrice::DoClearReqInfo(SOCKET socket)
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
