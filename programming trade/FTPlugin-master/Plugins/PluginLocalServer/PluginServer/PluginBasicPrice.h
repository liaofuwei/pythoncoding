#pragma once
#include "Include/Define_Struct_Plugin.h"
#include "Include/IQuote.h"
#include "Include/ITrade.h"
#include "Protocol/ProtoDataStruct_Quote.h"
#include "TimerWnd.h"
#include "MsgHandler.h"
#include "JsonCpp/json.h"

class CPluginQuoteServer;

class CPluginBasicPrice : public CTimerWndInterface, public CMsgHandlerEventInterface
{
public:
	CPluginBasicPrice();
	virtual ~CPluginBasicPrice();
	
	void Init(CPluginQuoteServer* pQuoteServer, IFTQuoteData*  pQuoteData);
	void Uninit();	
	void SetQuoteReqData(int nCmdID, const Json::Value &jsnVal, SOCKET sock);
	void NotifyQuoteDataUpdate(int nCmdID, INT64 nStockID);

protected:
	//CTimerWndInterface
	virtual void OnTimeEvent(UINT nEventID);

	//CMsgHandlerEventInterface
	virtual void OnMsgEvent(int nEvent,WPARAM wParam,LPARAM lParam);

protected:
	struct	StockDataReq
	{
		SOCKET	sock;
		DWORD	dwReqTick; 
		INT64	nStockID;		
		BasicPrice_Req req;
	};
	struct StockMktCode
	{
		int nMarketType;
		std::string strCode;
	};

	//tomodify 1
	typedef BasicPriceAckBody	QuoteAckDataBody;

	typedef std::vector<StockDataReq*>			VT_STOCK_DATA_REQ;
	typedef std::map<INT64, VT_STOCK_DATA_REQ>	MAP_STOCK_DATA_REQ;
	typedef std::map<INT64, QuoteAckDataBody>	MAP_STOCK_CACHE_DATA;
	typedef std::map<INT64, DWORD>				MAP_CACHE_TO_DESTROY;
	typedef std::map<INT64, StockMktCode>		MAP_STOCK_ID_CODE;


protected:
	void ClearQuoteDataCache();
	void HandleTimeoutReq();
	void ReplyAllReadyReq();
	void ReplyStockDataReq(StockDataReq *pReq, const QuoteAckDataBody &data);
	void ReplyDataReqError(StockDataReq *pReq, int nErrCode, LPCWSTR pErrDesc);	
	void SetTimerHandleTimeout(bool bStartOrStop);
	void SetTimerClearCache(bool bStartOrStop);
	bool GetStockMktCode(INT64 nStockID, StockMktCode &stkMktCode);
	void ClearAllReqCache();
	
protected:
	CPluginQuoteServer* m_pQuoteServer;
	IFTQuoteData*		m_pQuoteData;
	BOOL				m_bStartTimerClearCache;
	BOOL				m_bStartTimerHandleTimeout;

	MAP_STOCK_DATA_REQ		m_mapReqInfo;		//����δ������������󣬳���5000����δ��Ӧ����ֱ��Ӧ�����
	MAP_STOCK_CACHE_DATA	m_mapCacheData;		//�л���Ϳ��������ظ�
	MAP_CACHE_TO_DESTROY	m_mapCacheToDel;	//��������ϢΪ��ʱ��ʼ��ʱ��ʱ(500����)ɾ������
	MAP_STOCK_ID_CODE		m_mapStockIDCode;
	
	CTimerMsgWndEx		m_TimerWnd;
	CMsgHandler			m_MsgHandler;
};