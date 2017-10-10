#pragma once
#include <vector>
#include "ProtoDataStruct.h"


//////////////////////////////////////////////////////////////////////////
//��ȡ��������Э��, PROTO_ID_GET_BASIC_PRICE

struct	BasicPriceReqBody
{
	int nStockMarket;
	std::string strStockCode;
	
	BasicPriceReqBody()
	{
		nStockMarket = 0;
	}
};

struct BasicPriceAckBody
{
	int nHigh;
	int nOpen;
	int nClose;
	int nLastClose;
	int nLow;
	int nCur;
	INT64 nVolume;
	INT64 nTurnover;
	INT64 nLotSize;

	int nStockMarket;
	std::string strStockCode;
	DWORD dwTime;

	BasicPriceAckBody()
	{
		nHigh = 0;
		nOpen = 0;
		nClose = 0;
		nLastClose = 0;
		nLow = 0;
		nCur = 0;
		nVolume = 0;
		nTurnover = 0;
		nLotSize = 0;

		nStockMarket = 0;
		dwTime = 0;
	}
};

struct	BasicPrice_Req
{
	ProtoHead			head;
	BasicPriceReqBody	body;
};

struct	BasicPrice_Ack
{
	ProtoHead				head;
	BasicPriceAckBody		body;
};


//////////////////////////////////////////////////////////////////////////
//��ȡ������ϢЭ��, PROTO_ID_GET_GEAR_PRICE

struct	GearPriceReqBody
{
	int nNum;
	int nStockMarket;	
	std::string strStockCode;	

	GearPriceReqBody()
	{
		nNum = 0;
		nStockMarket = 0;
	}
};

struct GearPriceAckItem
{
	int nBuyOrder;
	int nSellOrder;
	int nBuyPrice;
	int nSellPrice;
	INT64 nBuyVolume;
	INT64 nSellVolume;

	GearPriceAckItem()
	{
		nBuyOrder = 0;
		nSellOrder = 0;
		nBuyPrice = 0;
		nSellPrice = 0;
		nBuyVolume = 0;
		nSellVolume = 0;
	}
};

typedef std::vector<GearPriceAckItem>	VT_GEAR_PRICE;

struct GearPriceAckBody 
{
	int nStockMarket;
	std::string strStockCode;
	VT_GEAR_PRICE vtGear;

	GearPriceAckBody()
	{
		nStockMarket = 0;
	}
};

struct	GearPrice_Req
{
	ProtoHead			head;
	GearPriceReqBody	body;
};

struct	GearPrice_Ack
{
	ProtoHead			head;
	GearPriceAckBody	body;
};

//////////////////////////////////////////////////////////////////////////
//��ȡ��ʱ����Э��, PROTO_ID_QT_GET_RT_DATA

struct	RTDataReqBody
{
	int nStockMarket;	
	std::string strStockCode;

	RTDataReqBody()
	{
		nStockMarket = 0;
	}
};

struct RTDataAckItem
{
	int   nDataStatus; 
	std::wstring strTime; 
	DWORD dwOpenedMins;  //���̵ڶ��ٷ���  

	int   nCurPrice;
	DWORD nLastClosePrice; //��������̼� 

	int   nAvgPrice;

	INT64 ddwTDVolume;
	INT64 ddwTDValue;  

	RTDataAckItem()
	{
		nDataStatus = 0;
		dwOpenedMins = 0;

		nCurPrice = 0;
		nLastClosePrice = 0;

		nAvgPrice = 0;

		ddwTDVolume = 0;
		ddwTDValue = 0;
	}
};

typedef std::vector<RTDataAckItem>	VT_RT_DATA;

struct RTDataAckBody 
{
	int nNum;
	int nStockMarket;
	std::string strStockCode;
	VT_RT_DATA vtRTData;

	RTDataAckBody()
	{
		nNum = 0;
		nStockMarket = 0;
	}
};

struct RTData_Req
{
	ProtoHead			head;
	RTDataReqBody		body;
};

struct RTData_Ack
{
	ProtoHead			head;
	RTDataAckBody		body;
};

//////////////////////////////////////////////////////////////////////////
//��ȡ��ǰK������Э��

struct	KLDataReqBody
{
	int nRehabType;
	int nKLType;
	int nStockMarket;
	int nNum;
	std::string strStockCode;

	KLDataReqBody()
	{
		nRehabType = 0;
		nKLType = 0;
		nStockMarket = 0;
		nNum = 0;
	}
};

struct KLDataAckItem
{
	int   nDataStatus; 
	std::wstring strTime; 
	INT64   nOpenPrice;
	INT64   nClosePrice;
	INT64   nHighestPrice;
	INT64   nLowestPrice;
	int   nPERatio; //��ӯ��(��λС��)
	int   nTurnoverRate;//������(���ɼ�ָ������/��/��K��)
	INT64 ddwTDVol; 
	INT64 ddwTDVal;

	KLDataAckItem()
	{
		nDataStatus = 0;
		nOpenPrice = 0;
		nClosePrice = 0;
		nHighestPrice = 0;
		nLowestPrice = 0;
		nPERatio = 0;
		nTurnoverRate = 0;
		ddwTDVol = 0;
		ddwTDVal = 0;
	}
};

typedef std::vector<KLDataAckItem>	VT_KL_DATA;

struct KLDataAckBody 
{
	int nRehabType;
	int nKLType;
	int nStockMarket;
	std::string strStockCode;
	VT_KL_DATA vtKLData;

	KLDataAckBody()
	{
		nRehabType = 0;
		nKLType = 0;
		nStockMarket = 0;
	}
};

struct KLData_Req
{
	ProtoHead			head;
	KLDataReqBody		body;
};

struct KLData_Ack
{
	ProtoHead			head;
	KLDataAckBody		body;
};

/////////////////////////////////////////
//��Ʊ����Э��, PROTO_ID_QT_SUBSTOCK

struct StockSubReqBody
{
	int nStockSubType;
	int nStockMarket;
	std::string strStockCode;

	StockSubReqBody()
	{
		nStockSubType = 0;
		nStockMarket = 0;
	}
};

struct StockSubAckBody
{
	int nStockSubType;
	int nStockMarket;
	std::string strStockCode;

	StockSubAckBody()
	{
		nStockSubType = 0;
		nStockMarket = 0;
	}
};

struct StockSub_Req
{
	ProtoHead		head;
	StockSubReqBody	body;
};

struct StockSub_Ack
{
	ProtoHead			head;
	StockSubAckBody		body;
};

/////////////////////////////////////////
//��Ʊ������Э��, PROTO_ID_QT_UNSUBSTOCK

struct StockUnSubReqBody
{
	int nStockSubType;
	int nStockMarket;
	std::string strStockCode;

	StockUnSubReqBody()
	{
		nStockSubType = 0;
		nStockMarket = 0;
	}
};

struct StockUnSubAckBody
{
	int nStockSubType;
	int nStockMarket;
	std::string strStockCode;

	StockUnSubAckBody()
	{
		nStockSubType = 0;
		nStockMarket = 0;
	}
};

struct StockUnSub_Req
{
	ProtoHead			head;
	StockUnSubReqBody	body;
};

struct StockUnSub_Ack
{
	ProtoHead				head;
	StockUnSubAckBody		body;
};

/////////////////////
////��ѯ���Ľӿ�PROTO_ID_QT_QueryStockSub
struct QueryStockSubReqBody
{
	QueryStockSubReqBody()
	{
		nQueryAllSocket = 0;
	}
	int nQueryAllSocket;//�Ƿ������������ӵĶ�����Ϣ 0��1
};

struct SubInfoAckItem
{
	int nStockSubType;
	int nStockMarket;
	std::wstring strStockCode;

	SubInfoAckItem()
	{
		nStockSubType = 0;
		nStockMarket = 0;
	}
};

typedef std::vector<SubInfoAckItem>	VT_SUB_INFO;

struct QueryStockSubAckBody
{
	VT_SUB_INFO vtSubInfo;
};

struct QueryStockSub_Req
{
	ProtoHead					head;
	QueryStockSubReqBody		body;
};

struct QueryStockSub_Ack
{
	ProtoHead					head;
	QueryStockSubAckBody		body;
};


//////////////////////////////////////////////////////////////////////////
//������Э��, PROTO_ID_QT_GET_TICKER

struct	TickerReqBody
{
	int nGetTickNum;
	int nStockMarket;	// enum StockMktType
	INT64 nSequence;	// Ŀǰû��������
	std::string strStockCode;	

	TickerReqBody()
	{
		nGetTickNum = 0;
		nStockMarket = 0;
		nSequence = 0;
	}
};

struct TickerAckItem
{
	int nPrice;	
	int nDealType;	
	INT64 nSequence;
	INT64 nVolume;
	INT64 nTurnover;
	std::string strTickTime;

	TickerAckItem()
	{
		nPrice = 0;
		nDealType = 0;
		nSequence = 0;
		nVolume = 0;
		nTurnover = 0;
	}

};

typedef std::vector<TickerAckItem>	VT_TICKER_DATA;

struct TickerAckBody 
{
	int nStockMarket;	// enum StockMktType
	std::string strStockCode;
	INT64 nNextSequence;// Ŀǰû��������
	VT_TICKER_DATA vtTicker;

	TickerAckBody()
	{
		nStockMarket = 0;
		nNextSequence = 0;
	}
};

struct	Ticker_Req
{
	ProtoHead		head;
	TickerReqBody	body;
};

struct	Ticker_Ack
{
	ProtoHead		head;
	TickerAckBody	body;
};


//////////////////////////////////////////////////////////////////////////
//���ָ��ʱ��εĽ������б�, PROTO_ID_QT_GET_TRADE_DATE

struct	TradeDateReqBody
{
	int nStockMarket;	// enum StockMktType	
	std::string strStartDate;//"YYYY-MM-DD"�����ַ�����ʾEnd��ǰһ��
	std::string strEndDate;//"YYYY-MM-DD"�����ַ�����ʾ����

	TradeDateReqBody()
	{
		nStockMarket = 0;
	}
};

//"YYYY-MM-DD"
typedef std::vector<std::string>	VT_TRADE_DATE; 

struct TradeDateAckBody 
{
	int nStockMarket;	// enum StockMktType	
	std::string strStartDate;
	std::string strEndDate;
	VT_TRADE_DATE vtTradeDate;

	TradeDateAckBody()
	{
		nStockMarket = 0;
	}
};

struct	TradeDate_Req
{
	ProtoHead			head;
	TradeDateReqBody	body;
};

struct	TradeDate_Ack
{
	ProtoHead			head;
	TradeDateAckBody	body;
};

//////////////////////////////////////////////////////////////////////////
//���ָ�����͵Ĺ�Ʊ��Ϣ, PROTO_ID_QT_GET_STOCK_LIST
struct	StockListReqBody
{
	int nStockMarket;	// enum StockMktType
	int nSecurityType;  // enum PluginSecurityType	

	StockListReqBody()
	{
		nStockMarket = 0;
		nSecurityType = 0;
	}
};

struct StockListAckItem
{
	INT64 nStockID;
	int nLotSize;	
	int nSecurityType;  // enum PluginSecurityType	
	int nStockMarket;
	std::string strStockCode;
	std::string strSimpName;

	int nSubType; //���������
	INT64 nOwnerStockID; //��������
	std::string strOwnerStockCode;
	int nOwnerMarketType;
	
	std::string strListDate; //����ʱ��

	StockListAckItem()
	{
		nStockID = 0;
		nLotSize = 0;
		nSecurityType = 0;
		nStockMarket = 0;

		nSubType = 0;
		nOwnerStockID = 0;
		nOwnerMarketType = 0;
	}
};

typedef std::vector<StockListAckItem>	VT_STOCK_INFO;

struct StockListAckBody 
{
	int nStockMarket;	// enum StockMktType	
	VT_STOCK_INFO vtStockList;

	StockListAckBody()
	{
		nStockMarket = 0;
	}
};

struct	StockList_Req
{
	ProtoHead			head;
	StockListReqBody	body;
};

struct	StockList_Ack
{
	ProtoHead			head;
	StockListAckBody	body;
};

//////////////////////////////////////////////////////////////////////////
//���ָ�����͵Ĺ�Ʊ��Ϣ, PROTO_ID_QT_GET_SNAPSHOT
struct SnapshotReqItem
{
	int nStockMarket; // enum StockMktType
	std::string strStockCode;	

	SnapshotReqItem()
	{
		nStockMarket = 0;
	}
};

typedef std::vector<SnapshotReqItem> VT_REQ_SNAPSHOT;

struct	SnapshotReqBody
{
	VT_REQ_SNAPSHOT vtReqSnapshot;
};

struct SnapshotAckItem
{
	INT64 nStockID;
	std::string strStockCode;
	int nStockMarket; // enum StockMktType
	int instrument_type;// enum PluginSecurityType	

	INT64 last_close_price;
	INT64 nominal_price;
	INT64 open_price;
	INT64  update_time;

	INT64  suspend_flag;
	INT64  listing_status;
	INT64  listing_date;
	INT64 shares_traded;

	INT64 turnover;
	INT64 highest_price;
	INT64 lowest_price;
	int turnover_ratio;
	int ret_err;//0Ϊ�ɹ�

	INT64 nTatalMarketVal; //��ֵ
	INT64 nCircularMarketVal; //��ͨ��ֵ
	UINT32 nLostSize; //ÿ��
	std::string strUpdateTime; //�����ַ���ʱ��

	struct tagWarrantData
	{
		BOOL bDataValid;  //��������� == 0 
		int  nWarrantType;  //�������� Quote_WarrantType
		UINT32 nConversionRatio; //���ɱ���
		INT64  nStrikePrice; //��ʹ��

		INT64  nMaturityDate; //������
		std::string strMaturityData;
		INT64 nEndtradeDate;  //�������
		std::string strEndtradeDate;

		std::string strOwnerStockCode;
		int nOwnerStockMarket; 

		INT64 nRecoveryPrice; //���ռ�
		UINT64 nStreetVol;  //�ֻ���
		UINT64 nIssueVol;  //������
		INT64 nOwnerStockPrice;  //���ɼ۸�

		int nStreetRatio; //�ֻ�ռ��
		int nDelta;	 //�Գ�ֵ
		int nImpliedVolatility; //���첨��
		int nPremium; //���

		tagWarrantData()
		{
			bDataValid = 0; nWarrantType = 0; nConversionRatio = 0; nStrikePrice = 0;
			nMaturityDate = 0; nEndtradeDate = 0;
			nOwnerStockMarket = 0;
			nRecoveryPrice = 0; nStreetVol = 0; nIssueVol = 0; nOwnerStockPrice = 0;
			nStreetRatio = 0; nDelta = 0; nImpliedVolatility = 0; nPremium = 0;
		}
	}stWrtData;

	SnapshotAckItem()
	{
		nStockID = 0;
		nStockMarket = 0;
		instrument_type = 0;

		last_close_price = 0;
		nominal_price = 0;
		open_price = 0;
		update_time = 0;

		suspend_flag = 0;
		listing_status = 0;
		listing_date = 0;
		shares_traded = 0;

		turnover = 0;
		highest_price = 0;
		lowest_price = 0;
		turnover_ratio = 0;
		ret_err = 0;

		nTatalMarketVal = 0;
		nCircularMarketVal = 0;
		nLostSize = 0;
	}
};

typedef std::vector<SnapshotAckItem>	VT_ACK_SNAPSHOT;

struct SnapshotAckBody 
{	
	VT_ACK_SNAPSHOT vtSnapshot;
};

struct	Snapshot_Req
{
	ProtoHead		head;
	SnapshotReqBody	body;
};

struct	Snapshot_Ack
{
	ProtoHead		head;
	SnapshotAckBody	body;
};

//////////////////////////////////////////////////////////////////////////
//��������, PROTO_ID_QT_GET_STOCK_INFO

struct BatchBasicReqItem
{
	int nStockMarket;
	std::string strStockCode;	

	BatchBasicReqItem()
	{
		nStockMarket = 0;
	}
};

typedef std::vector<BatchBasicReqItem>	VT_REQ_BATCHBASIC;

struct BatchBasicReqBody
{
	VT_REQ_BATCHBASIC vtReqBatchBasic;
};

struct BatchBasicAckItem
{
	int nStockMarket;
	std::string strStockCode;	

	int nHigh;
	int nOpen;
	int nLastClose;
	int nLow;

	int nCur;
	int nSuspension;
	int nTurnoverRate;

	INT64 nVolume;
	INT64 nValue;
	INT64 nAmpli;
	std::wstring strDate; 
	std::wstring strTime;
	std::wstring strListTime; 

	BatchBasicAckItem()
	{
		nStockMarket = 0;
		nHigh = 0;
		nOpen = 0;
		nLastClose = 0;
		nLow = 0;

		nCur = 0;
		nSuspension = 0;
		nTurnoverRate = 0;

		nVolume = 0;
		nValue = 0;
		nAmpli = 0;
	}
};

typedef std::vector<BatchBasicAckItem>	VT_ACK_BATCHBASIC;

struct BatchBasicAckBody 
{	
	VT_ACK_BATCHBASIC vtAckBatchBasic;
};

struct BatchBasic_Req
{
	ProtoHead			head;
	BatchBasicReqBody	body;
};

struct BatchBasic_Ack
{
	ProtoHead			head;
	BatchBasicAckBody	body;
};

//�����ʷK��, PROTO_ID_QT_GET_HISTORYKL

struct	HistoryKLReqBody
{
	int nRehabType;
	int nKLType;
	int nStockMarket;
	std::string strStockCode;
	std::string strStartDate;
	std::string strEndDate;

	HistoryKLReqBody()
	{
		nRehabType = 0;
		nKLType = 0;
		nStockMarket = 0;
	}
};

struct HistoryKLAckItem
{
	std::wstring strTime; 
	INT64   nOpenPrice;
	INT64   nClosePrice;
	INT64   nHighestPrice;
	INT64   nLowestPrice;

	int   nPERatio; //��ӯ��(��λС��)
	int   nTurnoverRate;//������(���ɼ�ָ������/��/��K��)
	INT64 ddwTDVol; 
	INT64 ddwTDVal;

	HistoryKLAckItem()
	{
		nOpenPrice = 0;
		nClosePrice = 0;
		nHighestPrice = 0;
		nLowestPrice = 0;

		nPERatio = 0;
		nTurnoverRate = 0;
		ddwTDVol = 0;
		ddwTDVal = 0;
	}
};

typedef std::vector<HistoryKLAckItem>	VT_HISTORY_KL;

struct HistoryKLAckBody 
{
	int nRehabType;
	int nKLType;
	int nStockMarket;
	std::string strStockCode;
	std::string strStartDate;
	std::string strEndDate;
	VT_HISTORY_KL vtHistoryKL;

	HistoryKLAckBody()
	{
		nRehabType = 0;
		nKLType = 0;
		nStockMarket = 0;
	}
};

struct HistoryKL_Req
{
	ProtoHead				head;
	HistoryKLReqBody		body;
};

struct HistoryKL_Ack
{
	ProtoHead				head;
	HistoryKLAckBody		body;
};

//////////////////////////////////////////////////////////////////////////
//��Ȩ, PROTO_ID_QT_GET_EXRIGHTINFO
struct ExRightInfoReqItem
{
	int nStockMarket; // enum StockMktType
	std::string strStockCode;	

	ExRightInfoReqItem()
	{
		nStockMarket = 0;
	}
};

typedef std::vector<ExRightInfoReqItem> VT_REQ_EXRIGHTINFO;

struct ExRightInfoReqBody
{
	VT_REQ_EXRIGHTINFO vtReqExRightInfo;
};

struct ExRightInfoAckItem
{
	int nStockMarket; // enum StockMktType
	std::string strStockCode;	
	std::string str_ex_date;    // ��Ȩ��Ϣ����, ����20160615
	
	INT32 split_ratio;//��Ϲɱ���
	INT64 per_cash_div;//�ֽ�����
	INT32 per_share_ratio;//�͹ɱ���
	INT32 per_share_trans_ratio;//ת���ɱ���

	INT32 allotment_ratio;//��ɱ���
	INT64 allotment_price;//��ɼ�
	INT32 stk_spo_ratio;//��������
	INT64 stk_spo_price;//�����۸�

	// result_self
	INT64 fwd_factor_a;
	INT64 fwd_factor_b;
	INT64 bwd_factor_a;
	INT64 bwd_factor_b;

	// ���������ı�����
	//std::wstring str_sc_txt;
	// ���������ı�����
	//std::wstring str_tc_txt;

	ExRightInfoAckItem()
	{
		nStockMarket = 0;
		split_ratio = 0;
		per_cash_div = 0;
		per_share_ratio = 0;
		per_share_trans_ratio = 0;

		allotment_ratio = 0;
		allotment_price = 0;
		stk_spo_ratio = 0;
		stk_spo_price = 0;

		fwd_factor_a = 0;
		fwd_factor_b = 0;
		bwd_factor_a = 0;
		bwd_factor_b = 0;
	}
};

typedef std::vector<ExRightInfoAckItem>	VT_ACK_EXRIGHTINFO;

struct ExRightInfoAckBody 
{	
	VT_ACK_EXRIGHTINFO vtAckExRightInfo;
};

struct	ExRightInfo_Req
{
	ProtoHead		head;
	ExRightInfoReqBody	body;
};

struct ExRightInfo_Ack
{
	ProtoHead		head;
	ExRightInfoAckBody	body;
};

/////////////////////////////////////////
//����Э��, PROTO_ID_QT_PushStockData

struct PushStockDataReqBody
{
	int nStockPushType;
	int nStockMarket;
	int nUnPush;
	std::string strStockCode;

	PushStockDataReqBody()
	{
		nStockPushType = 0;
		nStockMarket = 0;
		nUnPush = 0;
	}
};

struct PushStockDataAckBody
{
	int nStockPushType;
	int nStockMarket;
	std::string strStockCode;

	PushStockDataAckBody()
	{
		nStockPushType = 0;
		nStockMarket = 0;
	}
};

struct PushStockData_Req
{
	ProtoHead		head;
	PushStockDataReqBody	body;
};

struct PushStockData_Ack
{
	ProtoHead					head;
	PushStockDataAckBody		body;
};


//���ͱ���, PROTO_PUSH_BATCHPRICE

struct PushBatchBasicReqItem
{
	int nStockMarket;
	std::string strStockCode;

	PushBatchBasicReqItem()
	{
		nStockMarket = 0;
	}
};

typedef std::vector<PushBatchBasicReqItem>	VT_REQ_PUSHBATCHBASIC;

struct PushBatchBasicReqBody
{
	VT_REQ_PUSHBATCHBASIC vtReqBatchBasic;
};

struct PushBatchBasicAckItem
{
	int nStockMarket;
	std::string strStockCode;

	int nHigh;
	int nOpen;
	int nLastClose;
	int nLow;

	int nCur;
	int nSuspension;
	int nTurnoverRate;
	INT64 nVolume;
	INT64 nValue;
	INT64 nAmpli;
	std::wstring strDate;
	std::wstring strTime;
	std::wstring strListTime;

	PushBatchBasicAckItem()
	{
		nStockMarket = 0;
		nHigh = 0;
		nOpen = 0;
		nLastClose = 0;
		nLow = 0;

		nCur = 0;
		nSuspension = 0;
		nTurnoverRate = 0;
		nVolume = 0;
		nValue = 0;
		nAmpli = 0;
	}
};

typedef std::vector<PushBatchBasicAckItem>	VT_ACK_PUSHBATCHBASIC;

struct PushBatchBasicAckBody
{
	VT_ACK_PUSHBATCHBASIC vtAckBatchBasic;
};

struct PushBatchBasic_Req
{
	ProtoHead			head;
	PushBatchBasicReqBody	body;
};

struct PushBatchBasic_Ack
{
	ProtoHead			head;
	PushBatchBasicAckBody	body;
};

//
//heart beat
//
struct PushHeartBeatReqBody
{
	DWORD dwReserved;
};
struct PushHeartBeat_Req
{
	ProtoHead head;
	PushHeartBeatReqBody body;
};

struct PushHeartBeatAckBody
{
	UINT64 nTimeStamp;
	PushHeartBeatAckBody()
	{
		nTimeStamp = 0;
	}
};

struct PushHeartBeat_Ack
{
	ProtoHead head;
	PushHeartBeatAckBody body;
};

//////////////////////////////////////////////////////////////////////////
//���Ͱ���, PROTO_ID_PUSH_GERA

struct	PushGearPriceReqBody
{
	int nNum;
	int nStockMarket;
	std::string strStockCode;

	PushGearPriceReqBody()
	{
		nNum = 0;
		nStockMarket = 0;
	}
};

struct PushGearPriceAckItem
{
	int nBuyOrder;
	int nSellOrder;
	int nBuyPrice;
	int nSellPrice;
	INT64 nBuyVolume;
	INT64 nSellVolume;

	PushGearPriceAckItem()
	{
		nBuyOrder = 0;
		nSellOrder = 0;
		nBuyPrice = 0;
		nSellPrice = 0;
		nBuyVolume = 0;
		nSellVolume = 0;
	}
};

typedef std::vector<PushGearPriceAckItem>	VT_GEAR_PRICE_PUSH;

struct PushGearPriceAckBody
{
	int nStockMarket;
	std::string strStockCode;
	VT_GEAR_PRICE_PUSH vtGear;
};

struct	PushGearPrice_Req
{
	ProtoHead			head;
	PushGearPriceReqBody	body;
};

struct	PushGearPrice_Ack
{
	ProtoHead			head;
	PushGearPriceAckBody	body;
};

//////////////////////////////////////////////////////////////////////////
//�������Э��, PROTO_ID_PUSH_TICKER

struct	PushTickerReqBody
{
	int nGetTickNum;
	int nStockMarket;	// enum StockMktType
	INT64 nSequence;	// Ŀǰû��������
	std::string strStockCode;

	PushTickerReqBody()
	{
		nGetTickNum = 0;
		nStockMarket = 0;
		nSequence = 0;
	}
};

struct PushTickerAckItem
{
	int nPrice;
	int nDealType;
	INT64 nSequence;
	INT64 nVolume;
	INT64 nTurnover;
	std::string strTickTime;

	PushTickerAckItem()
	{
		nPrice = 0;
		nDealType = 0;
		nSequence = 0;
		nVolume = 0;
		nTurnover = 0;
	}
};

typedef std::vector<PushTickerAckItem>	VT_TICKER_DATA_PUSH;

struct PushTickerAckBody
{
	int nStockMarket;	// enum StockMktType
	std::string strStockCode;
	INT64 nNextSequence;// Ŀǰû��������
	VT_TICKER_DATA_PUSH vtTicker;

	PushTickerAckBody()
	{
		nStockMarket = 0;
		nNextSequence = 0;
	}
};

struct	PushTicker_Req
{
	ProtoHead		head;
	PushTickerReqBody	body;
};

struct	PushTicker_Ack
{
	ProtoHead		head;
	PushTickerAckBody	body;
};

//////////////////////////////////////////////////////////////////////////
//����K������Э��

struct	PushKLDataReqBody
{
	int nRehabType;
	int nKLType;
	int nStockMarket;
	int nNum;
	std::string strStockCode;

	PushKLDataReqBody()
	{
		nRehabType = 0;
		nKLType = 0;
		nStockMarket = 0;
		nNum = 0;
	}
};

struct PushKLDataAckItem
{
	std::wstring strTime;
	INT64   nOpenPrice;
	INT64   nClosePrice;
	INT64   nHighestPrice;
	INT64   nLowestPrice;

	int   nPERatio; //��ӯ��(��λС��)
	int   nTurnoverRate;//������(���ɼ�ָ������/��/��K��)
	INT64 ddwTDVol;
	INT64 ddwTDVal;

	PushKLDataAckItem()
	{
		nOpenPrice = 0;
		nClosePrice = 0;
		nHighestPrice = 0;
		nLowestPrice = 0;

		nPERatio = 0;
		nTurnoverRate = 0;
		ddwTDVol = 0;
		ddwTDVal = 0;
	}
};

typedef std::vector<PushKLDataAckItem>	VT_KL_DATA_PUSH;

struct PushKLDataAckBody
{
	int nRehabType;
	int nKLType;
	int nStockMarket;
	std::string strStockCode;
	VT_KL_DATA_PUSH vtKLData;

	PushKLDataAckBody()
	{
		nRehabType = 0;
		nKLType = 0;
		nStockMarket = 0;
	}
};

struct PushKLData_Req
{
	ProtoHead			head;
	PushKLDataReqBody		body;
};

struct PushKLData_Ack
{
	ProtoHead			head;
	PushKLDataAckBody		body;
};

//////////////////////////////////////////////////////////////////////////
//���ͷ�ʱ����Э��

struct	PushRTDataReqBody
{
	int nStockMarket;
	std::string strStockCode;

	PushRTDataReqBody()
	{
		nStockMarket = 0;
	}
};

struct PushRTDataAckItem
{
	int   nDataStatus;
	std::wstring strTime;
	DWORD dwOpenedMins;  //���̵ڶ��ٷ���  

	int   nCurPrice;
	DWORD nLastClosePrice; //��������̼� 

	int   nAvgPrice;

	INT64 ddwTDVolume;
	INT64 ddwTDValue;

	PushRTDataAckItem()
	{
		nDataStatus = 0;
		dwOpenedMins = 0;

		nCurPrice = 0;
		nLastClosePrice = 0;

		nAvgPrice = 0;

		ddwTDVolume = 0;
		ddwTDValue = 0;
	}
};

typedef std::vector<PushRTDataAckItem>	VT_RT_DATA_PUSH;

struct PushRTDataAckBody
{
	int nNum;
	int nStockMarket;
	std::string strStockCode;
	VT_RT_DATA_PUSH vtRTData;

	PushRTDataAckBody()
	{
		nNum = 0;
		nStockMarket = 0;
	}
};

struct PushRTData_Req
{
	ProtoHead			head;
	PushRTDataReqBody		body;
};

struct PushRTData_Ack
{
	ProtoHead			head;
	PushRTDataAckBody		body;
};

//ָ����鼯���µİ��ID�б�
//
enum PlateClass
{
	PlateClass_All = 0,
	PlateClass_Industry = 1,  //�����ҵ����
	PlateClass_Region = 2,    //���������
	PlateClass_Concept = 3,	  //���������
};

struct	PlatesetIDsReqBody
{
	int nPlateClassType;
	int nStockMarket;

	PlatesetIDsReqBody()
	{
		nPlateClassType = 0;
		nStockMarket = 0;
	}
};

struct PlatesetIDsAckItem
{
	int nStockMarket;
	std::string strStockCode;
	std::string strStockName;
	UINT64  nStockID;

	PlatesetIDsAckItem()
	{
		nStockMarket = 0;
		nStockID = 0;
	}
};
typedef std::vector<PlatesetIDsAckItem>	VT_PlatesetID;

struct PlatesetIDsAckBody
{
	int nPlateClassType;
	int nStockMarket;

	VT_PlatesetID vtPlatesetIDs;

	PlatesetIDsAckBody()
	{
		nPlateClassType = 0;
		nStockMarket = 0;
	}
};

struct	PlatesetIDs_Req
{
	ProtoHead			head;
	PlatesetIDsReqBody	body;
};

struct	PlatesetIDs_Ack
{
	ProtoHead				head;
	PlatesetIDsAckBody		body;
};

//ָ����鼯���µĹ�ƱID�б�
//
struct	PlateSubIDsReqBody
{
	int nStockMarket;
	std::string strStockCode;

	PlateSubIDsReqBody()
	{
		nStockMarket = 0;
	}
};

typedef StockListAckItem PlateSubIDsAckItem;
typedef std::vector<PlateSubIDsAckItem>	VT_PlateSubID;

struct PlateSubIDsAckBody
{
	int nStockMarket;
	std::string strStockCode;

	VT_PlateSubID vtPlateSubIDs;

	PlateSubIDsAckBody()
	{
		nStockMarket = 0;
	}
};

struct	PlateSubIDs_Req
{
	ProtoHead			head;
	PlateSubIDsReqBody	body;
};

struct	PlateSubIDs_Ack
{
	ProtoHead				head;
	PlateSubIDsAckBody		body;
};

//////////////////////////////////////////////////////////////////////////
//���Ͷ���Э��, PROTO_ID_GET_BROKER_QUEUE

struct	BrokerQueueReqBody
{
	int nStockMarket;
	std::string strStockCode;

	BrokerQueueReqBody()
	{
		nStockMarket = 0;
	}
};

struct BrokerQueueAckItem
{
	int nBrokerID;	//����id 
	std::string strBrokerName;  
	int nBrokerPos;  //���Ͱ��̵�λ, ȡֵ 0 , 1, 2, 3 ...

	BrokerQueueAckItem()
	{
		nBrokerID = 0;
		nBrokerPos = 0;
	}
};

typedef std::vector<BrokerQueueAckItem>	VT_BROKER_QUEUE;

struct BrokerQueueAckBody
{
	int nStockMarket;
	std::string strStockCode;
	VT_BROKER_QUEUE vtBrokerAsk; //���̾���
	VT_BROKER_QUEUE vtBrokerBid; //���̾���

	BrokerQueueAckBody()
	{
		nStockMarket = 0;
	}
};

struct	BrokerQueue_Req
{
	ProtoHead			head;
	BrokerQueueReqBody	body;
};

struct	BrokerQueue_Ack
{
	ProtoHead			head;
	BrokerQueueAckBody	body;
};



//////////////////////////////////////////////////////////////////////////
//��ȡȫ��״̬ , PROTO_ID_GET_GLOBAL_STATE

struct	GlobalStateReqBody
{
	int nStateType;  //no use now 
	
	GlobalStateReqBody()
	{
		nStateType = 0;
	}
};

struct GlobalStateAckBody
{
	int nMarketStateHK;
	int nMarketStateUS;
	int nMarketStateSH;
	int nMarketStateSZ;
	int nMarketStateHKFuture;
	int nQuoteLogined;
	int nTradeLogined;

	GlobalStateAckBody()
	{
		memset(this, 0, sizeof(*this));
	}
};

struct	GlobalState_Req
{
	ProtoHead			head;
	GlobalStateReqBody	body;
};

struct	GlobalState_Ack
{
	ProtoHead			head;
	GlobalStateAckBody	body;
};
