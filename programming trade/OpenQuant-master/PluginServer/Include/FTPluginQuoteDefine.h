#pragma once

#ifndef interface
#define interface struct
#endif

/*************************************************
Copyright: FUTU
Author: ysq
Date: 2015-03-18
Description: ����API�ͻص��ӿڶ���
**************************************************/

//Э������ʱ
#define REQ_TIMEOUT_MILLISECOND   10000

/**
*��Ʊ���г�����
*/
enum StockMktType
{
	StockMkt_None = 0,
	StockMkt_HK = 1,  //�۹� 
	StockMkt_US = 2,  //����
	StockMkt_SH = 3,  //����
	StockMkt_SZ = 4,  //���
	StockMkt_Feature_Old = 5,  //�ɵ��ڻ� code: 999000, 999001 �����ڻ���ʱ������һ��������
	StockMkt_Feature_New = 6,  //���ڻ� code: 999010, 999011 �����ڻ���ʱ���ݻ���죬�봫ͳ�������һ�£�
};
#define  IsValidMktID(mkt)  ((int)mkt >= (int)StockMkt_HK && (int)mkt <= (int)StockMkt_Feature_New)

enum StockSubErrCode
{
	StockSub_Suc = 0,	//���ĳɹ�
	StockSub_FailUnknown = 1,	//δ֪��ʧ��
	StockSub_FailMaxSubNum = 2,	//�����������
	StockSub_FailCodeNoFind = 3,	//����û�ҵ�(Ҳ�п������г����ʹ���)
	StockSub_FailGuidNoFind = 4,	//���GUID����
	StockSub_FailNoImplInf = 5,		//����ӿ�δ���
	StockSub_UnSubTimeError = 6,	//δ���㷴����Ҫ��ʱ��-1����
};

enum QueryDataErrCode
{
	QueryData_Suc = 0,	//��ѯ�ɹ�
	QueryData_FailUnknown = 1,	//δ֪��ʧ��
	QueryData_FailMaxSubNum = 2,	//��������ѯ��
	QueryData_FailCodeNoFind = 3,	//����û�ҵ�(Ҳ�п������г����ʹ���)
	QueryData_FailGuidNoFind = 4,	//���GUID����
	QueryData_FailNoImplInf = 5,		//����ӿ�δ���
	
	QueryData_FailFreqLimit = 7,	//��ѯƵ�����Ƶ���ʧ��
	QueryData_FailNetwork = 8,		//�����쳣������ʧ��
	QueryData_FailErrParam = 9,		//��������
};

enum StockSubType
{
	StockSubType_None = 0,
	StockSubType_Simple = 1,
	StockSubType_Gear = 2,
	StockSubType_Ticker = 4,
	StockSubType_RT = 5,
	StockSubType_KL_DAY = 6,
	StockSubType_KL_MIN5 = 7,
	StockSubType_KL_MIN15 = 8,
	StockSubType_KL_MIN30 = 9,
	StockSubType_KL_MIN60 = 10,
	StockSubType_KL_MIN1 = 11,
	StockSubType_KL_WEEK = 12,
	StockSubType_KL_MONTH = 13,
	StockSubType_Broker = 14, //���ľ��Ͷ���

	StockSubType_Max = StockSubType_Broker + 1,
};

#define  IsStockSubType_RTKL(eType)  (StockSubType_RT == eType || StockSubType_KL_DAY == eType || StockSubType_KL_MIN5 == eType ||  \
									  StockSubType_KL_MIN15 == eType || StockSubType_KL_MIN30 == eType || StockSubType_KL_MIN60 == eType ||  \
									  StockSubType_KL_MIN1 == eType || StockSubType_KL_WEEK == eType || StockSubType_KL_MONTH == eType)


//IFTQuoteOperation::QueryStockKLData��nKLType ���� 
enum
{
	FT_KL_CLASS_MIN_1 = 1,
	FT_KL_CLASS_DAY = 2,
	FT_KL_CLASS_WEEK = 3,
	FT_KL_CLASS_MONTH = 4,
	FT_KL_CLASS_YEAR = 5,
	FT_KL_CLASS_MIN_5 = 6,
	FT_KL_CLASS_MIN_15 = 7,
	FT_KL_CLASS_MIN_30 = 8,
	FT_KL_CLASS_MIN_60 = 9,
};

enum PluginSecurityType
{
	PluginSecurity_All = 0,
	PluginSecurity_Bond = 1, //ծȯ	
	PluginSecurity_Stock = 3, //����	
	PluginSecurity_ETF = 4,
	PluginSecurity_Warrant = 5, //����ţ��		
	PluginSecurity_Index = 6,
};

/**
* ��Ʊ����������Ϣ��
* �۸񾫶���3λС���� �籨��8.888�洢ֵ 88888
*/
typedef struct tagQuotePriceBase
{
	DWORD dwOpen;		//���̼�
	DWORD dwLastClose;  //���ռ�
	DWORD dwCur;		//��ǰ��
	DWORD dwHigh;		//��߼�
	DWORD dwLow;		//��ͼ�
	INT64 ddwVolume;	//�ɽ���
	INT64 ddwTrunover;	//�ɽ���
	DWORD dwTime;		//����ʱ��
	DWORD ddwLotSize;	//ÿ������
}Quote_PriceBase, *LPQuote_PriceBase;


/**
* ��Ʊʮ������
* IFTQuoteData::FillOrderQueue �Ľӿڲ���
*/
typedef struct tagQuoteOrderItem
{
	DWORD	dwBuyPrice, dwSellPrice;  //��� ����
	INT64	ddwBuyVol, ddwSellVol;    //���� ����
	int		nBuyOrders, nSellOrders;  //��λ 
}Quote_OrderItem, *LPQuote_OrderItem;

typedef struct tagPluginTickItem
{
	DWORD dwPrice;
	DWORD dwTime;
	int nDealType;
	INT64 nSequence;
	INT64 nVolume;
	INT64 nTurnover; //�ɽ���
}PluginTickItem, *LPPluginTickItem;

enum RTKL_DATA_STATUS
{
	RTKL_DATA_STATUS_NULL = 0,
	RTKL_DATA_STATUS_OVER = 1,		//���
	RTKL_DATA_STATUS_HALT = 2,    //ͣ�� 
	RTKL_DATA_STATUS_RUNNING = 3,	   //������ 
	RTKL_DATA_STATUS_FAKED = 4,		//α������� 
	RTKL_DATA_STATUS_NO_OCCUR = 5,     //δ������	
	RTKL_DATA_STATUS_CLIENT_FILL = 6,	//�ͻ����Զ����ĵ㣬��������ʱ�߼���ͬ��������ͬFG_RT_DATA_NO_OCCUR���ο�BugID:5073
	RTKL_DATA_STATUS_FACK_FUTURE = 7, //δ���㣬Ҳ��fake״̬��һ��
};

//������Զ���push��Ч��
#define  IS_RTKL_VALID_PUSH_DATA_STATUS(nStatus) (RTKL_DATA_STATUS_NULL != nStatus && RTKL_DATA_STATUS_FAKED != nStatus && RTKL_DATA_STATUS_NO_OCCUR != nStatus && RTKL_DATA_STATUS_FACK_FUTURE != nStatus)

/**
* ��ʱ����
*/
typedef struct tagQuoteStockRTData
{
	int   nDataStatus;
	DWORD dwTime;
	DWORD dwOpenedMins;  //���̵ڶ��ٷ���  

	int   nCurPrice;
	DWORD nLastClosePrice; //��������̼� 

	int   nAvgPrice;

	INT64 ddwTDVolume;
	INT64 ddwTDValue;
}Quote_StockRTData, *LPQuote_StockRTData;

/**
* K������
*/
typedef struct tagQueryStockKLData
{
	int   nDataStatus;
	DWORD dwTime;

	INT64   nOpenPrice;
	INT64   nClosePrice;

	INT64   nHighestPrice;
	INT64   nLowestPrice;

	int   nPERatio; //��ӯ��(��λС��)
	int   nTurnoverRate;//������(���ɼ�ָ������/��/��K��)

	INT64 ddwTDVol;
	INT64 ddwTDVal;
}Quote_StockKLData, *LPQuote_StockKLData;

typedef struct tagSubInfo
{
	INT64 ddwStockHash;
	StockSubType eStockSubType;
	bool operator==(const tagSubInfo &Item) const
	{
		return (this->ddwStockHash == Item.ddwStockHash && this->eStockSubType == Item.eStockSubType);
	};

}Quote_SubInfo, *LPQuote_SubInfo;

//nKLType:
//1 = 1��K;  
//2 = ��K; 
//3 = ��K;   
//4 = ��K;
//6 = 5��K; 
//7 = 15��K;  
//8 = 30��K;  
//9 = 60��K;
//
//nRehabType:
//0 = ����Ȩ��
//1 = ǰ��Ȩ��
//2 = ��Ȩ��

//nStockSubType:
//1 = ����
//2 = ����
//4 = ���
//5 = ��ʱ//δ��
//6 = ��K
//7 =  5��K
//8 =  15��K
//9 =  30��K
//10 =  60��K
//11 =  1��K
//12 = ��K
//13 = ��K

typedef struct tagPluginStockInfo
{
	INT64 nStockID;
	int  nLotSize;
	PluginSecurityType nSecType;
	WCHAR chSimpName[64];
	WCHAR chCodeSig[16];
	int nSubType; 
	INT64 nOwnerStockID;
	WCHAR chListDate[12];
}PluginStockInfo, *LPPluginStockInfo;

typedef struct tagBatchBasic
{
	DWORD dwOpen;		//���̼�
	DWORD dwLastClose;  //���ռ�
	DWORD dwCur;		//��ǰ��
	DWORD dwHigh;		//��߼�
	DWORD dwLow;		//��ͼ�
	DWORD dwAmpli;
	INT64 ddwVolume;	//�ɽ���
	INT64 ddwTurnover;	//�ɽ���
	int   nSuspension;
	int   nTurnoverRate;
	DWORD dwTime;		//����ʱ��
	DWORD dwListTime;	//����ʱ��
}Quote_BatchBasic, *LPQuote_BatchBasic;

enum PlugErtFlag
{
	Ert_NONE = 0x00,
	Ert_SPLIT = 0x01,    //���
	Ert_JOIN = 0x02,	 //�Ϲ�
	Ert_BONUS_STK = 0x04,//�͹�
	Ert_INTOSHARES = 0x08,//ת����
	Ert_ALLOT = 0x10,    //���
	Ert_ADD = 0x20,		//������
	Ert_DIVIDEND = 0x40,//���ֽ�ֺ�
	Ert_SPECIALDIVIDEND = 0x80, //���ֽ�ֺ�
};

//��Ȩ��¼��
struct PluginExRightItem
{
	INT64 stock_id;
	UINT ex_date;    // ��Ȩ��Ϣ����, ����20160615
	UINT ert_flag;    // ��˾�ж�������ϣ�ErtFlag

	//���(eg. 1��5��BaseΪ1��ERTΪ5)
	UINT split_base;
	UINT split_ert;

	//�Ϲ�(eg. 50��1��BaseΪ50��ERTΪ1)
	UINT join_base;
	UINT join_ert;

	//�͹�(eg. 10��3, BaseΪ10,ERTΪ3)
	UINT bonus_stk_base;
	UINT bonus_stk_ert;

	//���(eg. 10��2, ��ɼ�Ϊ6.3Ԫ, BaseΪ10, ERTΪ2, PriceΪ6300)
	UINT allot_base;
	UINT allot_ert;
	double allot_price;

	//ת����(���͹�����)
	UINT into_shr_base;
	UINT into_ert;

	//����(���������)
	UINT stk_add_base;
	UINT stk_add_ert;
	double stk_add_price;

	// �ֽ�ֺ�(eg. ÿ10������0.5Ԫ��BaseΪ10, AmountΪ500)
	UINT dividend_base;
	double dividend_amount;

	// �ر��Ϣ
	UINT dividend_special_base;
	double dividend_special_amount;

	// result_self
	double fwd_factor_a;
	double fwd_factor_b;
	double bwd_factor_a;
	double bwd_factor_b;

	// ���������ı�����
	wchar_t sc_txt[160];

	// ���������ı�����
	wchar_t tc_txt[160];
};

enum Quote_WarrantType
{
	Quote_WarrantType_Buy = 1,  //�Ϲ�
	Quote_WarrantType_Sell = 2, //�Ϲ�
	Quote_WarrantType_Bull = 3, //ţ
	Quote_WarrantType_Bear = 4, //��
};

// ��Ʊ�������������
struct PluginStockSnapshot
{
	PluginStockSnapshot()
	{
		memset(this, 0, sizeof(*this));
	}
	INT64 stock_id;    // ��Ʊid
	int  ret;    // �Ƿ��ҵ����ռ�¼��0Ϊ�ɹ��ҵ���snapshot�����ݡ�����ֵ��snapshot�����ݣ��������Ҳ�����Ʊ��
	//char   stock_code[16];
	UINT instrument_type;
	UINT market_code;

	// �۸����
	double nominal_price;
	double last_close_price;
	double open_price;
	INT64 update_time;

	INT64 suspend_flag;
	INT64 listing_status;
	INT64 listing_date;

	// �ɽ�ͳ����Ϣ
	INT64 shares_traded;
	double turnover;
	double highest_price;
	double lowest_price;
	float  turnover_ratio;

	//ÿ��
	UINT32  nLotSize;

	//������������
	struct tagEquitiesData
	{
		bool bDataValid; //�����Ƿ���Ч
		UINT64 nIssuedShares; //���йɱ�,���ܹɱ�
		double dbNetAssetValue; //�ʲ���ֵ
		double dbNetProfit; //ӯ��������
		double dbEarningPerShare; //ÿ��ӯ��
		UINT64 nOutStandingShares; //��ͨ�ɱ�
		double dbNetAssetPerShare; //ÿ�ɾ��ʲ�
		double dbEYRatio; //������
		double dbPERatio; //��ӯ��
		double dbPBRatio; //�о��� 
	}stEquitiesData;

	//�����������
	struct tagWarrantsData
	{
		bool bDataValid;  //��������� == 0
		UINT32 nConversionRatio; //���ɱ���
		int  nWarrantType;  //�������� Quote_WarrantType
		double dbStrikePrice; //��ʹ��
		INT64  nMaturityDate; //������
		INT64 nEndtradeDate;  //�������
		INT64 nWarrantOwnerID; //����ID

		UINT32 nIssuerCode; //������id
		char  strIssuerName[64]; //����������
		double dbRecoveryPrice; //���ռ�
		UINT64 nStreetVol;  //�ֻ���
		UINT64 nIssueVol;  //������
		double dbOwnerStockPrice;  //���ɼ۸�
		double dbStreetRatio; //�ֻ�ռ��
		double dbDelta;	 //�Գ�ֵ
		double dbImpliedVolatility; //���첨��
		double dbPremiun; //���		
	}stWrtData;
};

typedef struct tagStockPushInfo
{
	INT64 ddwStockID;
	INT64 nSequence;
	DWORD dwKLTime_1MIN;
	DWORD dwKLTime_5MIN;
	DWORD dwKLTime_15MIN;
	DWORD dwKLTime_30MIN;
	DWORD dwKLTime_60MIN;
	DWORD dwKLTime_DAY;
	DWORD dwKLTime_WEEK;
	DWORD dwKLTime_MONTH;
	DWORD dwRTTime;
	tagStockPushInfo()
	{
		ddwStockID = 0;
		nSequence = 0;
		dwKLTime_1MIN = 0;
		dwKLTime_5MIN = 0;
		dwKLTime_15MIN = 0;
		dwKLTime_30MIN = 0;
		dwKLTime_60MIN = 0;
		dwKLTime_DAY = 0;
		dwKLTime_WEEK = 0;
		dwKLTime_MONTH = 0;
		dwRTTime = 0;
	}
}Stock_PushInfo, *LPStock_PushInfo;

//���Ͷ���
typedef struct tagQuoteBrokerItem
{
	bool bAskOrBid;  //ask=��  bid=�� 
	int nBrokerID;
	int nBrokerPos;
	char strBrokerName[32];
}Quote_BrokerItem, *LPQuote_BrokerItem;


enum FT_MARKET_STATUS
{
	FT_MARKET_STATUS_NONE = 0, //�޽��ף�			���ɵ�δ����
	FT_MARKET_STATUS_JJJY = 1, //���۽���
	FT_MARKET_STATUS_WAITOPEN = 2,   //����ǰ�ȴ�����
	FT_MARKET_STATUS_MORNING = 3,     //����,		
	FT_MARKET_STATUS_NOON_REST = 4,   //����
	FT_MARKET_STATUS_NOON_TRADE = 5,  //���� (���ɵ�����)
	FT_MARKET_STATUS_TRADE_OVER = 6, //�����ս���,	���ɵ�������
	FT_MARKET_STATUS_BEFORE_BEGIN = 8, //���ɵ���ǰ��ʼ
	FT_MARKET_STATUS_BEFORE_END = 9,	//���ɵ���ǰ����
	FT_MARKET_STATUS_AFTER_BEGIN = 10,	//���ɵ��̺�ʼ
	FT_MARKET_STATUS_AFTER_END = 11,	//���ɵ��̺����
	FT_MARKET_STATUS_FUTU_SWITCH_DATE = 12, //��;���л�����״̬��ר������ֻ����һ�½�����

	FT_MARKET_STATUS_FUTURE_NIGHT_TRADE = 13,//ҹ�н�����
	FT_MARKET_STATUS_FUTURE_NIGHT_END = 14,//ҹ������
	FT_MARKET_STATUS_FUTURE_DAY_TRADE = 15,//���н�����
	FT_MARKET_STATUS_FUTURE_DAY_BREAK = 16, //��������
	FT_MARKET_STATUS_FUTURE_DAY_CLOSE = 17, //��������
	FT_MARKET_STATUS_FUTURE_DAY_WAIT_OPEN = 18, //���еȴ�����

	FT_MARKET_STATUS_HK_CLOSING_AUCTION = 19, //�۹��̺󾺼�
};

//ȫ��״̬ 
typedef struct tagNNGlobalState
{
	bool bQuoteSvrLogined;
	bool bTradeSvrLogined;

	FT_MARKET_STATUS eMktHK;
	FT_MARKET_STATUS eMktHKFuture;
	FT_MARKET_STATUS eMktUS;
	FT_MARKET_STATUS eMktSH;
	FT_MARKET_STATUS eMktSZ;

	tagNNGlobalState()
	{
		bQuoteSvrLogined = false;
		bTradeSvrLogined = false;

		eMktHK = FT_MARKET_STATUS_NONE;
		eMktHKFuture = FT_MARKET_STATUS_NONE;
		eMktUS = FT_MARKET_STATUS_NONE;
		eMktSH = FT_MARKET_STATUS_NONE;
		eMktSZ = FT_MARKET_STATUS_NONE;
	}
}NNGlobalState;