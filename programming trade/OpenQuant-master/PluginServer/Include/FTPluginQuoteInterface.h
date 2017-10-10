#pragma once
#include "FTPluginQuoteDefine.h"
#ifndef interface
#define interface struct
#endif


/*************************************************
Copyright: FUTU
Author: ysq
Date: 2015-03-18
Description: ����API�ͻص��ӿڶ���
**************************************************/

/**
*	������ݽӿ� IFTQuoteOperation���������ʵ�֣�ͨ����ѯIFTPluginCore::QueryFTInterface�õ�
*/
static const GUID IID_IFTQuoteOperation =
{ 0x9c65990c, 0x903, 0x4185, { 0x97, 0x12, 0x3e, 0xa7, 0xab, 0x34, 0xd, 0xc5 } };

interface IFTQuoteOperation
{
	//���鶨�ģ����ش�����
	virtual StockSubErrCode Subscribe_PriceBase(const GUID &guidPlugin, LPCWSTR wstrStockCode, StockMktType eType, bool bSubb, SOCKET sock) = 0;
	virtual StockSubErrCode Subscribe_OrderQueue(const GUID &guidPlugin, LPCWSTR wstrStockCode, StockMktType eType, bool bSubb, SOCKET sock) = 0;
	virtual StockSubErrCode Subscribe_Ticker(const GUID &guidPlugin, LPCWSTR wstrStockCode, StockMktType eType, bool bSubb, SOCKET sock) = 0;
	virtual StockSubErrCode Subscribe_RTData(const GUID &guidPlugin, LPCWSTR wstrStockCode, StockMktType eType, bool bSubb, SOCKET sock) = 0;
	virtual StockSubErrCode Subscribe_KLData(const GUID &guidPlugin, LPCWSTR wstrStockCode, StockMktType eType, bool bSubb, StockSubType eStockSubType, SOCKET sock) = 0;
	virtual StockSubErrCode Subscribe_BrokerQueue(const GUID &guidPlugin, LPCWSTR wstrStockCode, StockMktType eType, bool bSubb, SOCKET sock) = 0;

	virtual QueryDataErrCode QueryStockRTData(const GUID &guidPlugin, DWORD* pCookie, LPCWSTR wstrStockCode, StockMktType eType) = 0;
	virtual QueryDataErrCode QueryStockKLData(const GUID &guidPlugin, DWORD* pCookie, LPCWSTR wstrStockCode, StockMktType eType, int nKLType) = 0;

	//�����Ʊ���գ����һ��200��,ͨ�� IQuoteInfoCallback::OnReqStockSnapshot����
	virtual QueryDataErrCode QueryStockSnapshot(const GUID &guidPlugin, INT64 *arStockHash, int nStockNum, DWORD &dwReqCookie) = 0;
	virtual void  CancelQuerySnapshot(DWORD dwReqCookie) = 0;

	//֪ͨ���ӹر�
	virtual void  NotifyFTPluginSocketClosed(const GUID &guidPlugin, SOCKET sock) = 0;

	//�����鼰��鼯�ϵ��б�
	virtual QueryDataErrCode QueryPlatesetSubIDList(const GUID &guidPlugin, INT64 nPlatesetID, DWORD& dwCookie) = 0;
	virtual QueryDataErrCode QueryPlateStockIDList(const GUID &guidPlugin, INT64 nPlateID, DWORD& dwCookie) = 0;
};


/**
*	�������ݽӿ� IFTQuoteData���������ʵ�֣�ͨ����ѯIFTPluginCore::QueryFTInterface�õ�
*/
static const GUID IID_IFTQuoteData =
{ 0xb75073e3, 0xaa3a, 0x4717, { 0xac, 0xa2, 0x11, 0x94, 0xa1, 0x3, 0x78, 0xc7 } };

interface IFTQuoteData
{
	/**
	* ��ǰ�Ƿ���ĳֻ��Ʊĳ������λ(�����������)
	*/
	virtual bool   IsSubStockOneType(INT64 ddwStockHash, StockSubType eStockSubType) = 0;

	/**
	* ��ǰ�Ƿ���ʵʱ����
	*/
	virtual bool   IsRealTimeQuotes(INT64 ddwStockHash) = 0;

	/**
	* stock ��hashֵ, �ص��ӿڷ���
	*/
	virtual INT64  GetStockHashVal(LPCWSTR pstrStockCode, StockMktType eMkt) = 0;

	/**
	* stock ��hashֵ, �ص��ӿڷ���
	*/
	virtual bool  GetStockInfoByHashVal(INT64 ddwStockID, StockMktType& eMkt,
		wchar_t szStockCode[16], wchar_t szStockName[128], int* pLotSize = NULL,
		int* pSecurityType = NULL, int* pSubType = NULL, INT64* pnOwnerStockID = NULL) = 0;

	/**
	* ����������
	*/
	virtual bool   FillPriceBase(INT64 ddwStockHash, Quote_PriceBase* pInfo) = 0;

	/**
	* ���ʮ������
	*/
	virtual bool   FillOrderQueue(INT64 ddwStockHash, Quote_OrderItem* parOrder, int nCount) = 0;

	/**
	*	����ڴ�������ݣ�����ȥserver���µ����ݣ�����ʵ��fill�ĸ���
	*   ��nLastSequenceΪ0ʱ���õ����������������ݡ�nLastSequence��Ϊ0ʱΪ����nLastSequence��Ticker����
	*/
	virtual int    FillTickArr(INT64 ddwStockHash, PluginTickItem *parTickBuf, int nTickBufCount, INT64 nLastSequence = 0) = 0;

	/**
	* ����ʱ����
	*/
	virtual bool   FillRTData(INT64 ddwStockHash, Quote_StockRTData* &pQuoteRT, int& nCount) = 0;

	virtual BOOL   IsRTDataExist(INT64 ddwStockHash) = 0;

	virtual void   DeleteRTDataPointer(Quote_StockRTData* pRTData) = 0;

	/**
	* ���K������
	*/
	virtual BOOL   FillKLData(INT64 ddwStockHash, Quote_StockKLData* &pQuoteKL, int& nCount, int nKLType, int nRehabType) = 0;

	virtual BOOL   IsKLDataExist(INT64 ddwStockHash, int nKLType) = 0;

	virtual void   DeleteKLDataPointer(Quote_StockKLData* pRTData) = 0;

	/**
	*
	*/
	virtual void   CheckRemoveQuoteRT(INT64 ddwStockID) = 0;

	virtual void   CheckRemoveQuoteKL(INT64 ddwStockID, int nKLType) = 0;

	/**
	* �õ��������б�
	* @pszDateFrom: "YYYYMMDD"��ʽ��ΪNULL��Ĭ��ΪpszDateTo��ǰ��һ��
	* @pszDateTo: "YYYYMMDD"��ʽ��ΪNULL��Ĭ��Ϊ��ǰ����ʱ��
	* @nDateArr:  ����YYYYMMDD��ʽ�������������飬���շ����뽫���ص�����copyһ�ݱ�������
	* @nDateLen:  nDateArr���鳤��
	* @return:    ����true��false��ʾ�ɹ���ʧ�ܣ�ע�⼴ʹ�ɹ�nDateLenҲ�п���Ϊ0
	*/
	virtual bool GetTradeDateList(StockMktType mkt, LPCWSTR pszDateFrom, LPCWSTR pszDateTo, int* &narDateArr, int &nDateArrLen) = 0;

	//�õ���Ʊ�б�
	virtual bool GetStocksInfo(StockMktType mkt, PluginSecurityType eSecurityType, LPPluginStockInfo *&parInfo, int &nInfoArrLen) = 0;

	//�õ���Ȩ��Ϣ��Ϣ
	//����ֵ����ȫ�ɹ�����true, ���ֳɹ���ȫ��ʧ�ܶ�����false
	virtual bool  GetExRightInfo(INT64 *arStockHash, int nStockNum, PluginExRightItem* &arExRightInfo, int &nExRightInfoNum) = 0;

	//�õ���ʷK�� 
	//����ֵ�� ������������������δ���أ��򷵻�false�����ص�����������㶼����true
	//pszDateTimeFrom,pszDateTimeTo: ����Ϊnull, �����ַ�����ʽΪYYYY-MM-DD HH:MM:SS
	virtual bool  GetHistoryKLineTimeStr(StockMktType mkt, INT64 ddwStockHash, int nKLType, int nRehabType, LPCWSTR pszDateTimeFrom, LPCWSTR pszDateTimeTo, Quote_StockKLData* &arKLData, int &nKLNum) = 0;
	virtual bool  GetHistoryKLineTimestamp(StockMktType mkt, INT64 ddwStockHash, int nKLType, int nRehabType, INT64 nTimestampFrom, INT64 nTimestampTo, Quote_StockKLData *&arKLData, int &nKLNum) = 0;

	/**
	* dwTimeת��wstring ����+ʱ��
	*/
	virtual void TimeStampToStr(INT64 ddwStockHash, DWORD dwTime, wchar_t szTime[64]) = 0;

	/**
	* dwTimeת��wstring ����
	*/
	virtual void TimeStampToStrDate(INT64 ddwStockHash, DWORD dwTime, wchar_t szData[64]) = 0;

	/**
	* dwTimeת��wstring ʱ��
	*/
	virtual void TimeStampToStrTime(INT64 ddwStockHash, DWORD dwTime, wchar_t szTime[64]) = 0;

	/**
	* �õ���Ʊ�������
	* sockΪ0ʱ�õ��Ľ�����������ӵĶ�����Ϣ����sock��Ϊ0ʱ�����ص��ǵ�ǰsock���ӵĶ�����Ϣ
	*/
	virtual bool GetStockSubInfoList(Quote_SubInfo* &pSubInfoArr, int &nSubInfoLen, SOCKET sock = 0) = 0;

	/**
	* ���������������
	*/
	virtual bool FillBatchBasic(INT64 ddwStockHash, Quote_BatchBasic* pInfo) = 0;

	/**
	* ��Ʊ��������
	* bUnPush = true ������Ҫ����
	*/
	virtual bool RecordPushRequest(SOCKET sock, INT64 ddwStockHash, StockSubType nStockPushType, bool bUnPush = true) = 0;

	/**
	* ��鼯�ϵ��б�
	@GetPlatesetIDList ��鼯��ID
	@parID ID�б��أ��ɵ��÷�����ռ�, �ȴ�NULL�õ�nCount
	@nCount ID����
	@����ֵ��flase��ʾ���ݲ����ڣ� ��Ҫ����
	*/
	virtual bool GetPlatesetIDList(INT64 nPlatesetID, INT64* parID, int& nCount) = 0;

	/**
	* �������Ĺ�Ʊ�б�
	@GetPlateStockIDList ���Ĺ�Ʊ�б�ID
	@parID ID�б��أ��ɵ��÷�����ռ�, �ȴ�NULL�õ�nCount
	@nCount ID����
	@����ֵ��flase��ʾ���ݲ����ڣ� ��Ҫ����
	*/
	virtual bool GetPlateStockIDList(INT64 nPlateID, INT64* parID, int& nCount) = 0;

	/**
	* ���Ͷ���
	@GetBrokerQueueList  ���Ͷ����б�
	@parID ID�б��أ��ɵ��÷�����ռ�, �ȴ�NULL�õ�nCount
	@nCount ID����
	@����ֵ��flase��ʾ���ݲ����ڣ� ��Ҫ���ĵȴ�����push
	*/
	virtual bool GetBrokerQueueList(INT64 nStockID, Quote_BrokerItem* parID, int& nCount) = 0;

	/**
	* ��ȡȫ��״̬
	@GetNNGlobalState   
	@NNGlobalState  ���ض���
	@����ֵ��void 
	*/
	virtual void GetNNGlobalState(NNGlobalState* pState) = 0;
};

/**
*  �������֪ͨ����������ݱ仯�ӿ�
*/
interface IQuoteInfoCallback
{
	/**
	* ����������Ϣ�仯
	*/
	virtual void  OnChanged_PriceBase(INT64  ddwStockHash) = 0;

	/**
	* ʮ�����ݱ仯
	*/
	virtual void  OnChanged_OrderQueue(INT64 ddwStockHash) = 0;

	/**
	* ��ʱ���ݱ仯
	*/
	virtual void  OnChanged_RTData(INT64 ddwStockHash) = 0;

	/**
	* ��ʱ���ݱ仯
	*/
	virtual void  OnChanged_KLData(INT64 ddwStockHash, int nKLType) = 0;

	/**
	* ���Ͷ��б��
	*/
	virtual void  OnChanged_BrokerQueue(INT64 ddwStockHash) = 0;

	//�����Ʊ���շ���
	virtual void OnReqStockSnapshot(DWORD dwCookie, PluginStockSnapshot *arSnapshot, int nSnapshotNum) = 0;

	/**
	* �����鼯�ϵ������
	*/
	virtual void  OnReqPlatesetIDs(int nCSResult, DWORD dwCookie) = 0;

	/**
	* ������������
	*/
	virtual void  OnReqPlateStockIDs(int nCSResult, DWORD dwCookie) = 0;

	/**
	* ���ͻ�������
	*/
	virtual void  OnPushPriceBase(INT64  ddwStockHash, SOCKET sock) = 0;

	/**
	* ���Ͱ���
	*/
	virtual void  OnPushGear(INT64  ddwStockHash, SOCKET sock) = 0;

	/**
	* �������
	*/
	virtual void  OnPushTicker(INT64  ddwStockHash, SOCKET sock) = 0;

	/**
	* ����K��
	*/
	virtual void  OnPushKL(INT64  ddwStockHash, SOCKET sock, StockSubType eStockSubType) = 0;

	/**
	* ���ͷ�ʱ
	*/
	virtual void  OnPushRT(INT64  ddwStockHash, SOCKET sock) = 0;

	/**
	* ���;��Ͷ���
	*/
	virtual void  OnPushBrokerQueue(INT64  ddwStockHash, SOCKET sock) = 0;

	/**
	* �½���������
	*/
	virtual void  OnPushMarketNewTrade(StockMktType eMkt, INT64 ddwLastTradeStamp, INT64 ddwNewTradeStamp) = 0;

	/**
	* ά��push����alive����������
	*/
	virtual void  OnPushHeartBeat(SOCKET sock, UINT64 nTimeStampNow) = 0;
};

interface IQuoteKLCallback
{
	/**
	* �����ʱ�ص�
	*/
	virtual void OnQueryStockRTData(DWORD dwCookie, int nCSResult) = 0;

	/**
	* ����K�߻ص�
	*/
	virtual void OnQueryStockKLData(DWORD dwCookie, int nCSResult) = 0;
};

/**
*  �����ϱ��ӿ�
*/
static const GUID IID_IFTDataReport =
{ 0xdf2cba3e, 0x98c4, 0x4391, { 0x88, 0x23, 0x32, 0xc1, 0x7d, 0xea, 0x93, 0x1e } };

interface IFTDataReport
{
	/**
	* PLS�����ϱ�-CmdID
	*/
	virtual void PLSCmdIDReport(const GUID &guidPlugin, int nCmdID) = 0;
};

