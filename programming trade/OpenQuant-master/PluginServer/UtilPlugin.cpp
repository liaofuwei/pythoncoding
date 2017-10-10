#include "stdafx.h"
#include "UtilPlugin.h"
#include "UsTimezone.h"

int  UtilPlugin::GetMarketTimezone(StockMktType eMkt, int nTimestamp)
{
	switch (eMkt)
	{
	case StockMkt_HK:
	case StockMkt_Feature_Old:
	case StockMkt_Feature_New:
	case StockMkt_SH:
	case StockMkt_SZ:
		return 8;
		break;
	case StockMkt_US:
		return UsTimezone::GetTimestampTimezone(nTimestamp);
		break;
	default:
		assert(0);
		return 0;
		break;
	}
}

int  UtilPlugin::GetMarketTimezone2(StockMktType eMkt, int nYear, int nMonth, int nDay)
{
	switch (eMkt)
	{
	case StockMkt_HK:
	case StockMkt_Feature_Old:
	case StockMkt_Feature_New:
	case StockMkt_SH:
	case StockMkt_SZ:
		return 8;
		break;
	case StockMkt_US:
		return UsTimezone::GetTMStructTimezone(nYear, nMonth, nDay);
			break;
	default:
		assert(0);
		return 0;
		break;
	}
}

std::string UtilPlugin::FormatMktTimestamp(int nTimestamp, StockMktType eMkt, FormatTimeType eFmtType)
{
	int nTimezone = GetMarketTimezone(eMkt, nTimestamp);
	std::string strFmt = FormatTime::FormatTimestamp(nTimestamp, nTimezone, eFmtType);
	return strFmt;
}

std::string UtilPlugin::GetErrStrByCode(QueryDataErrCode eCode)
{
	CString strErr;
	switch (eCode)
	{
	case QueryData_Suc:
		strErr = "Success!";
		break;
	case QueryData_FailUnknown:
		strErr = L"��������δ֪����";
		break;
	case QueryData_FailMaxSubNum:
		strErr = L"���󵽴�����ѯ��";
		break;
	case QueryData_FailCodeNoFind:
		strErr = L"�������û�ҵ�";
		break;
	case QueryData_FailGuidNoFind:
		strErr = L"������GUID����";
		break;
	case QueryData_FailNoImplInf:
		strErr = L"��������ӿ�δ���";
		break;
	case QueryData_FailFreqLimit:
		strErr = L"�����ѯƵ�����Ƶ���ʧ��";
		break;
	case QueryData_FailNetwork:
		strErr = L"���������쳣������ʧ��";
		break;
	case QueryData_FailErrParam:
		strErr = L"��������";
		break;
	default:
		strErr.Format(_T("����δ֪����:%d"), (int)eCode);
		break;
	}

	std::string strTmp;
	CA::Unicode2UTF(CT2CW(strErr), strTmp);

	return strTmp;
}

ProtoErrCode UtilPlugin::ConvertErrCode(QueryDataErrCode eCode)
{
	ProtoErrCode eRet = PROTO_ERR_UNKNOWN_ERROR;
	switch (eCode)
	{
	case QueryData_Suc:
		eRet = PROTO_ERR_NO_ERROR;
		break;
	case QueryData_FailUnknown:
		eRet = PROTO_ERR_UNKNOWN_ERROR;
		break;
	case QueryData_FailMaxSubNum:
		eRet = PROTO_ERR_MAXSUB_ERR;
		break;
	case QueryData_FailCodeNoFind:
		eRet = PROTO_ERR_STOCK_NOT_FIND;
		break;
	case QueryData_FailGuidNoFind:
		eRet = PROTO_ERR_PARAM_ERR;
		break;
	case QueryData_FailNoImplInf:
		break;
	case QueryData_FailFreqLimit:
		eRet = PROTO_ERR_FREQUENCY_ERR;
		break;
	case QueryData_FailNetwork:
		eRet = PROTO_ERR_NETWORK;
		break;
	case QueryData_FailErrParam:
		eRet = PROTO_ERR_PARAM_ERR;
		break;
	default:
		break;
	}
	return eRet;
}
