#pragma once
#include "string.h"
#include "../JsonCpp/json.h"

class CppJsonConv
{
public:
	CppJsonConv();
	virtual ~CppJsonConv();

	void  SetJsonValue(const Json::Value &jsnValue);

	bool GetInt64Value(const std::string &strUtfKey, INT64 &nRet);
	bool GetInt32Value(const std::string &strUtfKey, int &nRet);
	bool GetStringValueA(const std::string &strUtfKey,std::string &strRetUtf);
	bool GetStringValueW(const std::string &strUtfKey,std::wstring &strRet);
	bool GetJsonValue(const std::string &strUtfKey, Json::Value &jsnRet);

	bool GetArrItemInt64Value(int nIndex, INT64 &nRet);
	bool GetArrItemInt32Value(int nIndex, int &nRet);
	bool GetArrItemStringValueA(int nIndex, std::string &strRetUtf);
	bool GetArrItemStringValueW(int nIndex, std::wstring &strRet);
	bool GetArrItemJsonValue(int nIndex, Json::Value &jsnRet);

	bool SetInt64Value(const std::string &strUtfKey, INT64 nVal);
	bool SetInt32Value(const std::string &strUtfKey, int nVal);
	bool SetStringValueA(const std::string &strUtfKey, const std::string &strValUtf);	
	bool SetStringValueW(const std::string &strUtfKey, const std::wstring &strVal);	
	bool SetJsonValue(const std::string &strUtfKey, Json::Value &jsnValue);

	bool SetArrItemInt64Value(int nIndex, INT64 nVal);
	bool SetArrItemInt32Value(int nIndex, int nVal);
	bool SetArrItemStringValueA(int nIndex, const std::string &strValUtf);
	bool SetArrItemStringValueW(int nIndex, const std::wstring &strVal);
	bool SetArrItemJsonValue(int nIndex, Json::Value &jsnValue);


protected:
	Json::Value *m_pJsonVal;
};
