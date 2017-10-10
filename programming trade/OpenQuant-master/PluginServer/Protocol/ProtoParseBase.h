#pragma once
#include <vector>
#include "../JsonCpp/json.h"
#include "ProtoDataStruct.h"

enum EProtoFildType
{
	ProtoFild_Int32		= 1,
	ProtoFild_Int64		= 2,
	ProtoFild_StringA	= 3,
	ProtoFild_StringW	= 4,
	ProtoFild_Struct	= 5,
	ProtoFild_Vector	= 6,
};

typedef struct tagProtoField
{	
	BOOL		   bOptional;
	EProtoFildType eFieldType;
	union
	{
		int		*pInt32;
		INT64	*pInt64;
		std::string *pStrA;
		std::wstring *pStrW;
		void	*pStruct;
		void	*pVector;
	};
	std::string strFieldKey;
}PROTO_FIELD, *LP_PROTO_FIELD;

typedef std::vector<PROTO_FIELD>	VT_PROTO_FIELD;

//////////////////////////////////////////////////////////////////////////

class CProtoParseBase
{
public:
	CProtoParseBase();
	virtual ~CProtoParseBase();

	static bool	 ConvBuffer2Json(const char *pBuf, int nBufLen, Json::Value &jsnVal);
	static bool  ConvJson2String(const Json::Value &jsnVal, std::string &strOut, bool bAppendCRLF);
	static int	 GetProtoID(const Json::Value &jsnVal);
	
	//����������Э������ʵ��(������Ӧ)Э��Ĵ�����������
	virtual bool ParseJson_Req(const Json::Value &jsnVal) = 0;
	virtual bool ParseJson_Ack(const Json::Value &jsnVal) = 0;
	virtual bool MakeJson_Req(Json::Value &jsnVal) = 0;
	virtual bool MakeJson_Ack(Json::Value &jsnVal) = 0;

protected:
	//�������������ṹ��todo: discard
	bool ParseProtoFields(const Json::Value &jsnVal, const VT_PROTO_FIELD &vtField);
	bool MakeProtoFields(Json::Value &jsnVal, const VT_PROTO_FIELD &vtField);

	//���JSON��������JSON����
	//nLevel:��0��ʼ��ÿ������{[�����1; ������]}�����1
	bool ParseJsonProtoStruct(const Json::Value &jsnVal, bool bReqOrAck, const std::string &strStructName, void *pStruct, int nLevel = 0);
	bool MakeJsonProtoStruct(Json::Value &jsnVal, bool bReqOrAck, const std::string &strStructName, void *pStruct, int nLevel = 0, bool bObjectOrArray = true);

	//�ṹ��ĳ�Ա��ָ��(����ʱnameΪ��,nLevelΪ0)
	//Ŀǰ��������ʵ��Ӧ�û���һ��
	virtual void GetStructField4ParseJson(bool bReqOrAck, int nLevel, const std::string &strStructName, VT_PROTO_FIELD &vtField, void *pStruct){}
	virtual void GetStructField4MakeJson(bool bReqOrAck, int nLevel, const std::string &strStructName, VT_PROTO_FIELD &vtField, void *pStruct){}
	
	//����Ľṹ���Ա��ָ��(����ʱnameΪ��,nLevelΪ0�����Ƕ��㲻�������飻nJsnArrSize���ڳ�ʼ������ռ�)
	//�������麯����ʵ���������ڣ�����jsonʱ��Ҫ��������ռ䣬����jsonʱ���������Ѿ�׼����
	virtual void GetArrayField4ParseJson(bool bReqOrAck, int nLevel, const std::string &strArrayName, int nJsnArrSize, VT_PROTO_FIELD &vtField, void *pVector){}
	virtual void GetArrayField4MakeJson(bool bReqOrAck, int nLevel, const std::string &strArrayName, VT_PROTO_FIELD &vtField, void *pVector){}

	bool  FillFieldMembers(PROTO_FIELD &field, BOOL bOptional, EProtoFildType eFieldType, const std::string &strKey, void *pValue);
	void  GetProtoHeadField_Req(VT_PROTO_FIELD &vtField, const ProtoHead &head);
	void  GetProtoHeadField_Ack(VT_PROTO_FIELD &vtField, const ProtoHead &head);
	
protected:	
	//todo: discard
	bool ParseProtoHead_Req(const Json::Value &jsnVal, ProtoHead &head);
	bool ParseProtoHead_Ack(const Json::Value &jsnVal, ProtoHead &head);
	bool MakeProtoHead_Req(Json::Value &jsnVal, const ProtoHead &head); 
	bool MakeProtoHead_Ack(Json::Value &jsnVal, const ProtoHead &head); 

};