#pragma once
#include <vector>

//�ο�"����.txt"

//Э��ID
enum
{
	PROTO_ID_QUOTE_MIN				= 1001,
	PROTO_ID_QT_GET_BASIC_PRICE		= 1001,  //����
	PROTO_ID_QT_GET_GEAR_PRICE		= 1002,	 //ʮ��
	PROTO_ID_QUOTE_MAX				= 1999,  

	PROTO_ID_TRADE_HK_MIN			= 6003,	  
	PROTO_ID_TDHK_PLACE_ORDER		= 6003,   //�µ�
	PROTO_ID_TDHK_SET_ORDER_STATUS	= 6004,   //����״̬����
	PROTO_ID_TDHK_CHANGE_ORDER		= 6005,   //�ĵ�
	PROTO_ID_TDHK_UNLOCK_TRADE		= 6006,	  //��������
	PROTO_ID_TDHK_QUERY_ACC_INFO	= 6007,	  //��ѯ�ʻ���Ϣ
	PROTO_ID_TDHK_QUERY_ORDER		= 6008,	  //��ѯ�۹ɶ����б�
	PROTO_ID_TDHK_QUERY_POSITION	= 6009,	  //��ѯ�۹ɳֲ�
	PROTO_ID_TRADE_HK_MAX			= 6999,    

	PROTO_ID_TRADE_US_MIN           = 7003,
	PROTO_ID_TDUS_PLACE_ORDER		= 7003,   //�µ�
	PROTO_ID_TDUS_SET_ORDER_STATUS	= 7004,   //����״̬����
	PROTO_ID_TDUS_CHANGE_ORDER		= 7005,   //�ĵ�

	PROTO_ID_TDUS_QUERY_ACC_INFO	= 7007,	  //��ѯ�ʻ���Ϣ
	PROTO_ID_TDUS_QUERY_ORDER		= 7008,	  //��ѯ���ɶ����б�
	PROTO_ID_TDUS_QUERY_POSITION	= 7009,	  //��ѯ�۹ɳֲ�
	PROTO_ID_TRADE_US_MAX			= 7999,    

};

#define KEY_REQ_PARAM	"ReqParam"
#define KEY_ACK_DATA	"RetData"

enum ProtoErrCode
{
	PROTO_ERR_NO_ERROR	= 0,

	PROTO_ERR_UNKNOWN_ERROR = 400,
	PROTO_ERR_VER_NOT_SUPPORT = 401,
	PROTO_ERR_STOCK_NOT_FIND = 402,
	PROTO_ERR_COMMAND_NOT_SUPPORT = 403,
	PROTO_ERR_PARAM_ERR = 404,

	PROTO_ERR_SERVER_BUSY	= 501,
	PROTO_ERR_SERVER_TIMEROUT = 502,
};

//////////////////////////////////////////////////////////////////////////
//ͨ��Э��ͷ��

struct ProtoHead
{
	int   nProtoVer;
	int   nProtoID;
	INT64 ddwErrCode;
	std::string strErrDesc;
};
