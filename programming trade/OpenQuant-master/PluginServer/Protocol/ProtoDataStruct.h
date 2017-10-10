#pragma once
#include <vector>

//�ο�"����.txt"

//Э��ID
enum
{
	PROTO_ID_QUOTE_MIN				= 1001,
	PROTO_ID_QT_GET_BASIC_PRICE		= 1001,  //����
	PROTO_ID_QT_GET_GEAR_PRICE		= 1002,	 //ʮ��
	PROTO_ID_QT_SUBSTOCK			= 1005,
	PROTO_ID_QT_UNSUBSTOCK			= 1006,
	PROTO_ID_QT_QueryStockSub		= 1007,
	PROTO_ID_QT_PushStockData		= 1008,
	PROTO_ID_QT_GET_RTDATA			= 1010,	 //��ʱ
	PROTO_ID_QT_GET_KLDATA			= 1011,  //K��
	PROTO_ID_QT_GET_TICKER			= 1012,  //���
	PROTO_ID_QT_GET_TRADE_DATE		= 1013,  //������
	PROTO_ID_QT_GET_STOCK_LIST		= 1014,  //��Ʊ��Ϣ
	PROTO_ID_QT_GET_SNAPSHOT		= 1015,  //�г�����
	PROTO_ID_QT_GET_BATCHBASIC		= 1023,  //��������
	PROTO_ID_QT_GET_HISTORYKL		= 1024,  //��ʷK��
	PROTO_ID_QT_GET_EXRIGHTINFO		= 1025,  //��Ȩ����
	PROTO_ID_QT_GET_PLATESETIDS		= 1026,  //��鼯�ϵ��б�
	PROTO_ID_QT_GET_PLATESUBIDS		= 1027,  //����µĹ�Ʊ�б�
	PROTO_ID_QT_GET_BROKER_QUEUE	= 1028,  //���Ͷ���
	PROTO_ID_QT_GET_GLOBAL_STATE	= 1029,  //ȫ��״̬ 

	PROTO_ID_PUSH_BATCHPRICE        = 1030,  //���ͱ���
	PROTO_ID_PUSH_GEARPRICE			= 1031,  //���Ͱ���
	PROTO_ID_PUSH_KLDATA			= 1032,  //����K��
	PROTO_ID_PUSH_TICKER			= 1033,  //�������
	PROTO_ID_PUSH_RTDATA			= 1034,  //���ͷ�ʱ
	PROTO_ID_PUSH_BROKER_QUEUE		= 1035,	 //���;��Ͷ���
	PROTO_ID_PUSH_HEART_BEAT		= 1036,	 //����

	PROTO_ID_QUOTE_MAX				= 1999,  

	PROTO_ID_TRADE_HK_MIN			= 6003,	  
	PROTO_ID_TDHK_PLACE_ORDER		= 6003,   //�µ�
	PROTO_ID_TDHK_SET_ORDER_STATUS	= 6004,   //����״̬����
	PROTO_ID_TDHK_CHANGE_ORDER		= 6005,   //�ĵ�
	PROTO_ID_TDHK_UNLOCK_TRADE		= 6006,	  //��������
	PROTO_ID_TDHK_QUERY_ACC_INFO	= 6007,	  //��ѯ�ʻ���Ϣ
	PROTO_ID_TDHK_QUERY_ORDER		= 6008,	  //��ѯ�۹ɶ����б�
	PROTO_ID_TDHK_QUERY_POSITION	= 6009,	  //��ѯ�۹ɳֲ�
	PROTO_ID_TDHK_QUERY_DEAL		= 6010,	  //��ѯ�۹ɳɽ���¼
	PROTO_ID_TRADE_HK_MAX			= 6999,    

	PROTO_ID_TRADE_US_MIN           = 7003,
	PROTO_ID_TDUS_PLACE_ORDER		= 7003,   //�µ�
	PROTO_ID_TDUS_SET_ORDER_STATUS	= 7004,   //����״̬����
	PROTO_ID_TDUS_CHANGE_ORDER		= 7005,   //�ĵ�

	PROTO_ID_TDUS_QUERY_ACC_INFO	= 7007,	  //��ѯ�ʻ���Ϣ
	PROTO_ID_TDUS_QUERY_ORDER		= 7008,	  //��ѯ���ɶ����б�
	PROTO_ID_TDUS_QUERY_POSITION	= 7009,	  //��ѯ���ɳֲ�
	PROTO_ID_TDUS_QUERY_DEAL		= 7010,	  //��ѯ���ɳɽ��б�
	PROTO_ID_TRADE_US_MAX			= 7999,    

};

#define KEY_REQ_PARAM	"ReqParam"
#define KEY_ACK_DATA	"RetData"
#define FIELD_KEY_HEAD  "head"
#define FIELD_KEY_BODY  "body"

enum ProtoErrCode
{
	PROTO_ERR_NO_ERROR	= 0,

	PROTO_ERR_UNKNOWN_ERROR = 400,   //δ֪����
	PROTO_ERR_VER_NOT_SUPPORT = 401,  //�汾�Ų�֧��
	PROTO_ERR_STOCK_NOT_FIND = 402,   //δ֪��Ʊ
	PROTO_ERR_COMMAND_NOT_SUPPORT = 403,
	PROTO_ERR_PARAM_ERR = 404,
	PROTO_ERR_FREQUENCY_ERR = 405,
	PROTO_ERR_MAXSUB_ERR = 406,
	PROTO_ERR_UNSUB_ERR = 407,
	PROTO_ERR_UNSUB_TIME_ERR = 408,

	PROTO_ERR_SERVER_BUSY	= 501,
	PROTO_ERR_SERVER_TIMEROUT = 502,
	PROTO_ERR_NETWORK = 503,
};

//////////////////////////////////////////////////////////////////////////
//ͨ��Э��ͷ��
#define  ProtoHead_Version  1 

struct ProtoHead
{
	int   nProtoVer;
	int   nProtoID;
	INT64 ddwErrCode;
	std::string strErrDesc;

	ProtoHead()
	{
		nProtoVer = ProtoHead_Version;
		nProtoID = 0;
		ddwErrCode = 0;
	}

};
