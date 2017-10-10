#pragma once
#include <afxmt.h>
#include <map>
#include <vector>
#include "Include/FTPluginQuoteInterface.h"

interface IPluginNetEvent
{
	virtual void OnReceive(SOCKET sock) = 0;
	virtual void OnSend(SOCKET sock) = 0;
	virtual void OnClose(SOCKET sock) = 0;
};

class CPluginNetwork
{
public:
	CPluginNetwork();
	~CPluginNetwork();

	void InitNetwork(IPluginNetEvent *pEvtSink);
	void UninitNetwork();
	void SendData(SOCKET sock, const char *pBuf, int nBufLen);
	void PushData(const char *pBuf, int nBufLen);
	bool GetRecvData(SOCKET sock, const char *&pBuf, int &nBufLen);	
	int  GetConnectNum();
	

protected:
	struct TransDataInfo
	{
		int	   nBufferID;
		BOOL   bToFree;
		BOOL   bSendRecvFinish;
		BOOL   bHasDeliver;
		int	   nRealBufLen;
		WSABUF buffer;
		TransDataInfo()
		{
			nBufferID = 0;
			bToFree = FALSE;
			bSendRecvFinish = FALSE;
			bHasDeliver = FALSE;
			buffer.len = 0;
			buffer.buf = NULL;
		}
	};
	typedef std::vector<TransDataInfo*>		VT_TRANS_DATA;
	typedef std::map<SOCKET, VT_TRANS_DATA> MAP_SOCK_TRANS_DATA;


	struct SockRuntimeInfo
	{
		SOCKET	sock;
		WSAOVERLAPPED overlap;
		WSAEVENT hEventHandle;
		VT_TRANS_DATA vtDeliverData;
		SockRuntimeInfo()
		{
			sock = NULL;			
			hEventHandle = NULL;
			ZeroMemory(&overlap, sizeof(WSAOVERLAPPED));
		}
	};
	typedef std::map<SOCKET, SockRuntimeInfo*> MAP_SOCK_RTINFO;

	typedef std::vector<SOCKET>	VT_CONNECT_SOCK;

protected:
	//�Ѿ������������
	void FreeSendFinishBuf(SOCKET sock);
	void FreeRecvFinishBuf(SOCKET sock);
	void ClearFreeBuf(VT_TRANS_DATA &vtData);
	
	//�շ�����(�շ��߳��е���)
	void ClearSocketSendData(SOCKET sock);
	void ClearSocketRecvData(SOCKET sock);

	//����ʼ��ʱ����(���߳�)
	void ClearAllSockRTInfo();
	void ClearSockRTInfo(MAP_SOCK_RTINFO &mapRTInfo);
	void ClearAllSockBuf();
	void ClearSockTransData(MAP_SOCK_TRANS_DATA &mapTransData);

	//�µ�����:�����߳�
	void SetNewConnectSocket(SOCKET sock);

	//�Ͽ�����:�շ��߳�
	void NotifySocketClosed(SOCKET sock, LPCTSTR pstrLogInfo = NULL);

	//����Ͽ�������:�����߳�
	void ClearClosedSocket();

	bool IsSocketInDisconList(SOCKET sock);

	
	//�߳�
	static DWORD WINAPI ThreadAccept(LPVOID lParam);
	void AccpetLoop();

	static DWORD WINAPI ThreadSend(LPVOID lParam);
	void SendLoop();

	static DWORD WINAPI ThreadRecv(LPVOID lParam);
	void RecvLoop();

protected:
	bool	m_bInit;
	HANDLE	m_hEvtNotifyExit;	
	IPluginNetEvent *m_pEvtNotify;
	

	int		m_nNextSendBufID;
	int		m_nNextRecvBufID;

	HANDLE	m_hThreadAccept;
	HANDLE	m_hThreadSend;
	HANDLE	m_hThreadRecv;	

	CCriticalSection	m_csAccpt;
	VT_CONNECT_SOCK		m_vtConnSock;
	VT_CONNECT_SOCK		m_vtDisconSock;

	CCriticalSection	m_csRecv;
	MAP_SOCK_RTINFO		m_mapRecvingInfo;
	MAP_SOCK_TRANS_DATA	m_mapRecvedData;
	
	CCriticalSection	m_csSend;
	MAP_SOCK_RTINFO		m_mapSendingInfo;
	MAP_SOCK_TRANS_DATA	m_mapToSendData;
};
