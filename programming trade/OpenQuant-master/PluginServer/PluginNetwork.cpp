#include "stdafx.h"
#include "PluginNetwork.h"
#include <algorithm>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

#define TIMEOUT_MILISECONDS		1
#define MIN_BUF_ALLOC_SIZE		1024*1024

CPluginNetwork::CPluginNetwork()
{
	m_bInit = false;
	m_hEvtNotifyExit = NULL;
	m_pEvtNotify = NULL;
	m_nNextSendBufID = 1;
	m_nNextRecvBufID = 1;
	m_hThreadAccept = NULL;
	m_hThreadSend = NULL;
	m_hThreadRecv = NULL;
}

CPluginNetwork::~CPluginNetwork()
{
	UninitNetwork();
}

void CPluginNetwork::InitNetwork(IPluginNetEvent *pEvtSink)
{
	CHECK_RET(pEvtSink, NORET);

	if ( m_bInit )
		return ;

	m_bInit = true;

	m_pEvtNotify = pEvtSink;

	m_hEvtNotifyExit = ::CreateEvent(NULL, TRUE, FALSE, NULL);

	DWORD nThreadID1 = 0;
	m_hThreadAccept = ::CreateThread(NULL, 0, ThreadAccept, (void *)this, 0, &nThreadID1);
	
	DWORD nThreadID2 = 0;
	m_hThreadSend = ::CreateThread(NULL, 0, ThreadSend, (void *)this, 0, &nThreadID2);

	DWORD nThreadID3 = 0;
	m_hThreadRecv = ::CreateThread(NULL, 0, ThreadRecv, (void *)this, 0, &nThreadID3);	
}

void CPluginNetwork::UninitNetwork()
{
	if ( !m_bInit )
		return;

	m_bInit = false;
	
	SetEvent(m_hEvtNotifyExit);

	Sleep(10);

	DWORD dwExitCode = 0;	
	if ( GetExitCodeThread(m_hThreadAccept, &dwExitCode) && dwExitCode == STILL_ACTIVE )
	{
		DWORD dwRet = WaitForSingleObject(m_hThreadAccept, TIMEOUT_MILISECONDS * 5);
		if ( dwRet == WAIT_TIMEOUT )
		{
			//CHECK_OP(false, NOOP);
			::TerminateThread(m_hThreadAccept, -1);
		}
	}
	if ( m_hThreadAccept )
	{
		CloseHandle(m_hThreadAccept);
		m_hThreadAccept = NULL;
	}

	DWORD dwExitCode2 = 0;
	if ( GetExitCodeThread(m_hThreadSend, &dwExitCode2) && dwExitCode2 == STILL_ACTIVE )
	{
		DWORD dwRet = WaitForSingleObject(m_hThreadSend, TIMEOUT_MILISECONDS * 5);
		if ( dwRet == WAIT_TIMEOUT )
		{
			//CHECK_OP(false, NOOP);
			::TerminateThread(m_hThreadSend, -1);
		}
	}
	if ( m_hThreadSend )
	{
		CloseHandle(m_hThreadSend);
		m_hThreadSend = NULL;
	}

	DWORD dwExitCode3 = 0;	
	if ( GetExitCodeThread(m_hThreadRecv, &dwExitCode3) && dwExitCode3 == STILL_ACTIVE )
	{
		DWORD dwRet = WaitForSingleObject(m_hThreadRecv, TIMEOUT_MILISECONDS * 5);
		if ( dwRet == WAIT_TIMEOUT )
		{
			//CHECK_OP(false, NOOP);
			::TerminateThread(m_hThreadRecv, -1);
		}
	}	
	if ( m_hThreadRecv )
	{
		CloseHandle(m_hThreadRecv);
		m_hThreadRecv = NULL;
	}

	m_pEvtNotify = NULL;	
	ClearAllSockRTInfo();
	ClearAllSockBuf();	

	CloseHandle(m_hEvtNotifyExit);
	m_hEvtNotifyExit = NULL;
}

void CPluginNetwork::SendData(SOCKET sock, const char *pBuf, int nBufLen)
{
	CHECK_RET(sock != INVALID_SOCKET && pBuf && (nBufLen > 0), NORET);

	CSingleLock lock(&m_csSend, TRUE);

	VT_TRANS_DATA &vtTransData = m_mapToSendData[sock];
	VT_TRANS_DATA::iterator itData = vtTransData.begin();
	TransDataInfo *pBufToUse = NULL;
	for ( ; itData != vtTransData.end(); ++itData )
	{
		TransDataInfo *pData = *itData;
		CHECK_OP(pData, continue);
		if ( pData->bHasDeliver || pData->bToFree
			|| int(pData->nRealBufLen - pData->buffer.len) < nBufLen )
		{
			continue;
		}
		else
		{
			pBufToUse = pData;
			break;
		}
	}

	if ( pBufToUse == NULL )
	{
		pBufToUse = new TransDataInfo;
		CHECK_RET(pBufToUse, NORET);

		pBufToUse->nBufferID = m_nNextSendBufID++;
		pBufToUse->nRealBufLen = max(nBufLen, MIN_BUF_ALLOC_SIZE);
		pBufToUse->buffer.len = 0;
		pBufToUse->buffer.buf = new char[pBufToUse->nRealBufLen];

		if ( pBufToUse->buffer.buf == NULL )
		{
			CHECK_OP(false, NOOP);
			delete pBufToUse;

			theLog << "CPluginNetwork::SendData - new TransDataInfo Error, Len=" << pBufToUse->nRealBufLen << endLog;
			return;
		}

		vtTransData.push_back(pBufToUse);
	}
	
	CHECK_RET(pBufToUse && pBufToUse->buffer.buf, NORET);
	memcpy(pBufToUse->buffer.buf + pBufToUse->buffer.len, pBuf, nBufLen);
	pBufToUse->buffer.len += nBufLen;
}

void CPluginNetwork::PushData(const char *pBuf, int nBufLen)
{
	CHECK_RET(pBuf && (nBufLen > 0), NORET);

	CSingleLock lock(&m_csSend, TRUE);

	MAP_SOCK_RTINFO::iterator itsock = m_mapSendingInfo.begin();
	for ( ; itsock != m_mapSendingInfo.end(); ++itsock )
	{
		SOCKET sock = itsock->first;
		VT_TRANS_DATA &vtTransData = m_mapToSendData[sock];
		VT_TRANS_DATA::iterator itData = vtTransData.begin();
		TransDataInfo *pBufToUse = NULL;
		for ( ; itData != vtTransData.end(); ++itData )
		{
			TransDataInfo *pData = *itData;
			CHECK_OP(pData, continue);
			if ( pData->bHasDeliver || pData->bToFree
				|| int(pData->nRealBufLen - pData->buffer.len) < nBufLen )
			{
				continue;
			}
			else
			{
				pBufToUse = pData;
				break;
			}
		}

		if ( pBufToUse == NULL )
		{
			pBufToUse = new TransDataInfo;
			CHECK_RET(pBufToUse, NORET);

			pBufToUse->nBufferID = m_nNextSendBufID++;
			pBufToUse->nRealBufLen = max(nBufLen, MIN_BUF_ALLOC_SIZE);
			pBufToUse->buffer.len = 0;
			pBufToUse->buffer.buf = new char[pBufToUse->nRealBufLen];

			if ( pBufToUse->buffer.buf == NULL )
			{
				CHECK_OP(false, NOOP);
				delete pBufToUse;
				return;
			}

			vtTransData.push_back(pBufToUse);
		}

		CHECK_OP(pBufToUse && pBufToUse->buffer.buf, continue);
		memcpy(pBufToUse->buffer.buf + pBufToUse->buffer.len, pBuf, nBufLen);
		pBufToUse->buffer.len += nBufLen;
	}	
}

bool CPluginNetwork::GetRecvData(SOCKET sock, const char *&pBuf, int &nBufLen)
{
	CHECK_RET(sock && sock != INVALID_SOCKET, false);

	CSingleLock lock(&m_csRecv, TRUE);

	MAP_SOCK_TRANS_DATA::iterator itMapSock = m_mapRecvedData.find(sock);
	if (itMapSock != m_mapRecvedData.end())
	{
		VT_TRANS_DATA &vtData = itMapSock->second;
		ClearFreeBuf(vtData);

		VT_TRANS_DATA::iterator itData = vtData.begin();
		for (; itData != vtData.end(); ++itData)
		{
			TransDataInfo *pData = *itData;
			CHECK_OP(pData, continue);
			if (pData->bSendRecvFinish && !pData->bToFree)
			{
				pBuf = pData->buffer.buf;
				nBufLen = pData->buffer.len;
				pData->bToFree = TRUE;
				return true;
			}
		}
	}
	return false;
}

void  CPluginNetwork::SetNewConnectSocket(SOCKET sock)
{
	CHECK_RET(sock != INVALID_SOCKET, NORET);

	CSingleLock lock(&m_csAccpt, TRUE);	

	VT_CONNECT_SOCK::iterator it_find = std::find(
		m_vtConnSock.begin(), m_vtConnSock.end(), sock); 
	
	//����accept������һֱ���ذ��� ���п�����socket ��¼��������,����������ݴ���
	if (it_find != m_vtConnSock.end())
	{
		CHECK_OP(false, NOOP);	
		theLog << "CPluginNetwork::SetNewConnectSocket Error - SOCKET Exist:" << (UINT)sock << " Is Wait To Close:" << (int)IsSocketInDisconList(sock) << endLog;

		lock.Unlock();
		{
			NotifySocketClosed(sock, L"SetNewConnectSocket - Reused Socket");
			ClearClosedSocket();
		}
		lock.Lock();
	}
	m_vtConnSock.push_back(sock);
	lock.Unlock();
	
	//lock
	{
		CSingleLock lock(&m_csSend, TRUE);	
		SockRuntimeInfo *pRTSend = new SockRuntimeInfo;
		if ( pRTSend )
		{
			pRTSend->sock = sock;
			pRTSend->hEventHandle = WSACreateEvent();
			pRTSend->overlap.hEvent = pRTSend->hEventHandle;
			CSingleLock lock(&m_csSend, TRUE);
			m_mapSendingInfo[sock] = pRTSend;
		}
	}
	
	//lock
	{
		CSingleLock lock(&m_csRecv, TRUE);	
		SockRuntimeInfo *pRTRecv = new SockRuntimeInfo;
		if ( pRTRecv )
		{
			pRTRecv->sock = sock;
			pRTRecv->hEventHandle = WSACreateEvent();
			pRTRecv->overlap.hEvent = pRTRecv->hEventHandle;
			CSingleLock lock(&m_csRecv, TRUE);
			m_mapRecvingInfo[sock] = pRTRecv;
		}
	}
}

void  CPluginNetwork::NotifySocketClosed(SOCKET sock, LPCTSTR pstrLogInfo)
{
	CSingleLock lock(&m_csAccpt, TRUE);
	if ( m_pEvtNotify )
		m_pEvtNotify->OnClose(sock);
	m_vtDisconSock.push_back(sock);
	if (pstrLogInfo)
	{
		theLog << "CPluginNetwork::NotifySocketClosed socket=" << (UINT32)sock << " Reason:" << pstrLogInfo << endLog;
	}
}
bool CPluginNetwork::IsSocketInDisconList(SOCKET sock)
{
	CSingleLock lock(&m_csAccpt, TRUE);
	return std::find(m_vtDisconSock.begin(), m_vtDisconSock.end(), sock) != m_vtDisconSock.end();
}

void CPluginNetwork::ClearClosedSocket()
{
	VT_CONNECT_SOCK	vtDisconSock;
	VT_CONNECT_SOCK::iterator it_sock;
	{
		CSingleLock lock(&m_csAccpt, TRUE);
		vtDisconSock.swap(m_vtDisconSock);

		if ( vtDisconSock.empty() )
		{
			lock.Unlock();
			return ;
		}

		it_sock = vtDisconSock.begin();
		for ( ; it_sock != vtDisconSock.end(); ++it_sock )
		{
			SOCKET sock = *it_sock;

			VT_CONNECT_SOCK::iterator it_find = std::find(
				m_vtConnSock.begin(), m_vtConnSock.end(), sock);

			if ( it_find != m_vtConnSock.end() )
			{		
				m_vtConnSock.erase(it_find);
				closesocket(sock);
			}
		}
		lock.Unlock();
	}

	it_sock = vtDisconSock.begin();	
	{
		CSingleLock lock(&m_csRecv, TRUE);
		for ( ; it_sock != vtDisconSock.end(); ++it_sock )
		{
			SOCKET sock = *it_sock;
			ClearSocketRecvData(sock);
		}
		lock.Unlock();
	}

	it_sock = vtDisconSock.begin();	
	{
		CSingleLock lock(&m_csSend, TRUE);
		for ( ; it_sock != vtDisconSock.end(); ++it_sock )
		{
			SOCKET sock = *it_sock;
			ClearSocketSendData(sock);
		}
		lock.Unlock();
	}	
}

int  CPluginNetwork::GetConnectNum()
{
	return (int)m_vtConnSock.size();
}

void CPluginNetwork::FreeSendFinishBuf(SOCKET sock)
{
	MAP_SOCK_TRANS_DATA::iterator it_find = m_mapToSendData.find(sock);
	if ( it_find != m_mapToSendData.end() )
	{
		ClearFreeBuf(it_find->second);
	}	
}

void CPluginNetwork::FreeRecvFinishBuf(SOCKET sock)
{
	MAP_SOCK_TRANS_DATA::iterator it_find = m_mapRecvedData.find(sock);
	if ( it_find != m_mapRecvedData.end() )
	{
		ClearFreeBuf(it_find->second);
	}
}

void CPluginNetwork::ClearFreeBuf(VT_TRANS_DATA &vtData)
{
	VT_TRANS_DATA::iterator it = vtData.begin();
	for ( ; it != vtData.end(); )
	{
		TransDataInfo *pData = *it;
		if ( pData == NULL )
		{
			++it;
			CHECK_OP(false, continue);
		}
		
		if ( pData->bToFree )
		{
			delete []pData->buffer.buf;
			delete pData;
			it = vtData.erase(it);
		}
		else
		{
			++it;
		}
	}
}

void CPluginNetwork::ClearSocketSendData(SOCKET sock)
{
	MAP_SOCK_RTINFO::iterator itRTInfo = m_mapSendingInfo.find(sock);
	if ( itRTInfo != m_mapSendingInfo.end() )
	{
		SockRuntimeInfo *pRTInfo = itRTInfo->second;
		m_mapSendingInfo.erase(itRTInfo);		
		if ( pRTInfo )
		{
			WSACloseEvent(pRTInfo->hEventHandle);			
			delete pRTInfo;
		}
	}

	MAP_SOCK_TRANS_DATA::iterator itSockData = m_mapToSendData.find(sock);
	if ( itSockData != m_mapToSendData.end() )
	{
		VT_TRANS_DATA &vtData = itSockData->second;
		VT_TRANS_DATA::iterator itData = vtData.begin();
		for ( ; itData != vtData.end(); ++itData )
		{
			TransDataInfo *pData = *itData;
			CHECK_OP(pData, continue);
			delete []pData->buffer.buf;
			delete pData;
		}

		vtData.clear();
		m_mapToSendData.erase(itSockData);
	}	
}


void CPluginNetwork::ClearSocketRecvData(SOCKET sock)
{
	MAP_SOCK_RTINFO::iterator itRTInfo = m_mapRecvingInfo.find(sock);
	if ( itRTInfo != m_mapRecvingInfo.end() )
	{
		SockRuntimeInfo *pRTInfo = itRTInfo->second;
		m_mapRecvingInfo.erase(itRTInfo);		
		if ( pRTInfo )
		{
			WSACloseEvent(pRTInfo->hEventHandle);			
			delete pRTInfo;
		}
	}

	MAP_SOCK_TRANS_DATA::iterator itTransData = m_mapRecvedData.find(sock);
	if ( itTransData != m_mapRecvedData.end() )
	{
		VT_TRANS_DATA &vtData = itTransData->second;
		VT_TRANS_DATA::iterator itData = vtData.begin();
		for ( ; itData != vtData.end(); ++itData )
		{
			TransDataInfo *pData = *itData;
			CHECK_OP(pData, continue);
			delete []pData->buffer.buf;
			delete pData;
		}

		vtData.clear();
		m_mapRecvedData.erase(itTransData);
	}	
}


void CPluginNetwork::ClearAllSockRTInfo()
{
	//lock
	{
		CSingleLock lock(&m_csSend, TRUE);
		ClearSockRTInfo(m_mapSendingInfo);
	}
	
	//lock
	{
		CSingleLock lock(&m_csRecv, TRUE);
		ClearSockRTInfo(m_mapRecvingInfo);
	}
}

void CPluginNetwork::ClearSockRTInfo(MAP_SOCK_RTINFO &mapRTInfo)
{
	MAP_SOCK_RTINFO::iterator it = mapRTInfo.begin();
	for ( ; it != mapRTInfo.end(); ++it )
	{
		SockRuntimeInfo *pRTInfo = it->second;
		CHECK_OP(pRTInfo, continue);
		WSACloseEvent(pRTInfo->hEventHandle);
		delete pRTInfo;
	}

	mapRTInfo.clear();
}

void CPluginNetwork::ClearAllSockBuf()
{
	CHECK_OP(m_mapSendingInfo.empty() && m_mapRecvingInfo.empty(), NOOP);
	ClearSockTransData(m_mapToSendData);
	ClearSockTransData(m_mapRecvedData);
}

void CPluginNetwork::ClearSockTransData(MAP_SOCK_TRANS_DATA &mapTransData)
{
	MAP_SOCK_TRANS_DATA::iterator itSockData = mapTransData.begin();
	for ( ; itSockData != mapTransData.end(); ++itSockData )	
	{
		VT_TRANS_DATA &vtData = itSockData->second;
		VT_TRANS_DATA::iterator itData = vtData.begin();
		for ( ; itData != vtData.end(); ++itData )
		{
			TransDataInfo *pData = *itData;
			CHECK_OP(pData, continue);
			delete []pData->buffer.buf;
			delete pData;
		}

		vtData.clear();
	}
}

DWORD WINAPI CPluginNetwork::ThreadAccept(LPVOID lParam)
{
	CPluginNetwork *pThis = (CPluginNetwork*)lParam;
	CHECK_RET(pThis, 0);
	pThis->AccpetLoop();
	return 0;
}

void CPluginNetwork::AccpetLoop()
{	
	theLog << "CPluginNetwork::AccpetLoop begin ..." << endLog;

	//�˿�
	CString strCfg = CA::GetMoudleFolder(AfxGetInstanceHandle());
	strCfg.Append(L"config.ini");
	int nPort = 11111;
	CString strIP = _T("127.0.0.1");
	if ( PathFileExists(strCfg) )
	{
		nPort = ::GetPrivateProfileInt(L"pluginserver", L"port", nPort, strCfg);
		TCHAR chIP[16]={0};
		::GetPrivateProfileString(L"pluginserver", L"ip", _T(""), chIP, _countof(chIP)-1,  strCfg);

		if (_tcslen(chIP))
		{ 
			strIP = chIP; 
		}
	}	

	//bind
	SOCKET sock_lstn = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

	BOOL opt = TRUE;
	setsockopt(sock_lstn, SOL_SOCKET, SO_DONTLINGER, (char*)&opt, sizeof(opt));
	setsockopt(sock_lstn, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt));

	sockaddr_in addr_local;
	addr_local.sin_family = AF_INET;
	addr_local.sin_port = htons((u_short)nPort);
	addr_local.sin_addr.s_addr = inet_addr(CT2CA(strIP));
	bind(sock_lstn, (SOCKADDR*)&addr_local, sizeof(addr_local));

	//listen
	int nListenRet = listen(sock_lstn, 5);
	if ( SOCKET_ERROR == nListenRet )
	{
		CHECK_OP(false, NOOP);
		closesocket(sock_lstn);
		return;
	}	

	WSAEVENT hAccpt = WSACreateEvent();	
	
	//accept		
	while ( true )
	{
		if ( WaitForSingleObject(m_hEvtNotifyExit, TIMEOUT_MILISECONDS) == WAIT_OBJECT_0 )
		{
			CloseHandle(m_hThreadAccept);
			m_hThreadAccept = NULL;
			theLog << "CPluginNetwork::AccpetLoop wait exit" << endLog;
			break;
		}
		//�л��߳�
		Sleep(0);

		ClearClosedSocket();

		WSAResetEvent(hAccpt);
		WSAEventSelect(sock_lstn, hAccpt, FD_ACCEPT);
		DWORD dwRet = WaitForSingleObject((HANDLE)hAccpt, TIMEOUT_MILISECONDS);

		if (WAIT_OBJECT_0 == dwRet )
		{
			sockaddr_in addr_client;
			int addr_len = sizeof(addr_client);
			SOCKET sock_accpt = accept(sock_lstn, (sockaddr*)&addr_client, &addr_len);
			if ( sock_accpt != INVALID_SOCKET )
			{				
				//���������һ�Σ� ����SetNewConnectSocketʧ��
				ClearClosedSocket();

				if (GetConnectNum() >= WSA_MAXIMUM_WAIT_EVENTS)
				{
					theLog << "CPluginNetwork::AccpetLoop Error - Only Support Conntect Count = " << (int)WSA_MAXIMUM_WAIT_EVENTS << endLog;

					closesocket(sock_accpt);
				}
				else
				{
					BOOL opt = TRUE;
					setsockopt(sock_accpt, SOL_SOCKET, SO_DONTLINGER, (char*)&opt, sizeof(opt));
					SetNewConnectSocket(sock_accpt);
				}
					
			}
			continue;
		}
		else if ( WAIT_TIMEOUT == dwRet )
		{
			continue;
		}
		else
		{
			//WAIT_FAILED �ݴ��߼�: event ���������⣿ 
			CHECK_OP(false, break);
			WSACloseEvent(hAccpt);
			hAccpt = WSACreateEvent();
			theLog << "CPluginNetwork::AccpetLoop Error : ret = " << dwRet << endLog; 
		}
	}

	WSACloseEvent(hAccpt);
	closesocket(sock_lstn);

	theLog << "CPluginNetwork::AccpetLoop end..." << endLog;
}

DWORD WINAPI CPluginNetwork::ThreadSend(LPVOID lParam)
{
	CPluginNetwork *pThis = (CPluginNetwork*)lParam;
	CHECK_RET(pThis, 0);
	pThis->SendLoop();
	return 0;
}

void CPluginNetwork::SendLoop()
{
	DWORD dwNextWaitExitTime = 0;
	WSAEVENT  hEvtCheckClose = WSACreateEvent();

	while (true)
	{
		if (WaitForSingleObject(m_hEvtNotifyExit, dwNextWaitExitTime) == WAIT_OBJECT_0)
		{
			CloseHandle(m_hThreadSend);
			m_hThreadSend = NULL;
			return;
		}
		//�л��߳�
		Sleep(0);
		dwNextWaitExitTime = TIMEOUT_MILISECONDS;

		//Ͷ��������
		CSingleLock lock(&m_csSend, TRUE);
		std::vector<SOCKET> vtSocketError;
		MAP_SOCK_RTINFO::iterator itRTInfo = m_mapSendingInfo.begin();
		for (; itRTInfo != m_mapSendingInfo.end(); ++itRTInfo)
		{
			SOCKET sock = itRTInfo->first;
			SockRuntimeInfo *pInfo = itRTInfo->second;
			CHECK_OP(pInfo, continue);

			if (!pInfo->vtDeliverData.empty())
				continue;

			MAP_SOCK_TRANS_DATA::iterator itToSend = m_mapToSendData.find(sock);
			if (itToSend == m_mapToSendData.end())
				continue;

			VT_TRANS_DATA &vtData = itToSend->second;
			VT_TRANS_DATA::iterator itData = vtData.begin();
			std::vector<WSABUF> vtBuf;
			for (; itData != vtData.end(); ++itData)
			{
				TransDataInfo *pData = *itData;
				CHECK_OP(pData, continue);

				if (!pData->bHasDeliver && !pData->bToFree)
				{
					CHECK_OP(!pData->bSendRecvFinish, continue);
					pData->bHasDeliver = TRUE;
					pInfo->vtDeliverData.push_back(pData);
					vtBuf.push_back(pData->buffer);
				}
			}

			if (!vtBuf.empty())
			{
				int nRet = WSASend(sock, &vtBuf[0], vtBuf.size(), NULL, 0, &pInfo->overlap, NULL);
				if (nRet == SOCKET_ERROR)
				{
					int nErr = WSAGetLastError();
					if (nErr != WSA_IO_PENDING)
					{
						OutputDebugStringA("CPluginNetwork::SendLoop - nErr != WSA_IO_PENDING !\n");
						WSASetEvent(pInfo->hEventHandle);
						vtSocketError.push_back(sock);
					}
				}
			}
		}

		//�ȴ��ź�
		std::vector<WSAEVENT> vtEvent;
		std::vector<SOCKET> vtSocket;
		itRTInfo = m_mapSendingInfo.begin();
		for (; itRTInfo != m_mapSendingInfo.end(); ++itRTInfo)
		{
			SOCKET sock = itRTInfo->first;
			SockRuntimeInfo *pInfo = itRTInfo->second;
			CHECK_OP(pInfo, continue);
			if (!pInfo->vtDeliverData.empty())
			{
				vtEvent.push_back(pInfo->hEventHandle);
				vtSocket.push_back(sock);
			}
		}
		lock.Unlock();

		if (vtEvent.empty())
		{
			continue;
		}

		dwNextWaitExitTime = 0;
		std::vector<SOCKET>  vtWaited; //��¼�ȴ�����õ�event 
		//ѭ��������event
		while (true)
		{
			CHECK_OP(vtEvent.size() == vtSocket.size(), NOOP);
			if (vtEvent.size() == 0)
				break;

			DWORD dwRet = WSAWaitForMultipleEvents((DWORD)vtEvent.size(), _vect2Ptr(vtEvent), FALSE, TIMEOUT_MILISECONDS, FALSE);
			if (dwRet == WSA_WAIT_TIMEOUT)
			{
				break;
			}
			else if (dwRet == WSA_WAIT_FAILED)
			{
				break;
			}
			//�л��߳�
			Sleep(0);

			//��������ɵ�����
			int nIndex = dwRet - WSA_WAIT_EVENT_0;
			CHECK_OP(nIndex >= 0 && nIndex < _vectIntSize(vtSocket), break);
			SOCKET sock = vtSocket[nIndex];
			WSAEVENT evt = vtEvent[nIndex];
			//ɾ����ǰ�������
			vtSocket.erase(vtSocket.begin() + nIndex);
			vtEvent.erase(vtEvent.begin() + nIndex);
			WSAResetEvent(evt);
			vtWaited.push_back(sock);

			lock.Lock();

			SockRuntimeInfo *pInfo = NULL;
			MAP_SOCK_RTINFO::iterator itFind = m_mapSendingInfo.find(sock);
			if (itFind != m_mapSendingInfo.end())
			{
				pInfo = itFind->second;
			}
			DWORD dwBytesTrans = 0;
			DWORD dwFlags = 0;
			bool bBadSocket = std::find(vtSocketError.begin(), vtSocketError.end(), sock) != vtSocketError.end();
			if (pInfo && !bBadSocket)
			{
				WSAGetOverlappedResult(sock, &pInfo->overlap, &dwBytesTrans, FALSE, &dwFlags);
			}
			if (dwBytesTrans == 0 || bBadSocket)
			{
				ClearSocketSendData(sock);
				lock.Unlock();

				CString strError;
				strError.Format(_T("SendLoop - BytesTrans=%d  badSocket=%d"), dwBytesTrans, (int)bBadSocket);
				NotifySocketClosed(sock, strError);
				continue;
			}

			VT_TRANS_DATA &vtFinishData = pInfo->vtDeliverData;
			VT_TRANS_DATA::iterator itFinishData = vtFinishData.begin();
			for (; itFinishData != vtFinishData.end(); ++itFinishData)
			{
				TransDataInfo *pData = *itFinishData;
				CHECK_OP(pData, NOOP);
				if (pData)
				{
					pData->bSendRecvFinish = TRUE;
					pData->bToFree = TRUE;
				}
			}
			vtFinishData.clear();

			//����֪ͨ		
			FreeSendFinishBuf(sock);
			lock.Unlock();

			if (m_pEvtNotify)
				m_pEvtNotify->OnSend(sock);
		} //while (true) events 

		//���û�еȴ�����event����Ƿ��Ѿ��ر�
		lock.Lock();
		std::vector<SOCKET> vtNotifyClose;
		MAP_SOCK_RTINFO::iterator itSendInfo = m_mapSendingInfo.begin();
		for (; itSendInfo != m_mapSendingInfo.end(); ++itSendInfo)
		{
			SOCKET sock = itSendInfo->first;
			std::vector<SOCKET>::iterator itSock = std::find(vtWaited.begin(), vtWaited.end(), sock);
			if (itSock == vtWaited.end())
			{
				WSAResetEvent(hEvtCheckClose);
				WSAEventSelect(sock, hEvtCheckClose, FD_CLOSE);
				DWORD dwRet = WaitForSingleObject(hEvtCheckClose, 0);
				if (WAIT_OBJECT_0 == dwRet)
				{
					vtNotifyClose.push_back(sock);
				}
			}
		}
		for (UINT i = 0; i < vtNotifyClose.size(); i++)
		{
			ClearSocketSendData(vtNotifyClose[i]);
		}
		lock.Unlock();

		for (UINT i = 0; i < vtNotifyClose.size(); i++)
		{
			NotifySocketClosed(vtNotifyClose[i], L" SendLoop Check Close Error -  event not waited");
		}

	}//while (true) begin
}

DWORD WINAPI CPluginNetwork::ThreadRecv(LPVOID lParam)
{
	CPluginNetwork *pThis = (CPluginNetwork*)lParam;
	CHECK_RET(pThis, 0);
	pThis->RecvLoop();
	return 0;
}

void CPluginNetwork::RecvLoop()
{
	DWORD dwNextWaitExitTime = 0;
	WSAEVENT  hEvtCheckClose = WSACreateEvent();
	while (true)
	{
		if (WaitForSingleObject(m_hEvtNotifyExit, dwNextWaitExitTime) == WAIT_OBJECT_0)
		{
			CloseHandle(m_hThreadRecv);
			m_hThreadRecv = NULL;
			return;
		}
		//�л��߳�
		Sleep(0);
		dwNextWaitExitTime = TIMEOUT_MILISECONDS;

		//Ͷ���µĽ�������
		CSingleLock lock(&m_csRecv, TRUE);
		MAP_SOCK_RTINFO::iterator itRTInfo = m_mapRecvingInfo.begin();
		std::vector<SOCKET> vtSocketError;
		for (; itRTInfo != m_mapRecvingInfo.end(); ++itRTInfo)
		{
			SOCKET sock = itRTInfo->first;
			SockRuntimeInfo *pInfo = itRTInfo->second;
			CHECK_OP(pInfo, continue);

			if (!pInfo->vtDeliverData.empty())
				continue;

			TransDataInfo *pData = new TransDataInfo;
			char *pBuf = new char[MIN_BUF_ALLOC_SIZE];
			if (pData == NULL || pBuf == NULL)
			{
				delete pData;
				delete[] pBuf;
				theLog << "CPluginNetwork::RecvLoop - new TransDataInfo Error" << endLog;
				continue;
			}

			pInfo->vtDeliverData.push_back(pData);
			m_mapRecvedData[sock].push_back(pData);

			ZeroMemory(pBuf, MIN_BUF_ALLOC_SIZE);
			pData->nBufferID = m_nNextRecvBufID++;
			pData->nRealBufLen = MIN_BUF_ALLOC_SIZE;
			pData->buffer.len = MIN_BUF_ALLOC_SIZE;
			pData->buffer.buf = pBuf;
			pData->bHasDeliver = TRUE;

			ZeroMemory(&pInfo->overlap, sizeof(pInfo->overlap));
			pInfo->overlap.hEvent = pInfo->hEventHandle;
			WSAResetEvent(pInfo->hEventHandle);

			DWORD dwFlags = 0;
			DWORD dwRecvBytes = 0;
			int nRet = WSARecv(sock, &pData->buffer, 1, NULL/*&dwRecvBytes*/, &dwFlags, &pInfo->overlap, NULL);
			int nErr = WSAGetLastError();
			if (nRet == SOCKET_ERROR)
			{
				if (nErr != WSA_IO_PENDING)
				{
					OutputDebugStringA("CPluginNetwork::RecvLoop -  nErr != WSA_IO_PENDING! \n");
					WSASetEvent(pInfo->hEventHandle);
					vtSocketError.push_back(sock);
				}
			}
		}

		//�ȴ��ź�
		std::vector<WSAEVENT> vtEvent;
		std::vector<SOCKET> vtSocket;
		itRTInfo = m_mapRecvingInfo.begin();
		for (; itRTInfo != m_mapRecvingInfo.end(); ++itRTInfo)
		{
			SOCKET sock = itRTInfo->first;
			SockRuntimeInfo *pInfo = itRTInfo->second;
			CHECK_OP(pInfo, continue);
			if (!pInfo->vtDeliverData.empty())
			{
				vtEvent.push_back(pInfo->hEventHandle);
				vtSocket.push_back(sock);
			}
		}
		lock.Unlock();

		if (vtEvent.empty())
		{
			continue;
		}
		dwNextWaitExitTime = 0;
		std::vector<SOCKET>  vtWaited; //��¼�ȴ�����õ�event 
		//ѭ��������event
		while (true)
		{
			if (vtEvent.size() == 0)
				break;

			CHECK_OP(vtEvent.size() == vtSocket.size(), NOOP);
			DWORD dwRet = WSAWaitForMultipleEvents((DWORD)vtEvent.size(), _vect2Ptr(vtEvent), FALSE, TIMEOUT_MILISECONDS, FALSE);
			if (dwRet == WSA_WAIT_TIMEOUT)
			{
				break;
			}
			else if (dwRet == WSA_WAIT_FAILED)
			{
				break;
			}
			//�л��߳�
			Sleep(0);

			//������յ���
			int nIndex = dwRet - WSA_WAIT_EVENT_0;
			CHECK_OP(nIndex >= 0 && nIndex < _vectIntSize(vtSocket), break);
			SOCKET sock = vtSocket[nIndex];
			WSAEVENT evt = vtEvent[nIndex];
			//ɾ����ǰ�������
			vtSocket.erase(vtSocket.begin() + nIndex);
			vtEvent.erase(vtEvent.begin() + nIndex);
			WSAResetEvent(evt);
			vtWaited.push_back(sock);

			lock.Lock();

			SockRuntimeInfo *pInfo = NULL;
			MAP_SOCK_RTINFO::iterator itFind = m_mapRecvingInfo.find(sock);
			if (itFind != m_mapRecvingInfo.end())
			{
				pInfo = itFind->second;
			}

			DWORD dwBytesTrans = 0;
			DWORD dwFlags = 0;
			bool bBadSock = (std::find(vtSocketError.begin(), vtSocketError.end(), sock) != vtSocketError.end());
			if (pInfo && !bBadSock)
			{
				WSAGetOverlappedResult(sock, &pInfo->overlap, &dwBytesTrans, FALSE, &dwFlags);
			}
			int nErr = WSAGetLastError();
			if (dwBytesTrans == 0 || bBadSock)
			{
				ClearSocketRecvData(sock);
				pInfo = NULL;
				lock.Unlock();

				CString strError;
				strError.Format(_T("RecvLoop - BytesTrans=%d  badSocket=%d"), dwBytesTrans, (int)bBadSock);
				NotifySocketClosed(sock, strError);
				continue;
			}

			//���Ͽ�
			WSAResetEvent(pInfo->hEventHandle);
			WSAEventSelect(sock, pInfo->hEventHandle, FD_CLOSE);
			DWORD dwCloseRet = WaitForSingleObject(pInfo->hEventHandle, 0);
			bool bNoitfySocketClose = false;
			if (dwCloseRet == WAIT_OBJECT_0)
			{
				ClearSocketRecvData(sock);

				//ClearSocketRecvData���ú�, pInfoָ���Ѿ�Ұ����
				pInfo = NULL;
				bNoitfySocketClose = true;  //close�漰���߳�ͬ��֪ͨ���������ִ�бȽϺ�
			}

			//�����õ����ݣ���־����״̬
			if (pInfo)
			{
				VT_TRANS_DATA &vtFinishData = pInfo->vtDeliverData;
				if (vtFinishData.size() != 1)
				{
					OutputDebugStringA("PluginNetork - Check Err vtFinishData.size() !\n");
					theLog << "CPluginNetwork::RecvLoop - Error vtDeliverData, socket=" << (UINT32)sock << " size=" << pInfo->vtDeliverData.size() << endLog;
				}
				if (!vtFinishData.empty())
				{
					TransDataInfo *pData = vtFinishData[0];
					if (pData)
					{
						pData->bSendRecvFinish = TRUE;
						pData->buffer.len = dwBytesTrans;
					}
					vtFinishData.clear();
				}
			}
			//////////////////////////////////////////////////////////////////////////

			lock.Unlock();

			if (bNoitfySocketClose)
			{
				//Notify�漰���߳�ͬ��֪ͨ 
				NotifySocketClosed(sock, L" RecvLoop Check Close Error -  event waited");
			}
			else
			{
				if (m_pEvtNotify)
					m_pEvtNotify->OnReceive(sock);
			}
		} //while (true) event

		//���û�еȴ�����event����Ƿ��Ѿ��ر�
		lock.Lock();
		std::vector<SOCKET> vtNotifyClose;
		MAP_SOCK_RTINFO::iterator itRecvInfo = m_mapRecvingInfo.begin();
		for (; itRecvInfo != m_mapRecvingInfo.end(); ++itRecvInfo)
		{
			SOCKET sock = itRecvInfo->first;
			std::vector<SOCKET>::iterator itSock = std::find(vtWaited.begin(), vtWaited.end(), sock);
			if (itSock == vtWaited.end())
			{
				WSAResetEvent(hEvtCheckClose);
				WSAEventSelect(sock, hEvtCheckClose, FD_CLOSE);
				DWORD dwRet = WaitForSingleObject(hEvtCheckClose, 0);
				if (WAIT_OBJECT_0 == dwRet)
				{
					vtNotifyClose.push_back(sock);
				}
			}
		}
		for (UINT i = 0; i < vtNotifyClose.size(); i++)
		{
			ClearSocketRecvData(vtNotifyClose[i]);
		}
		lock.Unlock();

		for (UINT i = 0; i < vtNotifyClose.size(); i++)
		{
			NotifySocketClosed(vtNotifyClose[i], L" RecvLoop Check Close Error -  event not waited");
		}
	}// while (true) - begin 

	WSACloseEvent(hEvtCheckClose);
	hEvtCheckClose = NULL;
}