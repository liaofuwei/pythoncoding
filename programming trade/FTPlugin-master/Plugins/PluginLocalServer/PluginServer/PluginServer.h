// PluginServer.h : PluginServer DLL ����ͷ�ļ�
//

#pragma once

#ifndef __AFXWIN_H__
	#error "�ڰ������ļ�֮ǰ������stdafx.h�������� PCH �ļ�"
#endif

#include "resource.h"		// ������


// CPluginServerApp
// �йش���ʵ�ֵ���Ϣ������� PluginServer.cpp
//

class CPluginServerApp : public CWinApp
{
public:
	CPluginServerApp();

// ��д
public:
	virtual BOOL InitInstance();
	virtual int ExitInstance();

	DECLARE_MESSAGE_MAP()
};
