#pragma once

class UsTimezone
{
public:
	UsTimezone();
	virtual ~UsTimezone();
	
	static int GetTimestampTimezone(int nTimestamp);

	//nYear, nMonth, nDay: �ճ���֪��������
	static int GetTMStructTimezone(int nYear, int nMonth, int nDay);

private:
};