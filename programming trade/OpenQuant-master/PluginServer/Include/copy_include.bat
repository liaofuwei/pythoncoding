@echo off
set workdir=%~dp0
set workdir=%workdir:~0,-1%
echo ��ǰ����Ŀ¼��%workdir%

copy  "..\..\..\FTPlugin\FTPlugin\FTPluginCore.h"  "./"
copy  "..\..\..\FTPlugin\FTPlugin\FTPluginQuoteDefine.h"  "./"
copy  "..\..\..\FTPlugin\FTPlugin\FTPluginQuoteInterface.h"  "./"
copy  "..\..\..\FTPlugin\FTPlugin\FTPluginTradeInterface.h"  "./"


