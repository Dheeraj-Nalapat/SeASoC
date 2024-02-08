/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE78_OS_Command_Injection__char_connect_socket_popen_68b.c
Label Definition File: CWE78_OS_Command_Injection.fullpath.label.xml
Template File: sources-sink-68b.tmpl.c
*/
/*
 * @description
 * CWE: 78 OS Command Injection
 * BadSource: connect_socket Read data using a connect socket (client side)
 * GoodSource: Benign input
 * Sink: popen
 *    BadSink : Execute command using popen()
 * Flow Variant: 68 Data flow: data passed as a global variable from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifdef _WIN32
# define COMMAND_INT_PATH "%WINDIR%\\system32\\cmd.exe"
# define COMMAND_INT "cmd.exe"
# define COMMAND_ARG1 "/c"
# define COMMAND_ARG2 "dir"
# define FULL_COMMAND COMMAND_INT_PATH " " COMMAND_ARG1 " "COMMAND_ARG2 " "
#else /* NOT _WIN32 */
# define COMMAND_INT_PATH "/bin/sh"
# define COMMAND_INT "sh"
# define COMMAND_ARG1 "ls"
# define FULL_COMMAND COMMAND_INT_PATH " " COMMAND_ARG1 " "
#endif

#ifdef _WIN32
# include <winsock2.h>
# include <windows.h>
# include <direct.h>
# pragma comment(lib, "ws2_32") /* include ws2_32.lib when linking */
# define CLOSE_SOCKET closesocket
# define PATH_SZ 100
#else /* NOT _WIN32 */
# define INVALID_SOCKET -1
# define SOCKET_ERROR -1
# define CLOSE_SOCKET close
# define SOCKET int
# define PATH_SZ PATH_MAX
#endif

#define TCP_PORT 27015

/* define POPEN as _popen on Windows and popen otherwise */
#ifdef _WIN32
# define POPEN _popen
# define PCLOSE _pclose
#else /* NOT _WIN32 */
# define POPEN popen
# define PCLOSE pclose
#endif

extern char * CWE78_OS_Command_Injection__char_connect_socket_popen_68_bad_data;
extern char * CWE78_OS_Command_Injection__char_connect_socket_popen_68_goodG2B_data;

/* all the sinks are the same, we just want to know where the hit originated if a tool flags one */

#ifndef OMITBAD

void CWE78_OS_Command_Injection__char_connect_socket_popen_68b_bad_sink()
{
    char * data = CWE78_OS_Command_Injection__char_connect_socket_popen_68_bad_data;
    {
        FILE *pipe;
        /* POSSIBLE FLAW: Execute command without validating input possibly leading to command injection */
        pipe = POPEN(data, "wb");
        if (pipe != NULL) PCLOSE(pipe);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE78_OS_Command_Injection__char_connect_socket_popen_68b_goodG2B_sink()
{
    char * data = CWE78_OS_Command_Injection__char_connect_socket_popen_68_goodG2B_data;
    {
        FILE *pipe;
        /* POSSIBLE FLAW: Execute command without validating input possibly leading to command injection */
        pipe = POPEN(data, "wb");
        if (pipe != NULL) PCLOSE(pipe);
    }
}

#endif /* OMITGOOD */
