/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE78_OS_Command_Injection__char_connect_socket_execlp_66b.c
Label Definition File: CWE78_OS_Command_Injection.no_path.label.xml
Template File: sources-sink-66b.tmpl.c
*/
/*
 * @description
 * CWE: 78 OS Command Injection
 * BadSource: connect_socket Read data using a connect socket (client side)
 * GoodSource: Benign input
 * Sinks: execlp
 *    BadSink : execute command with execlp
 * Flow Variant: 66 Data flow: data passed in an array from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifdef _WIN32
# define COMMAND_INT_PATH "%WINDIR%\\system32\\cmd.exe"
# define COMMAND_INT "cmd.exe"
# define COMMAND_ARG1 "/c"
# define COMMAND_ARG2 "dir"
# define COMMAND_ARG3 data
#else /* NOT _WIN32 */
# define COMMAND_INT_PATH "/bin/sh"
# define COMMAND_INT "sh"
# define COMMAND_ARG1 "ls"
# define COMMAND_ARG2 data
# define COMMAND_ARG3 NULL
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

#ifdef _WIN32
#include <process.h>
# define EXECLP _execlp
#else /* NOT _WIN32 */
# define EXECLP execlp
#endif

#ifndef OMITBAD

void CWE78_OS_Command_Injection__char_connect_socket_execlp_66b_bad_sink(char * data_array[])
{
    /* copy data out of data_array */
    char * data = data_array[2];
    /* execlp - searches for the location of the command among
     * the directories specified by the PATH environment variable */
    /* POSSIBLE FLAW: Execute command without validating input possibly leading to command injection */
    EXECLP(COMMAND_INT, COMMAND_INT, COMMAND_ARG1, COMMAND_ARG2, COMMAND_ARG3, NULL);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE78_OS_Command_Injection__char_connect_socket_execlp_66b_goodG2B_sink(char * data_array[])
{
    char * data = data_array[2];
    /* execlp - searches for the location of the command among
     * the directories specified by the PATH environment variable */
    /* POSSIBLE FLAW: Execute command without validating input possibly leading to command injection */
    EXECLP(COMMAND_INT, COMMAND_INT, COMMAND_ARG1, COMMAND_ARG2, COMMAND_ARG3, NULL);
}

#endif /* OMITGOOD */