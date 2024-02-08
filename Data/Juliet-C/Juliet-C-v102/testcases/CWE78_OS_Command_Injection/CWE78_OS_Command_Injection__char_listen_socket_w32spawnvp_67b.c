/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE78_OS_Command_Injection__char_listen_socket_w32spawnvp_67b.c
Label Definition File: CWE78_OS_Command_Injection.no_path.label.xml
Template File: sources-sink-67b.tmpl.c
*/
/*
 * @description
 * CWE: 78 OS Command Injection
 * BadSource: listen_socket Read data using a listen socket (server side)
 * GoodSource: Benign input
 * Sinks: w32spawnvp
 *    BadSink : execute command with spawnvp
 * Flow Variant: 67 Data flow: data passed in a struct from one function to another in different source files
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
# define PATH_SZ 100
# pragma comment(lib, "ws2_32") /* include ws2_32.lib when linking */
# define CLOSE_SOCKET closesocket
#else
# define PATH_SZ PATH_MAX
# define INVALID_SOCKET -1
# define SOCKET_ERROR -1
# define CLOSE_SOCKET close
# define SOCKET int
#endif

#define TCP_PORT 27015
#define LISTEN_BACKLOG 5

#include <process.h>

typedef struct _CWE78_OS_Command_Injection__char_listen_socket_w32spawnvp_67_struct_type
{
    char * a;
} CWE78_OS_Command_Injection__char_listen_socket_w32spawnvp_67_struct_type;

#ifndef OMITBAD

void CWE78_OS_Command_Injection__char_listen_socket_w32spawnvp_67b_bad_sink(CWE78_OS_Command_Injection__char_listen_socket_w32spawnvp_67_struct_type my_struct)
{
    char * data = my_struct.a;
    {
        char *args[] = {COMMAND_INT_PATH, COMMAND_ARG1, COMMAND_ARG2, COMMAND_ARG3, NULL};
        /* spawnvp - searches for the location of the command among
         * the directories specified by the PATH environment variable */
        /* POSSIBLE FLAW: Execute command without validating input possibly leading to command injection */
        _spawnvp(_P_WAIT, COMMAND_INT, args);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE78_OS_Command_Injection__char_listen_socket_w32spawnvp_67b_goodG2B_sink(CWE78_OS_Command_Injection__char_listen_socket_w32spawnvp_67_struct_type my_struct)
{
    char * data = my_struct.a;
    {
        char *args[] = {COMMAND_INT_PATH, COMMAND_ARG1, COMMAND_ARG2, COMMAND_ARG3, NULL};
        /* spawnvp - searches for the location of the command among
         * the directories specified by the PATH environment variable */
        /* POSSIBLE FLAW: Execute command without validating input possibly leading to command injection */
        _spawnvp(_P_WAIT, COMMAND_INT, args);
    }
}

#endif /* OMITGOOD */
