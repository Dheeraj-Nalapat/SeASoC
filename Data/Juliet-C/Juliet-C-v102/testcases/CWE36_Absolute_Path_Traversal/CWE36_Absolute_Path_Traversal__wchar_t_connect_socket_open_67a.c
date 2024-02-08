/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE36_Absolute_Path_Traversal__wchar_t_connect_socket_open_67a.c
Label Definition File: CWE36_Absolute_Path_Traversal.label.xml
Template File: sources-sink-67a.tmpl.c
*/
/*
 * @description
 * CWE: 36 Absolute Path Traversal
 * BadSource: connect_socket Read data using a connect socket (client side)
 * GoodSource: Full path and file name
 * Sinks: open
 *    BadSink :
 * Flow Variant: 67 Data flow: data passed in a struct from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#ifdef _WIN32
# include <winsock2.h>
# include <windows.h>
# include <direct.h>
# pragma comment(lib, "ws2_32") /* include ws2_32.lib when linking */
# define CLOSE_SOCKET closesocket
# define PATH_SZ FILENAME_MAX
#else /* NOT _WIN32 */
# define INVALID_SOCKET -1
# define SOCKET_ERROR -1
# define CLOSE_SOCKET close
# define SOCKET int
# define PATH_SZ PATH_MAX
#endif

#define TCP_PORT 27015

#ifdef _WIN32
# define OPEN _wopen
# define CLOSE _close
#else
# define OPEN wopen
# define CLOSE close
#endif

typedef struct _CWE36_Absolute_Path_Traversal__wchar_t_connect_socket_open_67_struct_type
{
    wchar_t * a;
} CWE36_Absolute_Path_Traversal__wchar_t_connect_socket_open_67_struct_type;

#ifndef OMITBAD

/* bad function declaration */
void CWE36_Absolute_Path_Traversal__wchar_t_connect_socket_open_67b_bad_sink(CWE36_Absolute_Path_Traversal__wchar_t_connect_socket_open_67_struct_type my_struct);

void CWE36_Absolute_Path_Traversal__wchar_t_connect_socket_open_67_bad()
{
    wchar_t * data;
    CWE36_Absolute_Path_Traversal__wchar_t_connect_socket_open_67_struct_type my_struct;
    wchar_t data_buf[FILENAME_MAX] = L"";
    data = data_buf;
    {
#ifdef _WIN32
        WSADATA wsa_data;
        int wsa_data_init = 0;
#endif
        int recv_rv;
        struct sockaddr_in s_in;
        wchar_t *replace;
        SOCKET connect_socket = INVALID_SOCKET;
        size_t data_len = wcslen(data);
        do
        {
#ifdef _WIN32
            if (WSAStartup(MAKEWORD(2,2), &wsa_data) != NO_ERROR) break;
            wsa_data_init = 1;
#endif
            connect_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
            if (connect_socket == INVALID_SOCKET) break;
            memset(&s_in, 0, sizeof(s_in));
            s_in.sin_family = AF_INET;
            s_in.sin_addr.s_addr = inet_addr("127.0.0.1");
            s_in.sin_port = htons(TCP_PORT);
            if (connect(connect_socket, (struct sockaddr*)&s_in, sizeof(s_in)) == SOCKET_ERROR) break;
            /* Abort on error or the connection was closed, make sure to recv one
             * less char than is in the recv_buf in order to append a terminator */
            recv_rv = recv(connect_socket, (char *)data+data_len, (int)(FILENAME_MAX-data_len-1), 0);
            if (recv_rv == SOCKET_ERROR || recv_rv == 0) break;
            /* Append null terminator */
            data[recv_rv] = L'\0';
            /* Eliminate CRLF */
            replace = wcschr(data, L'\r');
            if (replace) *replace = L'\0';
            replace = wcschr(data, L'\n');
            if (replace) *replace = L'\0';
        }
        while (0);
        if (connect_socket != INVALID_SOCKET) CLOSE_SOCKET(connect_socket);
#ifdef _WIN32
        if (wsa_data_init) WSACleanup();
#endif
    }
    my_struct.a = data;
    CWE36_Absolute_Path_Traversal__wchar_t_connect_socket_open_67b_bad_sink(my_struct);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE36_Absolute_Path_Traversal__wchar_t_connect_socket_open_67b_goodG2B_sink(CWE36_Absolute_Path_Traversal__wchar_t_connect_socket_open_67_struct_type my_struct);

static void goodG2B()
{
    wchar_t * data;
    CWE36_Absolute_Path_Traversal__wchar_t_connect_socket_open_67_struct_type my_struct;
    wchar_t data_buf[FILENAME_MAX] = L"";
    data = data_buf;
#ifdef _WIN32
    /* FIX: Full path and file name */
    wcscpy(data, L"c:\\temp\\file.txt");
#else
    /* FIX: Full path and file name */
    wcscpy(data, L"/tmp/file.txt");
#endif
    my_struct.a = data;
    CWE36_Absolute_Path_Traversal__wchar_t_connect_socket_open_67b_goodG2B_sink(my_struct);
}

void CWE36_Absolute_Path_Traversal__wchar_t_connect_socket_open_67_good()
{
    goodG2B();
}

#endif /* OMITGOOD */

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */

#ifdef INCLUDEMAIN

int main(int argc, char * argv[])
{
    /* seed randomness */
    srand( (unsigned)time(NULL) );
#ifndef OMITGOOD
    printLine("Calling good()...");
    CWE36_Absolute_Path_Traversal__wchar_t_connect_socket_open_67_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE36_Absolute_Path_Traversal__wchar_t_connect_socket_open_67_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
