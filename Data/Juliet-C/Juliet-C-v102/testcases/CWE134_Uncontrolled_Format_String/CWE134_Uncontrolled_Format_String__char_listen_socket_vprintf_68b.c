/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE134_Uncontrolled_Format_String__char_listen_socket_vprintf_68b.c
Label Definition File: CWE134_Uncontrolled_Format_String.vasinks.label.xml
Template File: sources-vasinks-68b.tmpl.c
*/
/*
 * @description
 * CWE: 134 Uncontrolled Format String
 * BadSource: listen_socket Read data using a listen socket (server side)
 * GoodSource: Copy a fixed string into data
 * Sinks: vprintf
 *    GoodSink: vprintf with a format string
 *    BadSink : vprintf without a format string
 * Flow Variant: 68 Data flow: data passed as a global variable from one function to another in different source files
 *
 * */

#include <stdarg.h>
#include "std_testcase.h"

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

extern char * CWE134_Uncontrolled_Format_String__char_listen_socket_vprintf_68_bad_data;
extern char * CWE134_Uncontrolled_Format_String__char_listen_socket_vprintf_68_goodG2B_data;
extern char * CWE134_Uncontrolled_Format_String__char_listen_socket_vprintf_68_goodB2G_data;

#ifndef OMITBAD

static void bad_vasink(char * data, ...)
{
    {
        va_list args;
        va_start(args, data);
        /* POTENTIAL FLAW: Do not specify the format allowing a possible format string vulnerability */
        vprintf(data, args);
        va_end(args);
    }
}

void CWE134_Uncontrolled_Format_String__char_listen_socket_vprintf_68b_bad_sink()
{
    char * data = CWE134_Uncontrolled_Format_String__char_listen_socket_vprintf_68_bad_data;
    bad_vasink(data, data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
static void goodG2B_vasink(char * data, ...)
{
    {
        va_list args;
        va_start(args, data);
        /* POTENTIAL FLAW: Do not specify the format allowing a possible format string vulnerability */
        vprintf(data, args);
        va_end(args);
    }
}

void CWE134_Uncontrolled_Format_String__char_listen_socket_vprintf_68b_goodG2B_sink()
{
    char * data = CWE134_Uncontrolled_Format_String__char_listen_socket_vprintf_68_goodG2B_data;
    goodG2B_vasink(data, data);
}

/* goodB2G uses the BadSource with the GoodSink */
static void goodB2G_vasink(char * data, ...)
{
    {
        va_list args;
        va_start(args, data);
        /* FIX: Specify the format disallowing a format string vulnerability */
        vprintf("%s", args);
        va_end(args);
    }
}

void CWE134_Uncontrolled_Format_String__char_listen_socket_vprintf_68b_goodB2G_sink()
{
    char * data = CWE134_Uncontrolled_Format_String__char_listen_socket_vprintf_68_goodB2G_data;
    goodB2G_vasink(data, data);
}

#endif /* OMITGOOD */
